#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss

from transformers import AutoConfig, AutoModelForCausalLM, LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput
from transformers import AutoModelForCausalLM
from transformers import AutoProcessor
import deepspeed

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
from ..multimodal_encoder.builder import build_vision_tower
from ..multimodal_resampler.builder import build_vision_resampler
from ..multimodal_projector.builder import build_vision_projector
from mu.mu_vision.utils import rank0_print, rank_print
from .helpers import extend_instance, VisionCrossAttnMixin


def stack_tensors_with_mask(tensor_list):
    # Determine the maximum length along the first dimension
    max_length = max(tensor.size(0) for tensor in tensor_list)

    # Initialize a list to store padded tensors and their masks
    padded_tensors = []
    padding_masks = []

    for tensor in tensor_list:
        # Calculate the padding size
        padding_size = max_length - tensor.size(0)

        # Pad the tensor
        padded_tensor = torch.nn.functional.pad(tensor, (0, 0, 0, padding_size))
        padded_tensors.append(padded_tensor)

        # Create the padding mask
        mask = torch.cat([torch.ones(tensor.size(0)), torch.zeros(padding_size)], dim=0).to(device=padded_tensor.device)
        padding_masks.append(mask)

    # Stack the padded tensors and masks along a new dimension
    stacked_tensor = torch.stack(padded_tensors)
    padding_mask = torch.stack(padding_masks)

    return stacked_tensor, padding_mask


def soft_cross_entropy(predicts, targets, temperature=1, mask=None):
    student_likelihood = torch.nn.functional.log_softmax(predicts / temperature, dim=-1)
    targets_prob = torch.nn.functional.softmax(targets / temperature, dim=-1)
    if mask is None:
        loss_per_loc_flat = (temperature**2) * (-targets_prob * student_likelihood).sum(dim=-1)
        return loss_per_loc_flat.mean()
    else:
        loss_per_loc_flat = (temperature**2) * (-targets_prob * student_likelihood).sum(dim=-1).view(-1)
        mask_flat = mask.view(-1)
        loss_per_loc_selected = loss_per_loc_flat[mask_flat]
        return loss_per_loc_selected.mean()


class GatingTower(nn.Module):
    def __init__(self):
        super(GatingTower, self).__init__()
        self.hidden_size = 128
        self.mm_gating_tower = nn.Sequential(
            nn.Linear(336 * 336, self.hidden_size), nn.GELU(), nn.Linear(self.hidden_size, 1), nn.Sigmoid()
        )

    def forward(self, images):
        N, C, _, _ = images.shape
        gray_images = images.mean(dim=1)
        flat_images = gray_images.reshape(N, -1)
        scores = self.mm_gating_tower(flat_images)
        return scores


def get_llava_mu_class(mu_version="mu_v2"):
    if mu_version == "mu_v2":
        from .mu_v2.modeling_mu import Mu, MuConfig
    else:
        raise ValueError("Do not support Mu version: {}".format(mu_version))

    class LlavaMuConfig(MuConfig):
        model_type = "llava_mu"

    class LlavaMuForCausalLM(Mu, LlavaMetaForCausalLM):
        config_class = LlavaMuConfig
        supports_gradient_checkpointing = True

        def __init__(self, config):
            super().__init__(config)

            if hasattr(config, "mm_vision_tower"):
                self.vision_tower = build_vision_tower(config, delay_load=False)
                self.vision_resampler = build_vision_resampler(config, vision_tower=self.vision_tower)
                self.mm_projector = build_vision_projector(config)

                if "unpad" in getattr(config, "mm_patch_merge_type", ""):
                    self.image_newline = nn.Parameter(torch.empty(config.hidden_size, dtype=self.dtype))
            if "gating" in getattr(config, "image_aspect_ratio", ""):
                self.mm_gating_tower = GatingTower()
            if getattr(config, "cross_image_pos_embed", False):
                self.cross_image_pos_embeddings = nn.Embedding(32, 1024)  # self.vision_tower.config.hidden_size

            if getattr(config, "use_vision_cross_attn", False):
                self.set_vision_cross_attn(config)

            self.use_seq2seq = True
            self.loss_mse = MSELoss()

        def set_tokenizer(self, tokenizer):
            self.tokenizer = tokenizer

        def get_input_embeddings(self):
            return self.embed_tokens

        @property
        def embed_tokens(self):
            return self.transformer.wte

        def get_vision_tower(self):
            vision_tower = getattr(self, "vision_tower", None)
            if type(vision_tower) is list:
                vision_tower = vision_tower[0]
            return vision_tower

        def initialize_gating_tower(self):
            if "gating" in getattr(self.config, "image_aspect_ratio", ""):
                if getattr(self, "mm_gating_tower", None) is None:
                    self.mm_gating_tower = GatingTower()

        def initialize_vision_modules(self, model_args, fsdp=None):
            vision_tower = model_args.vision_tower
            mm_vision_select_layer = model_args.mm_vision_select_layer
            mm_vision_select_feature = model_args.mm_vision_select_feature
            pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
            mm_patch_merge_type = model_args.mm_patch_merge_type

            self.config.mm_vision_tower = vision_tower
            self.config.vision_tower_pretrained = getattr(model_args, "vision_tower_pretrained", "")

            if self.get_vision_tower() is None:
                vision_tower = build_vision_tower(model_args)
                vision_resampler = build_vision_resampler(model_args, vision_tower=vision_tower)
                for k, v in vision_resampler.config.items():
                    setattr(self.config, k, v)

                if fsdp is not None and len(fsdp) > 0:
                    self.vision_tower = [vision_tower]
                    self.vision_resampler = [vision_resampler]
                else:
                    self.vision_tower = vision_tower
                    self.vision_resampler = vision_resampler
            else:
                if fsdp is not None and len(fsdp) > 0:
                    vision_resampler = self.vision_resampler[0]
                    vision_tower = self.vision_tower[0]
                else:
                    vision_resampler = self.vision_resampler
                    vision_tower = self.vision_tower
                vision_tower.load_model()

                # In case it is frozen by LoRA
                for p in self.vision_resampler.parameters():
                    p.requires_grad = True

            self.config.use_mm_proj = True
            self.config.mm_projector_type = getattr(model_args, "mm_projector_type", "linear")
            self.config.mm_hidden_size = getattr(vision_resampler, "hidden_size", vision_tower.hidden_size)
            self.config.mm_vision_select_layer = mm_vision_select_layer
            self.config.mm_vision_select_feature = mm_vision_select_feature
            self.config.mm_patch_merge_type = mm_patch_merge_type
            self.config.mm_gating_tower = getattr(model_args, "image_aspect_ratio", "")
            self.config.cross_image_pos_embed = getattr(model_args, "cross_image_pos_embed", False)
            self.config.use_vision_cross_attn = getattr(model_args, "use_vision_cross_attn", False)
            self.config.cross_attn_every_n_layers = getattr(model_args, "cross_attn_every_n_layers", 1)
            self.config.use_vision_cross_attn_pass_vision_token_to_encoder = getattr(
                model_args, "use_vision_cross_attn_pass_vision_token_to_encoder", False
            )
            self.config.use_vision_cross_which_tower = getattr(model_args, "use_vision_cross_which_tower", "encoder")
            self.config.use_knowledge_distillation = getattr(model_args, "use_knowledge_distillation", False)
            self.config.kd_loss_coef = getattr(model_args, "kd_loss_coef", 1.0)
            self.config.teacher_temperature = getattr(model_args, "teacher_temperature", 1.0)
            self.config.kd_one_word_prompt = getattr(model_args, "kd_one_word_prompt", False)
            self.config.kd_loss = getattr(model_args, "kd_loss", "soft_cross_entropy")
            self.config.kd_debug = True  # getattr(model_args, "kd_debug", False)

            if not hasattr(self.config, "add_faster_video"):
                if model_args.add_faster_video:
                    embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                    self.faster_token = nn.Parameter(torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std)

            if getattr(self, "mm_projector", None) is None:
                self.mm_projector = build_vision_projector(self.config, vision_cfg=vision_tower.config)

                if "unpad" in mm_patch_merge_type:
                    embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                    self.image_newline = nn.Parameter(
                        torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std
                    )
            else:
                # In case it is frozen by LoRA
                for p in self.mm_projector.parameters():
                    p.requires_grad = True

            if getattr(model_args, "cross_image_pos_embed", False):
                if getattr(self, "cross_image_pos_embeddings", None) is None:
                    self.cross_image_pos_embeddings = nn.Embedding(32, 1024)  # self.vision_tower.config.hidden_size

            if getattr(model_args, "use_vision_cross_attn", False):
                self.set_vision_cross_attn(self.config)

            if pretrain_mm_mlp_adapter is not None:
                mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location="cpu")

                def get_w(weights, keyword):
                    return {k.split(keyword + ".")[1]: v for k, v in weights.items() if keyword in k}

                incompatible_keys = self.mm_projector.load_state_dict(get_w(mm_projector_weights, "mm_projector"))
                rank0_print(
                    f"Loaded mm projector weights from {pretrain_mm_mlp_adapter}. Incompatible keys: {incompatible_keys}"
                )
                incompatible_keys = self.vision_resampler.load_state_dict(
                    get_w(mm_projector_weights, "vision_resampler"), strict=False
                )
                rank0_print(
                    f"Loaded vision resampler weights from {pretrain_mm_mlp_adapter}. Incompatible keys: {incompatible_keys}"
                )

        def get_model(self):
            return self

        @property
        def model(self):
            return self

        def set_vision_cross_attn(self, config):
            extend_instance(self, VisionCrossAttnMixin)
            if self.config.use_vision_cross_which_tower == "encoder":
                self.set_decoder_layers_attr_name("transformer.e")  # encoder
            elif self.config.use_vision_cross_which_tower == "decoder":
                self.set_decoder_layers_attr_name("transformer.h")  # decoder
            else:
                raise ValueError(
                    "cannot apply vision cross attn to tower {}".format(self.config.use_vision_cross_which_tower)
                )
            self.init_flamingo(
                media_token_id=0,
                lang_hidden_size=config.n_embd,
                vis_hidden_size=config.n_embd,
                cross_attn_every_n_layers=getattr(config, "cross_attn_every_n_layers", 1),
                gradient_checkpointing=False,
                use_vision_cross_which_tower=self.config.use_vision_cross_which_tower,
            )
            if getattr(self, "tokenizer", None) is not None:
                if (
                    getattr(config, "use_vision_cross_attn", False)
                    and "<image_tkn>" not in self.tokenizer.all_special_tokens
                ):
                    self.tokenizer.add_special_tokens({"additional_special_tokens": ["<image_tkn>"]})

        def condition_vision_x(self, image_features):
            stacked_image_features, stack_padding_mask = stack_tensors_with_mask(image_features)
            stack_padding_mask = stack_padding_mask.unsqueeze(1).unsqueeze(2)
            for layer in self._get_decoder_layers():
                layer.condition_vis_x(stacked_image_features, vis_attn_mask=stack_padding_mask)

        def pad_sequence(self, input_ids, batch_first, padding_value, padding_min_length=None):
            if self.teacher_processor.tokenizer.padding_side == "left":
                input_ids = [torch.flip(_input_ids, [0]) for _input_ids in input_ids]
            if padding_min_length is not None:
                original_batch_size = len(input_ids)
                dummy_input_id = torch.ones(padding_min_length) * padding_value
                dummy_input_id.to(device=input_ids[0].device, dtype=input_ids[0].dtype)
                input_ids.append(dummy_input_id)
            input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=batch_first, padding_value=padding_value)
            if padding_min_length is not None:
                input_ids = input_ids[:original_batch_size]
            if self.teacher_processor.tokenizer.padding_side == "left":
                input_ids = torch.flip(input_ids, [1])
            return input_ids

        def prepare_teacher_inputs(self, kwargs):
            images_raw = kwargs["images_raw"]
            inputs_text = kwargs["inputs_text"]
            labels_text = kwargs["labels_text"]

            inputs_text_img_processed = []
            for input_text_core in kwargs["inputs_text_core"]:
                message = input_text_core.replace("<image>", "<|image_1|>")
                if self.config.kd_one_word_prompt is True:
                    message += "\nAnswer the question using a single word or phrase."
                messages = [
                    {"role": "user", "content": message},
                ]
                inputs_text_template_applied = self.teacher_processor.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                inputs_text_img_processed.append(inputs_text_template_applied)

            teacher_inputs_list = []
            teacher_labels_list = []

            teacher_inputs_labels_list = []
            for idx in range(len(labels_text)):
                teacher_inputs = self.teacher_processor(
                    inputs_text_img_processed[idx], kwargs["images_raw"][idx], return_tensors="pt"
                )
                teacher_inputs_list.append(teacher_inputs)

                teacher_labels = self.teacher_processor(kwargs["labels_text"][idx], return_tensors="pt")
                teacher_labels["input_ids"] = torch.concat(
                    [
                        teacher_labels["input_ids"][:, 1:],
                        torch.tensor([[self.teacher_processor.tokenizer.eos_token_id]]),
                    ],
                    dim=-1,
                )
                teacher_labels_list.append(teacher_labels)

            original_padding_side = self.teacher_processor.tokenizer.padding_side
            self.teacher_processor.tokenizer.padding_side = "left"
            teacher_input_ids_concat = self.pad_sequence(
                [item["input_ids"][0] for item in teacher_inputs_list],
                True,
                self.teacher_processor.tokenizer.pad_token_id,
            )
            teacher_inputs_attn_mask_concat = teacher_input_ids_concat.ne(self.teacher_processor.tokenizer.pad_token_id)
            teacher_inputs_pixel_values_concat = self.pad_sequence(
                [item["pixel_values"][0] for item in teacher_inputs_list],
                True,
                self.teacher_processor.tokenizer.pad_token_id,
            )
            teacher_inputs_image_sizes_concat = self.pad_sequence(
                [item["image_sizes"][0] for item in teacher_inputs_list],
                True,
                self.teacher_processor.tokenizer.pad_token_id,
            )

            self.teacher_processor.tokenizer.padding_side = "right"
            teacher_labels_ids_concat = self.pad_sequence(
                [item["input_ids"][0] for item in teacher_labels_list],
                True,
                self.teacher_processor.tokenizer.pad_token_id,
                padding_min_length=getattr(self.tokenizer, "model_min_length_decoder", None),
            )
            teacher_labels_attn_mask_concat = teacher_labels_ids_concat.ne(
                self.teacher_processor.tokenizer.pad_token_id
            )
            self.teacher_processor.tokenizer.padding_side = original_padding_side

            teacher_input_label_ids = torch.concat([teacher_input_ids_concat, teacher_labels_ids_concat], dim=-1)
            teacher_input_label_attn_mask = torch.concat(
                [teacher_inputs_attn_mask_concat, teacher_labels_attn_mask_concat], dim=-1
            )

            teacher_input_kwargs = {
                "input_ids": teacher_input_label_ids.to(device="cuda"),
                "attention_mask": teacher_input_label_attn_mask.to(device="cuda"),
                "pixel_values": teacher_inputs_pixel_values_concat.to(device="cuda", dtype=torch.bfloat16),
                "image_sizes": teacher_inputs_image_sizes_concat.to(device="cuda"),
            }
            return teacher_input_kwargs

        def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            decoder_input_ids: torch.LongTensor = None,
            decoder_attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            images: Optional[torch.FloatTensor] = None,
            image_sizes: Optional[List[List[int]]] = None,
            modalities: Optional[List[str]] = ["image"],
            return_dict: Optional[bool] = None,
            **kwargs,
        ) -> Union[Tuple, CausalLMOutputWithPast]:
            if getattr(self.config, "use_vision_cross_attn", False):
                if images is not None:
                    if getattr(self.config, "use_vision_cross_attn_pass_vision_token_to_encoder", False):
                        (
                            input_ids,
                            position_ids,
                            attention_mask,
                            past_key_values,
                            inputs_embeds,
                            new_labels,
                            image_features,
                        ) = self.prepare_inputs_labels_for_multimodal(
                            input_ids,
                            position_ids,
                            attention_mask,
                            past_key_values,
                            labels,
                            images,
                            modalities=modalities,
                            image_sizes=image_sizes,
                            decoder_attention_mask=decoder_attention_mask,
                            use_seq2seq=self.use_seq2seq,
                        )
                        self.condition_vision_x(image_features)
                        # print("use vision cross attn, and pass vision tokens to encoder!")
                    else:
                        (_, _, _, _, _, _, image_features) = self.prepare_inputs_labels_for_multimodal(
                            input_ids,
                            position_ids,
                            attention_mask,
                            past_key_values,
                            labels,
                            images,
                            modalities=modalities,
                            image_sizes=image_sizes,
                            decoder_attention_mask=decoder_attention_mask,
                            use_seq2seq=self.use_seq2seq,
                        )
                        inputs_embeds = None
                        input_ids[input_ids == -200] = self.tokenizer.encode("<image_tkn>")[0]
                        self.condition_vision_x(image_features)
            else:
                if inputs_embeds is None:
                    (
                        input_ids,
                        position_ids,
                        attention_mask,
                        past_key_values,
                        inputs_embeds,
                        new_labels,
                        image_features,
                    ) = self.prepare_inputs_labels_for_multimodal(
                        input_ids,
                        position_ids,
                        attention_mask,
                        past_key_values,
                        labels,
                        images,
                        modalities=modalities,
                        image_sizes=image_sizes,
                        decoder_attention_mask=decoder_attention_mask,
                        use_seq2seq=self.use_seq2seq,
                    )
            results = super().forward(
                encoder_input_ids=input_ids,
                decoder_input_ids=decoder_input_ids,
                encoder_inputs_embeds=inputs_embeds,
                encoder_attention_mask=attention_mask,
                targets=labels,
            )
            return results

        @torch.no_grad()
        def generate(
            self,
            inputs: Optional[torch.Tensor] = None,
            images: Optional[torch.Tensor] = None,
            image_sizes: Optional[torch.Tensor] = None,
            modalities: Optional[List[str]] = ["image"],
            max_new_tokens: Optional[int] = 100,
            **kwargs,
        ) -> Union[GenerateOutput, torch.LongTensor]:
            position_ids = kwargs.pop("position_ids", None)
            attention_mask = kwargs.pop("attention_mask", None)
            if "inputs_embeds" in kwargs:
                raise NotImplementedError("`inputs_embeds` is not supported")
            if attention_mask is None:
                attention_mask = inputs != self.tokenizer.pad_token_id
            if getattr(self.config, "use_vision_cross_attn", False):
                if images is not None:
                    if getattr(self.config, "use_vision_cross_attn_pass_vision_token_to_encoder", False):
                        (inputs, position_ids, attention_mask, _, inputs_embeds, _, image_features) = (
                            self.prepare_inputs_labels_for_multimodal(
                                inputs,
                                position_ids,
                                attention_mask,
                                None,
                                None,
                                images,
                                modalities=modalities,
                                image_sizes=image_sizes,
                                use_seq2seq=self.use_seq2seq,
                            )
                        )
                        # print("use vision cross attn, and pass vision tokens to encoder!")
                    else:
                        (_, _, _, _, _, _, image_features) = self.prepare_inputs_labels_for_multimodal(
                            inputs,
                            position_ids,
                            attention_mask,
                            None,
                            None,
                            images,
                            modalities=modalities,
                            image_sizes=image_sizes,
                            use_seq2seq=self.use_seq2seq,
                        )
                        inputs_embeds = None
                        inputs[inputs == -200] = self.tokenizer.encode("<image_tkn>")[0]
                else:
                    inputs_embeds = self.get_model().embed_tokens(inputs)
                    inputs_embeds = None
                    inputs[inputs == -200] = self.tokenizer.encode("<image_tkn>")[0]
                self.condition_vision_x(image_features)
            else:
                if images is not None:
                    (inputs, position_ids, attention_mask, _, inputs_embeds, _, image_features) = (
                        self.prepare_inputs_labels_for_multimodal(
                            inputs,
                            position_ids,
                            attention_mask,
                            None,
                            None,
                            images,
                            modalities=modalities,
                            image_sizes=image_sizes,
                            use_seq2seq=self.use_seq2seq,
                        )
                    )
                else:
                    inputs_embeds = self.get_model().embed_tokens(inputs)
            res = super().generate(
                input_ids=inputs,
                inputs_embeds=inputs_embeds,
                encoder_attention_mask=attention_mask,
                eos_token_id=self.tokenizer.eos_token_id,
                decoder_start_token_id=self.tokenizer.bos_token_id,
                bos_token_id=self.tokenizer.bos_token_id,
                top_k=1,  # 50,
                temperature=1,  # 10,
                max_new_tokens=max_new_tokens,  # self.tokenizer.model_max_length,
            )
            return res

        def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
            print("=================calling llava_mu prepare_inputs_for_generation==================")
            images = kwargs.pop("images", None)
            image_sizes = kwargs.pop("image_sizes", None)
            inputs = {
                "decoder_input_ids": input_ids,
            }
            if images is not None:
                inputs["images"] = images
            if image_sizes is not None:
                inputs["image_sizes"] = image_sizes

            return inputs

    AutoConfig.register("llava_mu", LlavaMuConfig)
    AutoModelForCausalLM.register(LlavaMuConfig, LlavaMuForCausalLM)

    return LlavaMuForCausalLM, LlavaMuConfig
