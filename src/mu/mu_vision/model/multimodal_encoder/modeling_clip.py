from typing import Any, Optional, Tuple, Union
import torch
import torch.nn as nn
from transformers import CLIPImageProcessor, CLIPVisionConfig, CLIPPreTrainedModel
from transformers.models.clip.modeling_clip import CLIPVisionEmbeddings, CLIPEncoder
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.utils import replace_return_docstrings, add_start_docstrings_to_model_forward


class CLIPVisionTransformer(nn.Module):
    def __init__(self, config: CLIPVisionConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = CLIPVisionEmbeddings(config)
        self.pre_layrnorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        self.encoder = CLIPEncoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=CLIPVisionConfig)
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        interpolate_pos_encoding: Optional[bool] = False,
        cross_image_pos_embed: Optional[torch.FloatTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:

        """
        # print("===========customized CLIP model!============")
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        hidden_states = self.embeddings(pixel_values)
        if cross_image_pos_embed is not None:
            assert cross_image_pos_embed.shape[-1] == self.config.hidden_size
            hidden_states += cross_image_pos_embed
            # print("use cross_image_pos_embed!")
        hidden_states = self.pre_layrnorm(hidden_states)

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        pooled_output = last_hidden_state[:, 0, :]
        pooled_output = self.post_layernorm(pooled_output)

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class CLIPVisionModel(CLIPPreTrainedModel):
    config_class = CLIPVisionConfig
    main_input_name = "pixel_values"
    _no_split_modules = ["CLIPEncoderLayer"]

    def __init__(self, config: CLIPVisionConfig):
        super().__init__(config)
        self.vision_model = CLIPVisionTransformer(config)
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.vision_model.embeddings.patch_embedding

    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=CLIPVisionConfig)
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: bool = False,
        return_dict: Optional[bool] = None,
        cross_image_pos_embed: Optional[torch.FloatTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, CLIPVisionModel

        >>> model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
        >>> processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> last_hidden_state = outputs.last_hidden_state
        >>> pooled_output = outputs.pooler_output  # pooled CLS states
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        return self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            interpolate_pos_encoding=interpolate_pos_encoding,
            cross_image_pos_embed=cross_image_pos_embed,
        )
