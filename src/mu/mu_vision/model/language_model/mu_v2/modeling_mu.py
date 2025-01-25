from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import Seq2SeqLMOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig

from .mu_utils import MuDecoderBlock, MuEncoderBlock, MuRMSNorm
from .optim import CombinedOptimizer, OrthogonalNesterov

"""
Training plan:
Pretraining with block_size=1024, rope_factor=1.0
Finetuning with block_size=4096, rope_factor=4.0
Inference from finetuned checkpoint with block_size=4096, rope_factor=1.0
"""


@dataclass
class MuConfig(PretrainedConfig):
    model_type = "mu"

    def __init__(
        self,
        vocab_size=32128,
        block_size=1024,
        n_layer=20,
        n_encoder_layer=None,
        n_decoder_layer=None,
        n_head=16,
        n_embd=768,
        n_kv_head=8,
        hidden_dim=3072,
        dropout=0.1,
        rope_theta=10000,
        rope_factor=1.0,
        pad_token_id=0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.hidden_size = n_embd
        self.n_kv_head = n_kv_head
        self.hidden_dim = hidden_dim

        self.dropout = dropout
        self.pad_token_id = pad_token_id
        self.rope_theta = rope_theta
        self.rope_factor = rope_factor

        self.n_encoder_layer = n_encoder_layer
        self.n_decoder_layer = n_decoder_layer

        if self.n_encoder_layer is None:
            self.n_encoder_layer = self.n_layer

        if self.n_decoder_layer is None:
            self.n_decoder_layer = self.n_layer

        self.n_layer = None


class Mu(PreTrainedModel):
    config_class = MuConfig

    def __init__(self, config, device=None):
        super().__init__(config)
        torch.manual_seed(42)

        self.config = config
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(config.vocab_size, config.n_embd, device=device),
                "e": nn.ModuleList([MuEncoderBlock(config, device) for _ in range(config.n_encoder_layer)]),
                "ln_e": MuRMSNorm(config.n_embd, device=device),
                "h": nn.ModuleList([MuDecoderBlock(config, device) for _ in range(config.n_decoder_layer)]),
                "ln_f": MuRMSNorm(config.n_embd, device=device),
            }
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, device=device)
        self.lm_head.weight = self.transformer.wte.weight

        self.init_rng = torch.Generator(device=device)
        self.init_rng.manual_seed(42)
        self.apply(self._init_weights)
        num_params = sum(p.numel() for p in self.parameters())
        print(f"Number of parameters: {num_params / 1e6:.2f}M")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def encode(self, encoder_input_ids=None, encoder_attention_mask=None, encoder_inputs_embeds=None):
        if encoder_attention_mask is None:
            encoder_attention_mask = ~(encoder_input_ids == self.config.pad_token_id).bool()

        if encoder_input_ids is not None:
            e = self.transformer.wte(encoder_input_ids)
        elif encoder_inputs_embeds is not None:
            e = encoder_inputs_embeds
        else:
            raise ValueError("You have to specify either encoder_input_ids or encoder_inputs_embeds.")

        for block in self.transformer.e:
            e = block(e, encoder_mask=encoder_attention_mask)
        e = self.transformer.ln_e(e)
        return e, encoder_attention_mask

    def decode(self, decoder_input_ids, encoder_outputs, encoder_attention_mask):
        decoder_attention_mask = ~(decoder_input_ids == self.config.pad_token_id).bool()
        x = self.transformer.wte(decoder_input_ids)
        for block in self.transformer.h:
            x = block(
                x, enc_out=encoder_outputs, encoder_mask=encoder_attention_mask, decoder_mask=decoder_attention_mask
            )
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        return logits

    def forward(
        self,
        encoder_input_ids=None,
        decoder_input_ids=None,
        targets=None,
        encoder_attention_mask=None,
        decoder_attention_mask=None,
        encoder_inputs_embeds=None,
    ):
        if encoder_input_ids is not None and encoder_inputs_embeds is not None:
            raise ValueError("You cannot specify both encoder_input_ids and encoder_inputs_embeds.")

        encoder_out, encoder_attention_mask = self.encode(
            encoder_input_ids, encoder_attention_mask, encoder_inputs_embeds
        )
        logits = self.decode(decoder_input_ids, encoder_out, encoder_attention_mask)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
            )
        return Seq2SeqLMOutput(logits=logits, loss=loss)

    @torch.no_grad()
    def generate(
        self,
        input_ids=None,
        max_new_tokens=-1,
        bos_token_id=None,
        eos_token_id=None,
        temperature=1.0,
        top_k=None,
        num_return_sequences=1,
        encoder_attention_mask=None,
        inputs_embeds=None,
        **kwargs,
    ):
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds.")

        if input_ids is not None:
            B = input_ids.size(0)
            decoder_input_ids = input_ids.new_full((B, 1), bos_token_id, dtype=torch.long)

            if num_return_sequences > 1:
                input_ids = input_ids.repeat(num_return_sequences, 1)
                decoder_input_ids = decoder_input_ids.repeat(num_return_sequences, 1)
        elif inputs_embeds is not None:
            B, T, _ = inputs_embeds.size()
            decoder_input_ids = inputs_embeds.new_full((B, 1), bos_token_id, dtype=torch.long)

            if num_return_sequences > 1:
                decoder_input_ids = decoder_input_ids.repeat(num_return_sequences, 1)
                inputs_embeds = inputs_embeds.repeat(num_return_sequences, 1, 1)
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds.")

        encoder_out, encoder_attention_mask = self.encode(input_ids, encoder_attention_mask, inputs_embeds)
        for _ in range(max_new_tokens):
            logits = self.decode(decoder_input_ids, encoder_out, encoder_attention_mask)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                topk = min(top_k, logits.size(-1))
                v, _ = torch.topk(logits, topk)
                logits[logits < v[:, [-1]]] = -float("Inf")
            probs = F.softmax(logits, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1)
            decoder_input_ids = torch.cat([decoder_input_ids, next_tokens], dim=-1)
            if (next_tokens == eos_token_id).all():
                break
        return decoder_input_ids

    def configure_optimizers(self, weight_decay, learning_rate, betas):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}

        decay_params = []
        nodecay_params = []
        orthogonal_params = []

        for n, p in param_dict.items():
            if p.dim() == 2 and ("transformer.h." in n or "transformer.e." in n):
                orthogonal_params.append(p)
            elif p.dim() >= 2:
                decay_params.append(p)
            else:
                nodecay_params.append(p)

        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]

        adam_optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, fused=True)
        orthogonal_nesterov = OrthogonalNesterov(orthogonal_params, lr=0.1 * learning_rate, momentum=0.95)
        optimizer = CombinedOptimizer([adam_optimizer, orthogonal_nesterov])

        return optimizer


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # test model
    config = MuConfig(block_size=512, rope_factor=2.0)
    model = Mu(config, device)

    model = model.to(torch.bfloat16)
    model = torch.compile(model)

    encoder_input_ids = torch.randint(0, config.vocab_size, (2, 1024), device=device)
    decoder_input_ids = torch.randint(0, config.vocab_size, (2, 1024), device=device)
    targets = torch.randint(0, config.vocab_size, (2, 1024), device=device)
    output = model(encoder_input_ids, decoder_input_ids, targets)
    print(output.loss, output.logits)

    # test generate
    generated = model.generate(
        encoder_input_ids, 1024, bos_token_id=1, eos_token_id=32000, temperature=1.0, top_k=50, num_return_sequences=1
    )
    print(generated)

    # test optimizer
    optimizer = model.configure_optimizers(weight_decay=0.01, learning_rate=1e-4, betas=(0.9, 0.98))
    optimizer.zero_grad()
    output = model(encoder_input_ids, decoder_input_ids, targets)
    output.loss.backward()
    optimizer.step()
    print("Optimizer test passed")
