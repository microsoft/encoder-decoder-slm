import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import Seq2SeqLMOutput

from mu.models.mu_utils import MuDecoderBlock, MuEncoderBlock, MuRMSNorm

logger = logging.getLogger(__name__)


@dataclass
class MuConfig:
    n_encoder_layer: int = 20
    n_decoder_layer: int = 10
    n_embd: int = 1024
    qkv_embd: Optional[int] = 384
    is_interleaved: bool = False
    hidden_dim: int = 4096

    n_head: int = 6
    n_kv_head: int = 3
    block_size: int = 1024

    dropout: float = 0.1
    rope_theta: int = 10000
    rope_factor: float = 1.0

    pad_token_id: int = 0
    vocab_size: int = 32128

    def __post_init__(self):
        if self.qkv_embd is None:
            self.qkv_embd = self.n_embd


class Mu(nn.Module):
    def __init__(self, config, device):
        super(Mu, self).__init__()
        torch.manual_seed(42)

        self.config = config
        self.device = device

        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(config.vocab_size, config.n_embd, device=device),
                "e": nn.ModuleList([MuEncoderBlock(config, device) for _ in range(config.n_encoder_layer)]),
                "ln_e": MuRMSNorm(config.n_embd, device=device),
                "h": nn.ModuleList([MuDecoderBlock(config, device) for _ in range(config.n_decoder_layer)]),
                "ln_f": MuRMSNorm(config.n_embd, device=device),
            }
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, device=device, bias=False)
        self.lm_head.weight = self.transformer.wte.weight

        self.init_rng = torch.Generator(device=device)
        self.init_rng.manual_seed(42)
        self.apply(self._init_weights)
        num_params = sum(p.numel() for p in self.parameters())
        logger.info(f"Number of parameters: {num_params / 1e6:.2f}M")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02, generator=self.init_rng)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02, generator=self.init_rng)

    def encode(self, encoder_input_ids):
        encoder_attention_mask = ~(encoder_input_ids == self.config.pad_token_id).bool()
        e = self.transformer.wte(encoder_input_ids)
        for block in self.transformer.e:
            e = block(e, encoder_mask=encoder_attention_mask)
        e = self.transformer.ln_e(e)
        return e, encoder_attention_mask

    def decode(self, decoder_input_ids, encoder_outputs, encoder_attention_mask):
        decoder_attention_mask = ~(decoder_input_ids == self.config.pad_token_id).bool()
        x = self.transformer.wte(decoder_input_ids)
        for block in self.transformer.h:
            x = block(
                x,
                enc_out=encoder_outputs,
                encoder_mask=encoder_attention_mask,
                decoder_mask=decoder_attention_mask,
            )
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        return logits

    def forward(
        self,
        encoder_input_ids,
        decoder_input_ids,
        targets=None,
        encoder_attention_mask=None,
        decoder_attention_mask=None,
    ):
        assert encoder_input_ids.size(0) == decoder_input_ids.size(0), (
            "Batch size mismatch between encoder and decoder inputs"
        )
        encoder_out, encoder_attention_mask = self.encode(encoder_input_ids)
        logits = self.decode(decoder_input_ids, encoder_out, encoder_attention_mask)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=self.config.pad_token_id,
            )
        return Seq2SeqLMOutput(logits=logits, loss=loss)

    @torch.no_grad()
    def generate(
        self,
        input_ids,
        bos_token_id,
        eos_token_id,
        max_new_tokens=64,
        temperature=1.0,
        top_k=None,
        num_return_sequences=1,
        **kwargs,
    ):
        B = input_ids.size(0)
        decoder_input_ids = input_ids.new_full((B, 1), bos_token_id, dtype=torch.long)

        if num_return_sequences > 1:
            input_ids = input_ids.repeat(num_return_sequences, 1)
            decoder_input_ids = decoder_input_ids.repeat(num_return_sequences, 1)

        is_finished = torch.zeros(B, dtype=torch.bool, device=self.device)
        encoder_out, encoder_attention_mask = self.encode(input_ids)
        for _ in range(max_new_tokens):
            logits = self.decode(decoder_input_ids, encoder_out, encoder_attention_mask)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                topk = min(top_k, logits.size(-1))
                v, _ = torch.topk(logits, topk)
                logits[logits < v[:, [-1]]] = -float("Inf")
            probs = F.softmax(logits, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1)
            is_finished |= next_tokens.squeeze(-1) == eos_token_id
            next_tokens[is_finished, :] = eos_token_id
            decoder_input_ids = torch.cat([decoder_input_ids, next_tokens], dim=-1)
            if is_finished.all():
                break
        return decoder_input_ids


def get_model(model_path: Path, device: str = "cpu"):
    config = MuConfig()
    model = Mu(config, device=device)
    model = model.to(dtype=torch.bfloat16)
    model = model.to(device=device)

    loaded = torch.load(model_path, map_location=device, weights_only=True)
    state_dict = loaded["model_state_dict"]
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model
