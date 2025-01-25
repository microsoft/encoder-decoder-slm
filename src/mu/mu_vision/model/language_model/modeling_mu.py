import math

import torch
import torch.nn.functional as F
from torch import nn


def create_attention_mask(input_ids, pad_token_id=0):
    return (input_ids != pad_token_id).unsqueeze(1).unsqueeze(2)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, scaling_factor=1.0, center=False):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.scaling_factor = scaling_factor
        self.center = center

    def forward(self, seq_len, device, dtype):
        pos = torch.arange(seq_len, device=device, dtype=dtype)
        if self.center:
            pos = pos - (seq_len - 1) / 2
        pos = pos.unsqueeze(1)
        scaled_inv_freq = self.inv_freq / self.scaling_factor
        freqs = torch.einsum("i,j->ij", pos.squeeze(-1), scaled_inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        # Return cosine and sine embeddings separately
        cos_emb = emb.cos()
        sin_emb = emb.sin()
        return cos_emb, sin_emb


def rotate_half(x):
    """
    Rotate half the hidden dims of the input.
    """
    x1, x2 = torch.chunk(x, 2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    """
    Apply rotary position embeddings to queries and keys.
    """
    # Ensure cos and sin have the correct shape
    # q, k: [batch_size, seq_len, num_heads, head_dim]
    # cos, sin: [seq_len, head_dim]
    cos = cos[None, :, None, :]  # [1, seq_len, 1, head_dim]
    sin = sin[None, :, None, :]  # [1, seq_len, 1, head_dim]

    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    return q_rot, k_rot


class NewGELU(nn.Module):
    def forward(self, input):
        return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))


class GroupQueryAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, n_kv_heads, dropout=0.0):
        super(GroupQueryAttention, self).__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads

        self.num_heads = num_heads
        self.n_kv_heads = n_kv_heads

        assert num_heads % n_kv_heads == 0
        self.num_groups = num_heads // n_kv_heads

        self.q_proj = nn.Linear(embed_dim, self.num_heads * self.head_dim)
        self.k_proj = nn.Linear(embed_dim, self.n_kv_heads * self.head_dim)
        self.v_proj = nn.Linear(embed_dim, self.n_kv_heads * self.head_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, cos_emb=None, sin_emb=None, mask=None, use_print=False):
        assert query.shape[0] == key.shape[0] == value.shape[0]
        assert query.shape[2] == key.shape[2] == value.shape[2] == self.embed_dim
        batch_size, Tq, embed_dim = query.size()
        _, Tk, _ = key.size()

        q = self.q_proj(query).view(batch_size, Tq, self.num_heads, self.head_dim)
        k = self.k_proj(key).view(batch_size, Tk, self.n_kv_heads, self.head_dim)
        v = self.v_proj(value).view(batch_size, Tk, self.n_kv_heads, self.head_dim)

        k = k[:, :, :, None, :].expand(batch_size, Tk, self.n_kv_heads, self.num_groups, self.head_dim)
        v = v[:, :, :, None, :].expand(batch_size, Tk, self.n_kv_heads, self.num_groups, self.head_dim)

        k = k.reshape(batch_size, Tk, self.num_heads, self.head_dim)
        v = v.reshape(batch_size, Tk, self.num_heads, self.head_dim)

        if cos_emb is not None and sin_emb is not None:
            q, k = apply_rotary_pos_emb(q, k, cos_emb, sin_emb)

        q = q.transpose(1, 2)  # [batch_size, num_heads, Tq, head_dim]
        k = k.transpose(1, 2)  # [batch_size, num_heads, Tk, head_dim]
        v = v.transpose(1, 2)  # [batch_size, num_heads, Tk, head_dim]

        k = k.transpose(2, 3)  # [batch_size, num_heads, head_dim, Tk]

        if use_print is True:
            print("q shape: {}, k shape: {}".format(q.shape, k.shape))
        scores = torch.matmul(q, k) / math.sqrt(self.head_dim)  # [batch_size, num_heads, Tq, Tk]

        if mask is not None:
            scores = scores.masked_fill(mask[:, :, :Tq, :Tk] == 0, float("-inf"))

        attn_probs = F.softmax(scores, dim=-1)
        attn_probs = self.attn_dropout(attn_probs)

        attn_output = torch.matmul(attn_probs, v)  # [batch_size, num_heads, Tq, head_dim]
        attn_output = attn_output.transpose(1, 2).contiguous()  # [batch_size, Tq, num_heads, head_dim]
        attn_output = attn_output.view(batch_size, Tq, self.embed_dim)

        output = self.out_proj(attn_output)

        return output


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.attn = GroupQueryAttention(config.n_embd, config.n_head, config.n_kv_heads)
        self.attn.out_proj.LLMC_RESIDUAL_SCALE_FLAG = 1
        scaling_factor = config.block_size / config.original_block_size
        self.rotary_emb = RotaryEmbedding(config.n_embd // config.n_head, scaling_factor, center=False)

    def forward(self, x, attention_mask=None):
        B, T, C = x.size()
        if attention_mask is None:
            attention_mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
        cos_emb, sin_emb = self.rotary_emb(T, device=x.device, dtype=x.dtype)
        y = self.attn(x, x, x, cos_emb=cos_emb, sin_emb=sin_emb, mask=attention_mask, use_print=False)
        return y


class SelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.attn = GroupQueryAttention(config.n_embd, config.n_head, config.n_kv_heads)
        self.attn.out_proj.LLMC_RESIDUAL_SCALE_FLAG = 1
        scaling_factor = config.block_size / config.original_block_size
        self.rotary_emb = RotaryEmbedding(config.n_embd // config.n_head, scaling_factor, center=True)

    def forward(self, x, attention_mask=None):
        B, T, C = x.size()
        cos_emb, sin_emb = self.rotary_emb(T, device=x.device, dtype=x.dtype)
        y = self.attn(x, x, x, cos_emb=cos_emb, sin_emb=sin_emb, mask=attention_mask)
        return y


class CrossAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.attn = GroupQueryAttention(config.n_embd, config.n_head, config.n_kv_heads)
        self.attn.out_proj.LLMC_RESIDUAL_SCALE_FLAG = 1

    def forward(self, x, enc_out, encoder_attention_mask=None):
        B, T, C = x.size()
        y = self.attn(query=x, key=enc_out, value=enc_out, mask=encoder_attention_mask)
        return y
