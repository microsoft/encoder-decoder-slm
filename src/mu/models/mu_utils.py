import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MuGELU(nn.Module):
    def forward(self, input):
        return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))


class MuRMSNorm(nn.Module):
    def __init__(self, n_embd, device, eps=1e-6):
        super(MuRMSNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(n_embd, device=device))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class MuAttention(nn.Module):
    def __init__(self, config, device, is_causal):
        super(MuAttention, self).__init__()
        self.config = config
        self.device = device
        self.is_causal = is_causal
        self.is_interleaved = config.is_interleaved

        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.num_key_value_groups = self.n_head // self.n_kv_head
        self.head_dim = config.qkv_embd // self.n_head

        assert self.head_dim * self.n_head == config.qkv_embd, "QKV Embedding size must be divisible by n_head"
        assert self.n_head % self.n_kv_head == 0, "n_head must be divisible by n_kv_head"

        self.q_linear = nn.Linear(config.n_embd, self.n_head * self.head_dim, device=device)
        self.k_linear = nn.Linear(config.n_embd, self.n_kv_head * self.head_dim, device=device)
        self.v_linear = nn.Linear(config.n_embd, self.n_kv_head * self.head_dim, device=device)
        self.out_linear = nn.Linear(config.qkv_embd, config.n_embd, device=device)
        self.attn_dropout = nn.Dropout(config.dropout)

        self.scale = math.sqrt(self.head_dim)

    def rotate_half(self, x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary_pos_emb(self, q, k, cos, sin):
        q_embed = (q * cos) + (self.rotate_half(q) * sin)
        k_embed = (k * cos) + (self.rotate_half(k) * sin)
        return q_embed, k_embed

    @torch._dynamo.disable()
    def generate_sin_cos_pos_emb(self, seq_len, device):
        base, rope_factor, dim, max_seq_len = (
            self.config.rope_theta,
            self.config.rope_factor,
            self.head_dim,
            self.config.block_size,
        )
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))
        if rope_factor > 1.0:  # Apply NTK dynamic scaling
            seq_len_eff = max(seq_len, max_seq_len)
            base_adjustment = ((rope_factor * seq_len_eff / max_seq_len) - (rope_factor - 1)) ** (dim / (dim - 2))
            adjusted_base = base * base_adjustment
            inv_freq = 1.0 / (adjusted_base ** (torch.arange(0, dim, 2, device=device).float() / dim))

        position_ids = torch.arange(seq_len, device=device, dtype=torch.float)
        if not self.is_causal:
            position_ids = position_ids - ((seq_len - 1) // 2)
        freqs = torch.einsum("i,j->ij", position_ids, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos_emb = emb.cos()[None, None, :, :]
        sin_emb = emb.sin()[None, None, :, :]
        return cos_emb, sin_emb

    def repeat_kv(self, x):
        batch_size, n_kv_head, seq_len, head_dim = x.size()
        if self.num_key_value_groups == 1:
            return x
        else:
            if self.is_interleaved:
                x = x.unsqueeze(2)
                x = x.expand(batch_size, n_kv_head, self.num_key_value_groups, seq_len, head_dim)
                x = x.contiguous().view(batch_size, n_kv_head * self.num_key_value_groups, seq_len, head_dim)
            else:
                x = torch.cat([x] * self.num_key_value_groups, dim=1)
            return x

    def forward(self, query, key, value, mask=None, apply_rope_pos_emb=True):
        batch_size, seq_len, _ = query.size()

        Q = self.q_linear(query).view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        K = self.k_linear(key).view(batch_size, -1, self.n_kv_head, self.head_dim).transpose(1, 2)
        V = self.v_linear(value).view(batch_size, -1, self.n_kv_head, self.head_dim).transpose(1, 2)

        if apply_rope_pos_emb:
            cos_emb, sin_emb = self.generate_sin_cos_pos_emb(seq_len, device=Q.device)
            torch_dtype = Q.dtype
            cos_emb = cos_emb.to(torch_dtype)
            sin_emb = sin_emb.to(torch_dtype)
            Q, K = self.apply_rotary_pos_emb(Q, K, cos_emb, sin_emb)

        K = self.repeat_kv(K)
        V = self.repeat_kv(V)

        attn = torch.matmul(Q, K.transpose(-2, -1))
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)
            if self.is_causal:
                causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=Q.device)).bool()
                mask = mask & causal_mask.unsqueeze(0).unsqueeze(0)
            attn = attn.masked_fill(mask == 0, torch.finfo(attn.dtype).min)

        attn_weights = F.softmax(attn / self.scale, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, V)

        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, self.config.qkv_embd)
        return self.out_linear(attn_output)


class MuMLP(nn.Module):
    def __init__(self, config, device):
        super(MuMLP, self).__init__()
        self.config = config
        self.device = device

        self.fc1 = nn.Linear(config.n_embd, config.hidden_dim, device=device)
        self.fc2 = nn.Linear(config.hidden_dim, config.n_embd, device=device)
        self.act = MuGELU()
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class MuEncoderBlock(nn.Module):
    def __init__(self, config, device):
        super(MuEncoderBlock, self).__init__()
        self.config = config
        self.device = device

        self.ln1 = MuRMSNorm(config.n_embd, device=device)
        self.ln2 = MuRMSNorm(config.n_embd, device=device)
        self.attn = MuAttention(config, device, is_causal=False)
        self.mlp = MuMLP(config, device)

    def forward(self, x, encoder_mask=None):
        x_ln1 = self.ln1(x)
        x = x + self.attn(x_ln1, x_ln1, x_ln1, mask=encoder_mask)
        x = x + self.mlp(self.ln2(x))
        return x


class MuDecoderBlock(nn.Module):
    def __init__(self, config, device):
        super(MuDecoderBlock, self).__init__()
        self.config = config
        self.device = device

        self.ln1 = MuRMSNorm(config.n_embd, device=device)
        self.ln2 = MuRMSNorm(config.n_embd, device=device)
        self.ln3 = MuRMSNorm(config.n_embd, device=device)
        self.causal_attn = MuAttention(config, device, is_causal=True)
        self.cross_attn = MuAttention(config, device, is_causal=False)
        self.mlp = MuMLP(config, device)

    def forward(self, x, enc_out, encoder_mask=None, decoder_mask=None):
        x_ln1 = self.ln1(x)
        x = x + self.causal_attn(x_ln1, x_ln1, x_ln1, mask=decoder_mask)

        # In cross-attention, query is the decoder hidden state and key-value is the encoder hidden state
        x = x + self.cross_attn(self.ln2(x), enc_out, enc_out, mask=encoder_mask, apply_rope_pos_emb=False)
        x = x + self.mlp(self.ln3(x))
        return x
