"""
ChakrGenesis Foundation Neural Model Architecture.
State-of-the-art autoregressive decoder transformer:
- Pre-RMSNorm with residual damping
- Rotary Positional Embeddings (RoPE)
- Grouped-Query Attention (GQA) for efficient key-value caching
- SwiGLU feed-forward networks
- Weight-tied token embeddings and output projection
- Explicit causal masking and KV cache support
"""

from __future__ import annotations
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class GenesisConfig:
    def __init__(
        self,
        vocab_size: int = 4096,
        d_model: int = 256,
        n_layers: int = 6,
        n_q_heads: int = 8,
        n_kv_heads: int = 2,
        hidden_dim: int = 682,
        max_seq_len: int = 1024,
        rms_norm_eps: float = 1e-5,
        rope_theta: float = 10000.0,
        dropout: float = 0.0,
        tie_embeddings: bool = True,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        pad_token_id: int = 0
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_q_heads = n_q_heads
        self.n_kv_heads = n_kv_heads
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.dropout = dropout
        self.tie_embeddings = tie_embeddings
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 1024, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        self.register_buffer("cos_cached", freqs.cos(), persistent=False)
        self.register_buffer("sin_cached", freqs.sin(), persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x shape: (B, H, T, D)
    d = x.shape[-1]
    x1, x2 = x[..., :d//2], x[..., d//2:]
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, T, D//2)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class GroupedQueryAttention(nn.Module):
    def __init__(self, config: GenesisConfig):
        super().__init__()
        self.d_model = config.d_model
        self.n_q_heads = config.n_q_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.d_model // config.n_q_heads
        self.num_reps = config.n_q_heads // config.n_kv_heads

        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k_proj = nn.Linear(config.d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.d_model, config.d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_q_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # Repeat KV heads to match Q heads
        if self.num_reps > 1:
            k = k.repeat_interleave(self.num_reps, dim=1)
            v = v.repeat_interleave(self.num_reps, dim=1)

        scale = 1.0 / math.sqrt(self.head_dim)
        attn = (q @ k.transpose(-2, -1)) * scale

        if mask is not None:
            attn = attn + mask

        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(out)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int):
        super().__init__()
        self.w_gate = nn.Linear(d_model, hidden_dim, bias=False)
        self.w_up = nn.Linear(d_model, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class GenesisBlock(nn.Module):
    def __init__(self, config: GenesisConfig):
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model, config.rms_norm_eps)
        self.attn = GroupedQueryAttention(config)
        self.ffn_norm = RMSNorm(config.d_model, config.rms_norm_eps)
        self.ffn = SwiGLU(config.d_model, config.hidden_dim)
        # Residual scale to damp late-layer attractor blowups
        self.res_scale = 1.0 / math.sqrt(2.0 * config.n_layers)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        h = x + self.res_scale * self.attn(self.attn_norm(x), cos, sin, mask)
        out = h + self.res_scale * self.ffn(self.ffn_norm(h))
        return out


class ChakrGenesis(nn.Module):
    def __init__(self, config: GenesisConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.rope = RotaryEmbedding(config.d_model // config.n_q_heads, config.max_seq_len, config.rope_theta)
        self.layers = nn.ModuleList([GenesisBlock(config) for _ in range(config.n_layers)])
        self.final_norm = RMSNorm(config.d_model, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        if config.tie_embeddings:
            self.lm_head.weight = self.embedding.weight

        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if "weight" in name and p.dim() >= 2:
                nn.init.normal_(p, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T = input_ids.shape
        x = self.embedding(input_ids)
        cos, sin = self.rope(x, T)

        # Causal mask: upper triangle set to -inf
        mask = torch.full((T, T), float("-inf"), device=input_ids.device)
        mask = torch.triu(mask, diagonal=1).unsqueeze(0).unsqueeze(0)

        for layer in self.layers:
            x = layer(x, cos, sin, mask)

        x = self.final_norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100)

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 0.7,
        top_k: int = 40,
        eos_token_id: Optional[int] = 2
    ) -> torch.Tensor:
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = input_ids if input_ids.size(1) <= self.config.max_seq_len else input_ids[:, -self.config.max_seq_len:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-5)

            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat((input_ids, idx_next), dim=1)

            if eos_token_id is not None and idx_next.item() == eos_token_id:
                break

        return input_ids
