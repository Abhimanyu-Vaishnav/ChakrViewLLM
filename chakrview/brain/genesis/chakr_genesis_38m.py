"""
chakrview/brain/genesis/chakr_genesis_38m.py
ChakrGenesis-38M Model Architecture:
- Designed for NR-050 Controlled Scaling Study.
- Target: ~37.5M unique trainable parameters (with tied embeddings).
- Architectural characteristics identical to ChakrGenesis-17M family:
  * Byte-Level BPE 4096 vocabulary
  * Rotary Position Embeddings (RoPE)
  * RMSNorm pre-normalization
  * SwiGLU / Gated-MLP feedforward blocks
  * Grouped Query Attention (GQA 8:4)
- Scaled dimensions:
  * d_model = 512 (scaled from 384)
  * n_layers = 12 (scaled from 10)
  * n_q_heads = 8 (head_dim = 64)
  * n_kv_heads = 4 (Grouped Query Attention 2:1)
  * hidden_dim = 1408 (intermediate MLP dim)
  * max_seq_len = 512
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class Genesis38MConfig:
    vocab_size: int = 4096
    d_model: int = 512
    n_layers: int = 12
    n_q_heads: int = 8
    n_kv_heads: int = 4
    hidden_dim: int = 1408
    max_seq_len: int = 512
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    dropout: float = 0.0
    tie_embeddings: bool = True
    bos_token_id: int = 0
    eos_token_id: int = 1
    pad_token_id: int = 2

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight

class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 512, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        # freqs is [seq_len, dim // 2]
        # Repeat to match full head_dim
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.cos_cached.shape[0]:
            t = torch.arange(seq_len, dtype=torch.float32, device=x.device)
            freqs = torch.outer(t, self.inv_freq.to(x.device))
            emb = torch.cat((freqs, freqs), dim=-1)
            return emb.cos(), emb.sin()
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # q: [b, n_heads, seq_len, head_dim]
    # cos, sin: [seq_len, head_dim]
    cos = cos.unsqueeze(0).unsqueeze(1)
    sin = sin.unsqueeze(0).unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class Attention38M(nn.Module):
    def __init__(self, cfg: Genesis38MConfig):
        super().__init__()
        self.cfg = cfg
        self.head_dim = cfg.d_model // cfg.n_q_heads
        self.n_q_heads = cfg.n_q_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.num_kv_groups = cfg.n_q_heads // cfg.n_kv_heads

        self.q_proj = nn.Linear(cfg.d_model, cfg.n_q_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(cfg.n_q_heads * self.head_dim, cfg.d_model, bias=False)
        self.rotary_emb = RotaryEmbedding(self.head_dim, max_seq_len=cfg.max_seq_len, theta=cfg.rope_theta)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, S, D = x.shape
        q = self.q_proj(x).view(B, S, self.n_q_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(x, S)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # GQA repeat kv heads
        if self.num_kv_groups > 1:
            k = k.repeat_interleave(self.num_kv_groups, dim=1)
            v = v.repeat_interleave(self.num_kv_groups, dim=1)

        # Scaled dot-product attention
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=(mask is None and S > 1))
        out = out.transpose(1, 2).contiguous().view(B, S, D)
        return self.out_proj(out)

class FeedForward38M(nn.Module):
    def __init__(self, cfg: Genesis38MConfig):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.d_model, cfg.hidden_dim, bias=False)
        self.up_proj = nn.Linear(cfg.d_model, cfg.hidden_dim, bias=False)
        self.down_proj = nn.Linear(cfg.hidden_dim, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

class TransformerBlock38M(nn.Module):
    def __init__(self, cfg: Genesis38MConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, eps=cfg.rms_norm_eps)
        self.attn = Attention38M(cfg)
        self.ffn_norm = RMSNorm(cfg.d_model, eps=cfg.rms_norm_eps)
        self.ffn = FeedForward38M(cfg)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), mask=mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x

class ChakrGenesis38M(nn.Module):
    def __init__(self, cfg: Genesis38MConfig):
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.layers = nn.ModuleList([TransformerBlock38M(cfg) for _ in range(cfg.n_layers)])
        self.final_norm = RMSNorm(cfg.d_model, eps=cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        if cfg.tie_embeddings:
            self.lm_head.weight = self.embedding.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, None]:
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x, mask=mask)
        x = self.final_norm(x)
        logits = self.lm_head(x)
        return logits, None
