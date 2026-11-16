"""UCIP-07 Neural Architecture Evolution Suite.

Implements modular architectural enhancements authorized by owner:
1. ChakrEvolvedLM: Support for arbitrary layer depth, model width, head counts,
   Grouped-Query Attention (GQA), SwiGLU FFN with configurable expansion,
   enhanced RoPE base theta (50,000 for longer context stability),
   and optional untied LMHead projection with pre-head normalization.
2. Architecture family factory:
   - ChakrMicro (3.44M)
   - ChakrMedium (12.93M)
   - ChakrCore (27.80M tied / 29.89M untied)
   - ChakrLarge (53.43M)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from chakrview.brain.normalization import RMSNorm
from chakrview.brain.rotary import RotaryEmbedding


@dataclass
class EvolvedModelConfig:
    vocab_size: int = 4096
    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 8
    n_kv_heads: Optional[int] = None  # None => Multi-Head Attention, < n_heads => GQA
    hidden_dim: int = 1408
    max_seq_len: int = 512
    rope_base: float = 50000.0
    tie_embeddings: bool = True
    dropout: float = 0.0
    norm_eps: float = 1e-5


class EvolvedAttention(nn.Module):
    """Causal Multi-Head / Grouped-Query Attention with Rotary Position Embeddings."""

    def __init__(self, config: EvolvedModelConfig):
        super().__init__()
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads if config.n_kv_heads is not None else config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.num_kv_groups = self.n_heads // self.n_kv_heads

        assert config.d_model % config.n_heads == 0, "d_model must be divisible by n_heads"
        assert self.n_heads % self.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"

        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k_proj = nn.Linear(config.d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)

        self.rope = RotaryEmbedding(
            dim=self.head_dim,
            max_seq_len=config.max_seq_len,
            theta=config.rope_base,
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE separately to q and k: forward(x, seq_len, offset)
        q = self.rope(q, seq_len=seq_len, offset=0)
        k = self.rope(k, seq_len=seq_len, offset=0)

        # Repeat KV for Grouped-Query Attention if needed
        if self.num_kv_groups > 1:
            k = k.repeat_interleave(self.num_kv_groups, dim=1)
            v = v.repeat_interleave(self.num_kv_groups, dim=1)

        # Scaled Dot-Product Attention with causal mask
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        if mask is None:
            # Construct standard causal lower-triangular mask
            causal_mask = torch.triu(torch.full((seq_len, seq_len), float("-inf"), device=x.device), diagonal=1)
            scores = scores + causal_mask
        else:
            scores = scores + mask

        attn_weights = F.softmax(scores, dim=-1)
        out = torch.matmul(attn_weights, v)  # (B, H, T, D)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        return self.out_proj(out)


class EvolvedSwiGLUFFN(nn.Module):
    """SwiGLU Feed-Forward Network: down_proj(SiLU(gate_proj(x)) * up_proj(x))."""

    def __init__(self, config: EvolvedModelConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.d_model, config.hidden_dim, bias=False)
        self.up_proj = nn.Linear(config.d_model, config.hidden_dim, bias=False)
        self.down_proj = nn.Linear(config.hidden_dim, config.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class EvolvedTransformerBlock(nn.Module):
    """Pre-RMSNorm Transformer Layer with Residual Connections."""

    def __init__(self, config: EvolvedModelConfig):
        super().__init__()
        self.norm_1 = RMSNorm(config.d_model, eps=config.norm_eps)
        self.attn = EvolvedAttention(config)
        self.norm_2 = RMSNorm(config.d_model, eps=config.norm_eps)
        self.ffn = EvolvedSwiGLUFFN(config)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(self.norm_1(x), mask=mask)
        x = x + self.ffn(self.norm_2(x))
        return x


class ChakrEvolvedLM(nn.Module):
    """ChakrView Evolved Autoregressive Language Model."""

    def __init__(self, config: EvolvedModelConfig):
        super().__init__()
        self.config = config

        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList([EvolvedTransformerBlock(config) for _ in range(config.n_layers)])
        self.final_norm = RMSNorm(config.d_model, eps=config.norm_eps)

        if config.tie_embeddings:
            self.lm_head = None
        else:
            self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x, mask=mask)
        x = self.final_norm(x)

        if self.config.tie_embeddings:
            logits = F.linear(x, self.embedding.weight)
        else:
            logits = self.lm_head(x)

        return logits


def get_architecture_preset(family_name: str) -> ChakrEvolvedLM:
    """Instantiate a model configuration from the tested architecture families."""
    family = family_name.lower().strip()
    if family in ("micro", "chakrmicro"):
        cfg = EvolvedModelConfig(
            vocab_size=4096,
            d_model=192,
            n_layers=6,
            n_heads=6,
            n_kv_heads=6,
            hidden_dim=512,
            max_seq_len=512,
            tie_embeddings=True,
        )
    elif family in ("medium", "chakrmedium"):
        cfg = EvolvedModelConfig(
            vocab_size=4096,
            d_model=384,
            n_layers=7,
            n_heads=6,
            n_kv_heads=6,
            hidden_dim=896,
            max_seq_len=512,
            tie_embeddings=True,
        )
    elif family in ("core", "chakrcore", "core-28m"):
        cfg = EvolvedModelConfig(
            vocab_size=4096,
            d_model=512,
            n_layers=8,
            n_heads=8,
            n_kv_heads=8,
            hidden_dim=1408,
            max_seq_len=512,
            rope_base=50000.0,
            tie_embeddings=True,
        )
    elif family in ("core-gqa", "chakrcore-gqa"):
        cfg = EvolvedModelConfig(
            vocab_size=4096,
            d_model=512,
            n_layers=8,
            n_heads=8,
            n_kv_heads=2,  # 4x GQA KV sharing
            hidden_dim=1408,
            max_seq_len=512,
            rope_base=50000.0,
            tie_embeddings=True,
        )
    elif family in ("large", "chakrlarge", "large-53m"):
        cfg = EvolvedModelConfig(
            vocab_size=4096,
            d_model=640,
            n_layers=10,
            n_heads=10,
            n_kv_heads=10,
            hidden_dim=1792,
            max_seq_len=512,
            rope_base=50000.0,
            tie_embeddings=True,
        )
    else:
        raise ValueError(f"Unknown architecture family: {family_name}")

    return ChakrEvolvedLM(cfg)
