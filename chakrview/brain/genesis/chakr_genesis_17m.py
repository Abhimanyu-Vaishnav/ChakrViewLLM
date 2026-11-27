"""
ChakrGenesis-17M Neural Foundation Model Architecture.
Mission CV-GENESIS-NR-009 Phase 5 Pilot.

Architecture Specification:
- Parameters: ~17.06M total (~15.49M non-tied)
- Model Dimension (d_model): 384
- Layers: 10
- Query Heads (n_q_heads): 12 (head_dim = 32)
- KV Heads (n_kv_heads): 3 (Grouped-Query Attention with 4:1 query-to-KV ratio)
- Feed-Forward Dimension (hidden_dim): 1024 (SwiGLU)
- Positional Encoding: Rotary Position Embedding (RoPE, theta=10000.0)
- Normalization: RMSNorm (eps=1e-5)
- Weight Tying: Token embeddings tied to output projection head
- Vocabulary: 4,096 canonical BPE tokens
"""

from __future__ import annotations
import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class Genesis17MConfig:
    def __init__(
        self,
        vocab_size: int = 4096,
        d_model: int = 384,
        n_layers: int = 10,
        n_q_heads: int = 12,
        n_kv_heads: int = 3,
        hidden_dim: int = 1024,
        max_seq_len: int = 1024,
        rms_norm_eps: float = 1e-5,
        rope_theta: float = 10000.0,
        dropout: float = 0.0,
        tie_embeddings: bool = True,
        bos_token_id: int = 0,
        eos_token_id: int = 1,
        pad_token_id: int = 2
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
        if seq_len > self.cos_cached.shape[0]:
            t = torch.arange(seq_len, dtype=torch.float32, device=x.device)
            freqs = torch.outer(t, self.inv_freq.to(x.device))
            return freqs.cos(), freqs.sin()
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1]
    x1, x2 = x[..., :d//2], x[..., d//2:]
    cos = cos.to(dtype=x.dtype).unsqueeze(0).unsqueeze(0)
    sin = sin.to(dtype=x.dtype).unsqueeze(0).unsqueeze(0)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class GroupedQueryAttention(nn.Module):
    def __init__(self, config: Genesis17MConfig):
        super().__init__()
        self.d_model = config.d_model
        self.n_q_heads = config.n_q_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.d_model // config.n_q_heads
        self.num_reps = config.n_q_heads // config.n_kv_heads

        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k_proj = nn.Linear(config.d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_q_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # Expand KV heads to match Q heads (GQA)
        if self.num_reps > 1:
            k = k.repeat_interleave(self.num_reps, dim=1)
            v = v.repeat_interleave(self.num_reps, dim=1)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask.to(dtype=scores.dtype)

        probs = F.softmax(scores, dim=-1, dtype=scores.dtype)
        output = torch.matmul(probs, v)
        output = output.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(output)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, hidden_dim, bias=False)
        self.up_proj = nn.Linear(d_model, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    def __init__(self, config: Genesis17MConfig):
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.attn = GroupedQueryAttention(config)
        self.ffn_norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.ffn = SwiGLU(config.d_model, config.hidden_dim)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), cos, sin, mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class ChakrGenesis17M(nn.Module):
    def __init__(self, config: Optional[Genesis17MConfig] = None):
        super().__init__()
        self.config = config or Genesis17MConfig()
        self.embedding = nn.Embedding(self.config.vocab_size, self.config.d_model)
        self.rope = RotaryEmbedding(
            dim=self.config.d_model // self.config.n_q_heads,
            max_seq_len=self.config.max_seq_len,
            theta=self.config.rope_theta
        )
        self.layers = nn.ModuleList([TransformerBlock(self.config) for _ in range(self.config.n_layers)])
        self.final_norm = RMSNorm(self.config.d_model, eps=self.config.rms_norm_eps)
        self.lm_head = nn.Linear(self.config.d_model, self.config.vocab_size, bias=False)

        if self.config.tie_embeddings:
            self.lm_head.weight = self.embedding.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T = input_ids.shape
        x = self.embedding(input_ids)
        cos, sin = self.rope(x, T)

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
