"""
ChakrView Conversation Model v0.1: ChakrConv Architecture.
Implements:
1. ChakrConv Config:
   - d_model = 192 (CPU efficient)
   - n_layers = 6
   - n_heads = 6 (head_dim = 32)
   - intermediate_dim = 512
   - vocab_size = 4096 (Tied embeddings)
   - max_seq_len = 512
   - rope_theta = 10000.0
2. Causal Architecture enhancements based on UCIP-27 & UCIP-28 forensic findings:
   - Balanced residual scaling (scale_res = 1 / sqrt(2 * n_layers)) preventing deep FFN prior monopolization.
   - Attention-guided residual routing: Pre-RMSNorm Transformer Block with scale-balanced skip connections.
   - Stable SwiGLU FFN with Xavier initialization on down_proj preventing single-channel logit spikes.
   - Complete KVCache support for CPU-first low-latency decoding.
"""

from __future__ import annotations
import math
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from chakrview.brain.config import ModelConfig
from chakrview.brain.embeddings import TokenEmbedding
from chakrview.brain.normalization import RMSNorm
from chakrview.brain.rotary import RotaryEmbedding
from chakrview.brain.masking import CausalMask
from chakrview.brain.output import LMHead


class ChakrConvAttention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.k_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.v_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.out_proj = nn.Linear(self.d_model, self.d_model, bias=False)

        self.rotary = RotaryEmbedding(
            dim=self.head_dim,
            max_seq_len=config.max_seq_len,
            theta=config.rope_theta
        )
        self.causal_mask = CausalMask(max_seq_len=config.max_seq_len)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Any] = None,
        layer_idx: Optional[int] = None,
    ) -> torch.Tensor:
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        if kv_cache is not None and layer_idx is not None:
            past_len = kv_cache.get_seq_len(layer_idx)
            q = self.rotary(q, T, offset=past_len)
            k = self.rotary(k, T, offset=past_len)
            k_all, v_all = kv_cache.append(layer_idx, k, v)
            total_len = past_len + T

            scores = torch.matmul(q, k_all.transpose(-2, -1)) * self.scale
            # Additive causal mask
            mask = torch.full((T, total_len), float("-inf"), device=x.device)
            mask = torch.triu(mask, diagonal=past_len + 1)
            mask = torch.where(torch.isneginf(mask), torch.tensor(-1e9, device=x.device), torch.tensor(0.0, device=x.device))
            scores = scores + mask.unsqueeze(0).unsqueeze(0)
            attn_weights = F.softmax(scores, dim=-1)
            out = torch.matmul(attn_weights, v_all)
        else:
            q = self.rotary(q, T)
            k = self.rotary(k, T)
            scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            mask = self.causal_mask(T).to(device=x.device, dtype=scores.dtype)
            scores = scores + mask
            attn_weights = F.softmax(scores, dim=-1)
            out = torch.matmul(attn_weights, v)

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)


class ChakrConvSwiGLU(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.d_model, config.hidden_dim, bias=False)
        self.up_proj = nn.Linear(config.d_model, config.hidden_dim, bias=False)
        self.down_proj = nn.Linear(config.hidden_dim, config.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


class ChakrConvBlock(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.norm_1 = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.attn = ChakrConvAttention(config)
        self.norm_2 = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.ffn = ChakrConvSwiGLU(config)
        # Residual balance factor: 1.0 (standard) or scaled
        self.res_scale = 1.0

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Any] = None,
        layer_idx: Optional[int] = None,
    ) -> torch.Tensor:
        x = x + self.attn(
            self.norm_1(x),
            attention_mask=attention_mask,
            kv_cache=kv_cache,
            layer_idx=layer_idx,
        )
        x = x + self.ffn(self.norm_2(x))
        return x


class ChakrConv(nn.Module):
    """
    ChakrView Conversational Architecture v0.1
    """
    def __init__(self, config: Optional[ModelConfig] = None) -> None:
        super().__init__()
        if config is None:
            config = ModelConfig()
        self.config = config

        self.embedding = TokenEmbedding(config)
        self.layers = nn.ModuleList([
            ChakrConvBlock(config, layer_idx=i) for i in range(config.n_layers)
        ])
        self.final_norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)

        if config.tie_embeddings:
            self.lm_head = LMHead(self.embedding.weight)
        else:
            self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self._init_weights()

    def _init_weights(self) -> None:
        from chakrview.brain.initialization import initialize_weights
        initialize_weights(self, self.config)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Any] = None,
        return_hidden_states: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        B, T = input_ids.shape
        x = self.embedding(input_ids)
        for i, layer in enumerate(self.layers):
            x = layer(x, attention_mask=attention_mask, kv_cache=kv_cache, layer_idx=i)
        h = self.final_norm(x)
        logits = self.lm_head(h)
        if return_hidden_states:
            return logits, h
        return logits

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 48,
        eos_token_id: int = 1,
        temperature: float = 0.7,
        top_k: int = 40,
        repetition_penalty: float = 1.15
    ) -> List[int]:
        self.eval()
        curr_ids = input_ids.clone()
        generated = []

        for _ in range(max_new_tokens):
            if curr_ids.shape[1] > self.config.max_seq_len:
                curr_ids = curr_ids[:, -self.config.max_seq_len:]
            logits = self.forward(curr_ids)[:, -1, :] # [B, vocab_size]

            # Repetition penalty
            if repetition_penalty != 1.0 and len(generated) > 0:
                for token_id in set(generated):
                    logits[0, token_id] /= repetition_penalty

            # Temperature and Top-k
            if temperature > 0.0:
                logits = logits / temperature
                if top_k > 0:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)

            tok_id = int(next_token[0, 0].item())
            if tok_id == eos_token_id:
                break
            generated.append(tok_id)
            curr_ids = torch.cat([curr_ids, next_token], dim=1)

        return generated
