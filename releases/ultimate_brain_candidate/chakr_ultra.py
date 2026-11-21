"""
ChakrUltra: Next-Generation Gated Residual Neural Core with Grouped-Query Attention (GQA).
Architecture Innovations:
- Grouped-Query Attention (GQA): 8 query heads, 2 key/value heads (group ratio = 4:1) for fast CPU inference.
- Rotary Position Embeddings (RoPE) with causal additive masking.
- Gated residual connections: x = x + gamma_attn * Attn(RMSNorm(x)) + gamma_ffn * SwiGLU(RMSNorm(x)).
- Bias-free SwiGLU feed-forward networks with intermediate dimension = 682.
- Tied embedding / output head (Vocab = 4,096, d_model = 256, 6 layers).
- Fully persistent KVCache implementation for streaming CPU generation.
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


class ChakrUltraAttention(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        n_q_heads: int = 8,
        n_kv_heads: int = 2,
        max_seq_len: int = 1024,
        rope_theta: float = 10000.0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_q_heads = n_q_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_q_heads  # 32
        self.n_rep = n_q_heads // n_kv_heads  # 4
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(d_model, n_q_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.rotary = RotaryEmbedding(
            dim=self.head_dim,
            max_seq_len=max_seq_len,
            theta=rope_theta
        )
        self.causal_mask = CausalMask(max_seq_len=max_seq_len)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[Any] = None,
        layer_idx: Optional[int] = None,
    ) -> torch.Tensor:
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_q_heads, self.head_dim).transpose(1, 2)  # [B, H_q, T, D]
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2) # [B, H_kv, T, D]
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2) # [B, H_kv, T, D]

        if kv_cache is not None and layer_idx is not None:
            past_len = kv_cache.get_seq_len(layer_idx)
            q = self.rotary(q, T, offset=past_len)
            k = self.rotary(k, T, offset=past_len)
            k_all, v_all = kv_cache.append(layer_idx, k, v)
            total_len = past_len + T
            
            # Repeat KV heads for GQA
            if self.n_rep > 1:
                k_all = k_all.repeat_interleave(self.n_rep, dim=1)
                v_all = v_all.repeat_interleave(self.n_rep, dim=1)

            scores = torch.matmul(q, k_all.transpose(-2, -1)) * self.scale
            mask = torch.full((T, total_len), float("-inf"), device=x.device)
            mask = torch.triu(mask, diagonal=past_len + 1)
            mask = torch.where(torch.isneginf(mask), torch.tensor(-1e9, device=x.device), torch.tensor(0.0, device=x.device))
            scores = scores + mask.unsqueeze(0).unsqueeze(0)
            attn_weights = F.softmax(scores, dim=-1)
            out = torch.matmul(attn_weights, v_all)
        else:
            q = self.rotary(q, T)
            k = self.rotary(k, T)

            if self.n_rep > 1:
                k = k.repeat_interleave(self.n_rep, dim=1)
                v = v.repeat_interleave(self.n_rep, dim=1)

            scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            mask = self.causal_mask(T).to(device=x.device, dtype=scores.dtype)
            scores = scores + mask
            attn_weights = F.softmax(scores, dim=-1)
            out = torch.matmul(attn_weights, v)

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)


class ChakrUltraSwiGLU(nn.Module):
    def __init__(self, d_model: int = 256, hidden_dim: int = 682) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(d_model, hidden_dim, bias=False)
        self.up_proj = nn.Linear(d_model, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


class ChakrUltraBlock(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        n_q_heads: int = 8,
        n_kv_heads: int = 2,
        hidden_dim: int = 682,
        max_seq_len: int = 1024,
        layer_idx: int = 0,
        n_layers: int = 6
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.norm_1 = RMSNorm(d_model, eps=1e-5)
        self.attn = ChakrUltraAttention(
            d_model=d_model,
            n_q_heads=n_q_heads,
            n_kv_heads=n_kv_heads,
            max_seq_len=max_seq_len
        )
        self.norm_2 = RMSNorm(d_model, eps=1e-5)
        self.ffn = ChakrUltraSwiGLU(d_model=d_model, hidden_dim=hidden_dim)

        # Gated residual scaling factor initialized to 1 / sqrt(2 * n_layers)
        scale_init = 1.0 / math.sqrt(2.0 * n_layers)
        self.gamma_attn = nn.Parameter(torch.full((1,), scale_init))
        self.gamma_ffn = nn.Parameter(torch.full((1,), scale_init))

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[Any] = None,
        layer_idx: Optional[int] = None,
    ) -> torch.Tensor:
        # Pre-norm with gated residual pathways
        x = x + self.gamma_attn * self.attn(self.norm_1(x), kv_cache=kv_cache, layer_idx=layer_idx)
        x = x + self.gamma_ffn * self.ffn(self.norm_2(x))
        return x


class ChakrUltra(nn.Module):
    def __init__(
        self,
        vocab_size: int = 4096,
        d_model: int = 256,
        n_layers: int = 6,
        n_q_heads: int = 8,
        n_kv_heads: int = 2,
        hidden_dim: int = 682,
        max_seq_len: int = 1024,
        tie_embeddings: bool = True
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len

        # Token embedding
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            ChakrUltraBlock(
                d_model=d_model,
                n_q_heads=n_q_heads,
                n_kv_heads=n_kv_heads,
                hidden_dim=hidden_dim,
                max_seq_len=max_seq_len,
                layer_idx=i,
                n_layers=n_layers
            ) for i in range(n_layers)
        ])
        self.final_norm = RMSNorm(d_model, eps=1e-5)

        if tie_embeddings:
            self.lm_head = LMHead(self.embedding.weight)
        else:
            self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        self._init_weights()

    def _init_weights(self) -> None:
        std = 0.02
        res_std = std / math.sqrt(2.0 * self.n_layers)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=std)
        for layer in self.layers:
            nn.init.normal_(layer.attn.q_proj.weight, mean=0.0, std=std)
            nn.init.normal_(layer.attn.k_proj.weight, mean=0.0, std=std)
            nn.init.normal_(layer.attn.v_proj.weight, mean=0.0, std=std)
            nn.init.normal_(layer.attn.out_proj.weight, mean=0.0, std=res_std)
            nn.init.normal_(layer.ffn.gate_proj.weight, mean=0.0, std=std)
            nn.init.normal_(layer.ffn.up_proj.weight, mean=0.0, std=std)
            nn.init.normal_(layer.ffn.down_proj.weight, mean=0.0, std=res_std)
            nn.init.ones_(layer.norm_1.weight)
            nn.init.ones_(layer.norm_2.weight)
        nn.init.ones_(self.final_norm.weight)

    def forward(
        self,
        input_ids: torch.Tensor,
        kv_cache: Optional[Any] = None,
        return_hidden_states: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        B, T = input_ids.shape
        x = self.embedding(input_ids)
        for i, layer in enumerate(self.layers):
            x = layer(x, kv_cache=kv_cache, layer_idx=i)
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
            if curr_ids.shape[1] > self.max_seq_len:
                curr_ids = curr_ids[:, -self.max_seq_len:]
            logits = self.forward(curr_ids)[:, -1, :]

            if repetition_penalty != 1.0 and len(generated) > 0:
                for token_id in set(generated):
                    logits[0, token_id] /= repetition_penalty

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
