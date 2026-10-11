"""
Sovereign Semantic Encoder neural core (Step 14).

A lightweight, bidirectional transformer encoder designed for CPU-friendly
dense semantic vector generation, contrastive learning, and embedding retrieval.
"""

import math
from typing import Dict, List, Optional, Tuple, Union, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from chakrview.semantic.config import SemanticEncoderConfig
from chakrview.semantic.pooling import PoolingLayer
from chakrview.semantic.projection import SemanticProjection


class TransformerEncoderBlock(nn.Module):
    """
    Standard bidirectional Transformer encoder block with Pre-LayerNorm and residual connections.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
            bias=False,
        )
        self.ln2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model, bias=False),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            key_padding_mask: Bool tensor [batch_size, seq_len] where True indicates positions to ignore.
        """
        # 1. Pre-LN Self-Attention
        norm_x = self.ln1(x)
        attn_out, _ = self.attn(
            norm_x,
            norm_x,
            norm_x,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = x + attn_out

        # 2. Pre-LN Feed-Forward
        norm_x = self.ln2(x)
        ffn_out = self.ffn(norm_x)
        x = x + ffn_out
        return x


class SemanticEncoder(nn.Module):
    """
    Trainable neural semantic encoder for ChakrView.
    
    Architecture Flow:
        Input Token IDs [B, T]
            ↓
        Token + Positional Embedding
            ↓
        N x Bidirectional Transformer Encoder Blocks
            ↓
        Final Sequence LayerNorm
            ↓
        Masked Mean Pooling [B, d_model]
            ↓
        Linear Projection Head [B, embedding_dim]
            ↓
        L2 Unit Normalization
            ↓
        Dense Semantic Vector [B, embedding_dim]
    """

    def __init__(self, config: Optional[SemanticEncoderConfig] = None) -> None:
        super().__init__()
        self.config = config or SemanticEncoderConfig()

        # Token & Positional Embeddings
        self.token_embeddings = nn.Embedding(
            num_embeddings=self.config.vocab_size,
            embedding_dim=self.config.d_model,
            padding_idx=self.config.pad_token_id,
        )
        self.position_embeddings = nn.Embedding(
            num_embeddings=self.config.max_seq_len,
            embedding_dim=self.config.d_model,
        )
        self.dropout = nn.Dropout(self.config.dropout)

        # Transformer Encoder Blocks
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(
                d_model=self.config.d_model,
                n_heads=self.config.n_heads,
                d_ff=self.config.d_ff,
                dropout=self.config.dropout,
                layer_norm_eps=self.config.layer_norm_eps,
            )
            for _ in range(self.config.n_layers)
        ])

        # Final LayerNorm
        self.final_ln = nn.LayerNorm(self.config.d_model, eps=self.config.layer_norm_eps)

        # Pooling & Projection
        self.pooling = PoolingLayer(mode=self.config.pooling_mode)
        self.projection = SemanticProjection(
            in_dim=self.config.d_model,
            out_dim=self.config.embedding_dim,
            normalize=self.config.normalize_embeddings,
            layer_norm_eps=self.config.layer_norm_eps,
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights with truncated normal distributions."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()

    @property
    def parameter_count(self) -> int:
        """Total trainable parameters in the semantic encoder."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode discrete token sequences into dense semantic vectors.
        
        Args:
            input_ids: Long tensor of shape [batch_size, seq_len]
            attention_mask: Binary tensor of shape [batch_size, seq_len] (1 for content, 0 for pad)
        Returns:
            Dense semantic vectors of shape [batch_size, embedding_dim]
        """
        batch_size, seq_len = input_ids.shape
        if seq_len > self.config.max_seq_len:
            raise ValueError(
                f"Input sequence length {seq_len} exceeds max_seq_len {self.config.max_seq_len}"
            )

        if attention_mask is None:
            # Generate mask based on pad_token_id (1 where token != pad_token_id)
            attention_mask = (input_ids != self.config.pad_token_id).long()

        # Build position IDs [0, 1, ..., seq_len - 1]
        device = input_ids.device
        position_ids = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0).expand(batch_size, -1)

        # Embeddings
        tok_emb = self.token_embeddings(input_ids)
        pos_emb = self.position_embeddings(position_ids)
        h = self.dropout(tok_emb + pos_emb)

        # PyTorch MultiheadAttention key_padding_mask: True indicates position SHOULD BE IGNORED
        key_padding_mask = (attention_mask == 0)

        # Transformer blocks
        for block in self.blocks:
            h = block(h, key_padding_mask=key_padding_mask)

        h = self.final_ln(h)

        # Pooling: [B, T, D] -> [B, D]
        pooled = self.pooling(h, attention_mask=attention_mask)

        # Projection & L2 Normalization: [B, D] -> [B, embedding_dim]
        projected = self.projection(pooled)
        return projected

    def encode_text(
        self,
        tokenizer: Any,
        text: str,
        max_len: Optional[int] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Convenience method to encode a single text string."""
        dev = device or next(self.parameters()).device
        limit = max_len or self.config.max_seq_len
        token_ids = tokenizer.encode(text, add_bos=True, add_eos=True)[:limit]
        if not token_ids:
            token_ids = [self.config.pad_token_id]

        input_tensor = torch.tensor([token_ids], dtype=torch.long, device=dev)
        mask = (input_tensor != self.config.pad_token_id).long()

        self.eval()
        with torch.no_grad():
            vec = self.forward(input_tensor, attention_mask=mask)
        return vec[0]

    def encode_batch(
        self,
        tokenizer: Any,
        texts: List[str],
        max_len: Optional[int] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Convenience method to encode multiple text strings with dynamic padding."""
        dev = device or next(self.parameters()).device
        limit = max_len or self.config.max_seq_len

        all_ids = []
        for t in texts:
            ids = tokenizer.encode(t, add_bos=True, add_eos=True)[:limit]
            if not ids:
                ids = [self.config.pad_token_id]
            all_ids.append(ids)

        max_batch_len = max(len(ids) for ids in all_ids)
        padded_ids = []
        masks = []
        for ids in all_ids:
            pad_needed = max_batch_len - len(ids)
            padded_ids.append(ids + [self.config.pad_token_id] * pad_needed)
            masks.append([1] * len(ids) + [0] * pad_needed)

        input_tensor = torch.tensor(padded_ids, dtype=torch.long, device=dev)
        mask_tensor = torch.tensor(masks, dtype=torch.long, device=dev)

        self.eval()
        with torch.no_grad():
            vecs = self.forward(input_tensor, attention_mask=mask_tensor)
        return vecs
