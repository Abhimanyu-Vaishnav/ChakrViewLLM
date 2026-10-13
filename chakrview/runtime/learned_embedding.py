"""
Indigenous Lightweight Learned Representation Component for ChakrView.
Location: chakrview/runtime/learned_embedding.py

Architecture:
- ChakrBiEncoder: Dual-encoder projection network mapping BPE token sequences
  into a normalized d_embed vector space.
- Mean pooling over non-padded tokens with L2 vector normalization.
- Conforms to the EmbeddingProvider abstract interface in chakrview.runtime.retrieval.
- Completely self-contained, trainable from scratch on ChakrView's 4096-vocab tokenizer.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from chakrview.runtime.retrieval import EmbeddingProvider
from chakrview.tokenizer.tokenizer import BPETokenizer
from chakrview.tokenizer.serialization import load_tokenizer_artifacts


class ChakrBiEncoder(nn.Module):
    """
    Lightweight transformer-based or feed-forward bi-encoder for semantic text embeddings.
    Maps token sequence [B, L] to dense normalized embedding [B, D].
    """

    def __init__(
        self,
        vocab_size: int = 4096,
        d_model: int = 128,
        d_embed: int = 64,
        max_seq_len: int = 256,
        n_layers: int = 2,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_embed = d_embed
        self.max_seq_len = max_seq_len
        self.n_layers = n_layers

        self.token_embeddings = nn.Embedding(vocab_size, d_model)
        self.pos_embeddings = nn.Embedding(max_seq_len, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            dim_feedforward=d_model * 2,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.proj = nn.Linear(d_model, d_embed, bias=False)
        self.norm = nn.LayerNorm(d_embed)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        input_ids: [B, L]
        attention_mask: [B, L] (1 for valid token, 0 for padding)
        Returns: [B, D] L2-normalized embeddings.
        """
        b, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(b, -1)

        x = self.token_embeddings(input_ids) + self.pos_embeddings(positions)
        
        # In PyTorch TransformerEncoder, src_key_padding_mask expects True for positions to IGNORE
        padding_mask = (attention_mask == 0) if attention_mask is not None else None
        encoded = self.encoder(x, src_key_padding_mask=padding_mask)

        # Mean pooling over valid tokens
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).float()
            sum_embeddings = torch.sum(encoded * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            pooled = sum_embeddings / sum_mask
        else:
            pooled = torch.mean(encoded, dim=1)

        projected = self.norm(self.proj(pooled))
        return F.normalize(projected, p=2, dim=-1)


class LearnedEmbeddingProvider(EmbeddingProvider):
    """
    Implements EmbeddingProvider for ChakrBiEncoder inference.
    """

    def __init__(
        self,
        model: ChakrBiEncoder,
        tokenizer: BPETokenizer,
        device: Optional[torch.device] = None,
        provider_id: str = "learned_bi_encoder_v1",
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or torch.device("cpu")
        self.model.to(self.device)
        self.model.eval()
        self._provider_id = provider_id

    @property
    def dimension(self) -> int:
        return self.model.d_embed

    @property
    def provider_id(self) -> str:
        return self._provider_id

    def embed_text(self, text: str) -> List[float]:
        tokens = self.tokenizer.encode(text, add_bos=True, add_eos=False)
        if not tokens:
            tokens = [0]
        tokens = tokens[: self.model.max_seq_len]

        input_ids = torch.tensor([tokens], dtype=torch.long, device=self.device)
        mask = torch.ones_like(input_ids)

        with torch.no_grad():
            emb = self.model(input_ids, attention_mask=mask)
            return emb[0].cpu().tolist()

    def embed_many(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_text(t) for t in texts]

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return self.embed_many(texts)

    def save_checkpoint(self, path: Path) -> None:
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "vocab_size": self.model.vocab_size,
                "d_model": self.model.d_model,
                "d_embed": self.model.d_embed,
                "max_seq_len": self.model.max_seq_len,
                "n_layers": self.model.n_layers,
                "provider_id": self._provider_id,
            },
            path,
        )

    @classmethod
    def load_checkpoint(cls, path: Path, tokenizer: BPETokenizer, device: Optional[torch.device] = None) -> "LearnedEmbeddingProvider":
        ckpt = torch.load(path, map_location=device or "cpu", weights_only=False)
        model = ChakrBiEncoder(
            vocab_size=ckpt["vocab_size"],
            d_model=ckpt["d_model"],
            d_embed=ckpt["d_embed"],
            max_seq_len=ckpt["max_seq_len"],
            n_layers=ckpt.get("n_layers", 2),
        )
        model.load_state_dict(ckpt["model_state_dict"])
        return cls(model=model, tokenizer=tokenizer, device=device, provider_id=ckpt.get("provider_id", "learned_bi_encoder_v1"))
