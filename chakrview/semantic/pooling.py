"""
Pooling abstractions for the Sovereign Semantic Encoder (Step 14).

Transforms variable-length sequence representations [B, T, D] into fixed-size
dense semantic vectors [B, D].

Architectural Rationale:
- Masked Mean Pooling is chosen as the primary default strategy. Unlike CLS pooling
  which concentrates representation on an artificial boundary token without extensive
  pre-training, masked mean pooling distributes semantic saliency across all constituent
  content tokens while strictly discarding padding tokens.
- Attention masks (1 for content, 0 for pad) ensure that padding tokens (PAD_ID = 2)
  do not skew vector magnitude, direction, or cosine similarity.
"""

from typing import Optional
import torch
import torch.nn as nn


class MeanPooling(nn.Module):
    """
    Standard unmasked average pooling across the temporal sequence dimension.
    
    Warning: If padding tokens exist in input_ids, use MaskedMeanPooling instead
    to prevent artificial padding bias.
    """

    def forward(self, token_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_embeddings: Tensor of shape [batch_size, seq_len, hidden_dim]
        Returns:
            Tensor of shape [batch_size, hidden_dim]
        """
        return torch.mean(token_embeddings, dim=1)


class MaskedMeanPooling(nn.Module):
    """
    Masked average pooling across the temporal sequence dimension.
    
    Zeros out representations at positions where attention_mask == 0 before
    accumulating, and divides strictly by the active token count per sequence:
    
        v_b = \\frac{\\sum_{t=1}^T H_{b, t} \\cdot M_{b, t}}{\\max(1, \\sum_{t=1}^T M_{b, t})}
    """

    def forward(
        self,
        token_embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            token_embeddings: Tensor of shape [batch_size, seq_len, hidden_dim]
            attention_mask: Tensor of shape [batch_size, seq_len] with 1 for content, 0 for pad.
        Returns:
            Tensor of shape [batch_size, hidden_dim]
        """
        if attention_mask is None:
            return torch.mean(token_embeddings, dim=1)

        # Expand mask from [batch_size, seq_len] to [batch_size, seq_len, hidden_dim]
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        # Sum content embeddings
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
        # Sum mask values (active tokens per sequence), clamped to at least 1e-9 to prevent divide-by-zero
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)

        return sum_embeddings / sum_mask


class CLSPooling(nn.Module):
    """
    First-token (BOS / CLS) selection pooling.
    
    Extracts representation at sequence index 0: [B, 0, D] -> [B, D].
    """

    def forward(
        self,
        token_embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            token_embeddings: Tensor of shape [batch_size, seq_len, hidden_dim]
            attention_mask: Ignored for CLS index extraction.
        Returns:
            Tensor of shape [batch_size, hidden_dim]
        """
        return token_embeddings[:, 0, :]


class PoolingLayer(nn.Module):
    """
    Unified pooling layer dispatching based on configured pooling_mode.
    """

    def __init__(self, mode: str = "masked_mean") -> None:
        super().__init__()
        self.mode = mode.lower()
        if self.mode == "masked_mean":
            self.pooler = MaskedMeanPooling()
        elif self.mode == "mean":
            self.pooler = MeanPooling()
        elif self.mode == "cls":
            self.pooler = CLSPooling()
        else:
            raise ValueError(f"Unknown pooling mode: '{mode}'. Must be 'masked_mean', 'mean', or 'cls'.")

    def forward(
        self,
        token_embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.mode == "mean":
            return self.pooler(token_embeddings)
        return self.pooler(token_embeddings, attention_mask)
