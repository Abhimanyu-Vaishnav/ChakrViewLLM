"""
Weighted Target Loss Module for ChakrView SFT.

Supports token-level weighted cross-entropy:
- Baseline: uniform weight (1.0) on all active targets.
- Answer-emphasis: higher weight on answer tokens (e.g. 5.0), 1.0 on template tokens.
- Answer-only: 1.0 on answer tokens, 0.0 on template tokens (or masked via -100).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedCausalLoss(nn.Module):
    """
    Computes per-token weighted cross entropy:
      loss = sum(w_i * CE(logits_i, targets_i)) / sum(w_i)
    where ignore_index positions have effective loss 0 and weight 0.
    """

    def __init__(self, ignore_index: int = -100) -> None:
        super().__init__()
        self.ignore_index = ignore_index

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        logits: [B, T, V] or [N, V]
        targets: [B, T] or [N]
        weights: [B, T] or [N] (float)
        """
        if logits.dim() == 3:
            B, T, V = logits.shape
            flat_logits = logits.reshape(-1, V)
            flat_targets = targets.reshape(-1)
            flat_weights = weights.reshape(-1) if weights is not None else None
        else:
            flat_logits = logits
            flat_targets = targets
            flat_weights = weights

        # Per-token unreduced cross entropy
        ce = F.cross_entropy(flat_logits, flat_targets, ignore_index=self.ignore_index, reduction="none")

        if flat_weights is None:
            # Standard mean over unmasked tokens
            valid_mask = (flat_targets != self.ignore_index).float()
            denom = valid_mask.sum().clamp(min=1.0)
            return (ce * valid_mask).sum() / denom

        # Mask weights where target is ignore_index
        active_weights = flat_weights * (flat_targets != self.ignore_index).float()
        denom = active_weights.sum().clamp(min=1.0)
        return (ce * active_weights).sum() / denom
