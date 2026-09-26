"""
Causal Cross-Entropy Loss Module for ChakrView (Phase 7).

Computes next-token prediction loss over vocabulary:
    logits:  [B, T, V]
    targets: [B, T]
Ignores PAD tokens (default ignore_index = 2) so padding positions do not contribute to loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalLoss(nn.Module):
    """
    Standard next-token prediction cross-entropy loss with PAD exclusion.
    
    Attributes:
        ignore_index: Token ID to ignore during loss computation (frozen pad_id = 2).
    """
    def __init__(self, ignore_index: int = 2) -> None:
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute mean cross-entropy loss over unmasked sequence tokens.
        
        Args:
            logits: Prediction tensor of shape [B, T, V].
            targets: Target token IDs tensor of shape [B, T].
            
        Returns:
            Scalar loss tensor.
        """
        if logits.dim() != 3:
            raise ValueError(f"Expected 3D logits [B, T, V], got {logits.dim()}D tensor")
        if targets.dim() != 2:
            raise ValueError(f"Expected 2D targets [B, T], got {targets.dim()}D tensor")

        B, T, V = logits.shape
        if targets.shape != (B, T):
            raise ValueError(
                f"Targets shape {targets.shape} does not match logits prefix {(B, T)}"
            )

        # Flatten into 2D logits and 1D targets
        flat_logits = logits.view(-1, V)
        flat_targets = targets.view(-1)

        # Cross entropy with ignore_index for PAD
        loss = F.cross_entropy(flat_logits, flat_targets, ignore_index=self.ignore_index)
        return loss
