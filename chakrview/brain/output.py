"""
Language modeling output head with strict weight tying.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LMHead(nn.Module):
    """
    Tied Language Modeling Output Head.
    
    Shares the underlying parameter tensor with TokenEmbedding:
        logits = x @ E^T
    Zero additional parameter tensors allocated.
    """
    def __init__(self, embedding_weight: nn.Parameter) -> None:
        super().__init__()
        # Direct reference to the shared embedding weight parameter
        self.weight = embedding_weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project hidden representations to vocabulary logit space.
        
        Args:
            x: Hidden representations of shape [B, T, d_model].
            
        Returns:
            Unnormalized categorical logits of shape [B, T, vocab_size].
        """
        return F.linear(x, self.weight)
