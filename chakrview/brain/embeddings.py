"""
Token embedding module for ChakrView Neural Core.
"""

import torch
import torch.nn as nn
from chakrview.brain.config import ModelConfig


class TokenEmbedding(nn.Module):
    """
    Embedding layer mapping discrete token indices into continuous hidden states.
    
    Attributes:
        weight: Embedding parameter matrix E in R[vocab_size, d_model].
    """
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.vocab_size = config.vocab_size
        self.d_model = config.d_model
        # Direct parameter tensor allocation
        self.weight = nn.Parameter(torch.empty(config.vocab_size, config.d_model))

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Embed discrete token indices.
        
        Args:
            input_ids: Integer tensor of shape [B, T].
            
        Returns:
            Hidden representations of shape [B, T, d_model].
        """
        if (input_ids < 0).any() or (input_ids >= self.vocab_size).any():
            raise ValueError(
                f"Token ID out of bounds. Valid vocabulary range is [0, {self.vocab_size - 1}]."
            )
        return torch.nn.functional.embedding(input_ids, self.weight)
