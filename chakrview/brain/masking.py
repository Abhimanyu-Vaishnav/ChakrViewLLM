"""
Causal masking module for ChakrView.
"""

import torch
import torch.nn as nn


class CausalMask(nn.Module):
    """
    Upper-triangular additive causal attention mask.
    
    Generates a tensor of shape [1, 1, T, T]:
    - 0.0 for allowed positions (j <= i)
    - -1e9 for blocked future positions (j > i)
    """
    def __init__(self, max_seq_len: int = 512) -> None:
        super().__init__()
        self.max_seq_len = max_seq_len
        mask = torch.full((max_seq_len, max_seq_len), float("-inf"))
        mask = torch.triu(mask, diagonal=1)
        # Convert -inf to float32 safe -1e9 to avoid NaN in softmax on edge backends
        mask = torch.where(torch.isneginf(mask), torch.tensor(-1e9), torch.tensor(0.0))
        self.register_buffer("mask", mask.unsqueeze(0).unsqueeze(0), persistent=False)

    def forward(self, seq_len: int) -> torch.Tensor:
        """
        Slice causal mask to the requested sequence length T.
        
        Args:
            seq_len: Integer sequence length T.
            
        Returns:
            Additive float mask of shape [1, 1, T, T].
        """
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"seq_len ({seq_len}) exceeds configured max_seq_len ({self.max_seq_len})"
            )
        return self.mask[:, :, :seq_len, :seq_len]
