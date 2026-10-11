"""
Root Mean Square Normalization (Pre-RMSNorm) module for ChakrView.
"""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """
    Root Mean Square Normalization.
    
    Formula:
        RMS(x) = sqrt(mean(x^2) + eps)
        RMSNorm(x) = (x / RMS(x)) * gamma
        
    Strictly bias-free, operating across the last dimension (d_model).
    """
    def __init__(self, d_model: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [..., d_model].
            
        Returns:
            Normalized tensor of shape [..., d_model].
        """
        input_dtype = x.dtype
        x_f32 = x.float()
        variance = x_f32.pow(2).mean(dim=-1, keepdim=True)
        rsqrt = torch.rsqrt(variance + self.eps)
        normalized = (x_f32 * rsqrt).to(input_dtype)
        return normalized * self.weight
