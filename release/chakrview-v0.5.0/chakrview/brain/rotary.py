"""
Rotary Position Embeddings (RoPE) for ChakrView.
"""

import torch
import torch.nn as nn


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embeddings (RoPE) applied to query and key head slices.
    
    Operates on [B, H, T, d_head] tensors via pairwise 2D Givens rotations.
    """
    def __init__(self, dim: int, max_seq_len: int = 512, theta: float = 10000.0) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"Rotary dimension must be even, got {dim}")
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta
        
        # Precompute frequencies: theta_k = theta^(-2k / dim)
        indices = torch.arange(0, dim, 2).float()
        freqs = 1.0 / (theta ** (indices / dim))
        self.register_buffer("freqs", freqs, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        """Precompute cosine and sine tables up to seq_len."""
        t = torch.arange(seq_len, dtype=torch.float32)
        # Outer product: [T, dim/2]
        freqs_matrix = torch.outer(t, self.freqs)
        cos = torch.cos(freqs_matrix)  # [T, dim/2]
        sin = torch.sin(freqs_matrix)  # [T, dim/2]
        self.register_buffer("cos_cache", cos, persistent=False)
        self.register_buffer("sin_cache", sin, persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int, offset: int = 0) -> torch.Tensor:
        """
        Apply RoPE to head tensor.
        
        Args:
            x: Tensor of shape [B, H, T, d_head]
            seq_len: Current sequence length (T)
            offset: Starting sequence position offset (0 for full sequence, pos for cached decode)
            
        Returns:
            Rotated tensor of shape [B, H, T, d_head]
        """
        if offset + seq_len > self.max_seq_len:
            raise ValueError(
                f"offset + seq_len ({offset + seq_len}) exceeds configured max_seq_len ({self.max_seq_len})"
            )
        B, H, T, D = x.shape
        x_paired = x.view(B, H, T, D // 2, 2)
        x0 = x_paired[..., 0]
        x1 = x_paired[..., 1]
        
        cos = self.cos_cache[offset : offset + seq_len].unsqueeze(0).unsqueeze(0)  # [1, 1, T, D//2]
        sin = self.sin_cache[offset : offset + seq_len].unsqueeze(0).unsqueeze(0)  # [1, 1, T, D//2]
        
        # Pairwise 2D rotation:
        # [x0 * cos - x1 * sin, x0 * sin + x1 * cos]
        rot_x0 = x0 * cos - x1 * sin
        rot_x1 = x0 * sin + x1 * cos
        
        return torch.stack((rot_x0, rot_x1), dim=-1).flatten(-2)

