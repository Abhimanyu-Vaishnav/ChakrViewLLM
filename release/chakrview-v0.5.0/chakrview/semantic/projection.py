"""
Projection and normalization modules for the Sovereign Semantic Encoder (Step 14).

Maps pooled representations into the target vector metric space and applies
unit L2 normalization for deterministic cosine similarity comparisons.
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class SemanticProjection(nn.Module):
    """
    Projects hidden representations [B, d_model] into target embedding space [B, embedding_dim].
    
    Optionally applies LayerNorm and L2 normalization:
        v_norm = \\frac{v}{\\|v\\|_2 + \\epsilon}
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        use_mlp: bool = False,
        normalize: bool = True,
        layer_norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.normalize = normalize

        if use_mlp:
            self.net = nn.Sequential(
                nn.Linear(in_dim, in_dim, bias=False),
                nn.GELU(),
                nn.LayerNorm(in_dim, eps=layer_norm_eps),
                nn.Linear(in_dim, out_dim, bias=False),
            )
        else:
            self.net = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch_size, in_dim]
        Returns:
            Output tensor [batch_size, out_dim], L2 normalized if configured.
        """
        out = self.net(x)
        if self.normalize:
            out = F.normalize(out, p=2, dim=-1, eps=1e-12)
        return out
