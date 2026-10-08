"""Step 473 & 474: Neural Cognitive Routing Gateway Implementation.

Differentiable bidirectional gateway connecting language representations (d_model=192)
with the relational cognitive core (d_model=96) without modifying the canonical baseline.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from chakrview.brain.normalization import RMSNorm


class NeuralCognitiveRoutingGateway(nn.Module):
    """
    Bidirectional Gated Neural Cognitive Routing Gateway.
    
    Transforms:
      Forward:  Language representations [B, T, 192] -> Cognitive space [B, T, 96]
      Reverse:  Cognitive space [B, T, 96] -> Language representations [B, T, 192]
    """
    def __init__(
        self,
        d_lang: int = 192,
        d_cog: int = 96,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.d_lang = d_lang
        self.d_cog = d_cog
        
        # Forward Gateway (Lang -> Cog)
        self.norm_fwd = RMSNorm(d_lang, eps=eps)
        self.w_down = nn.Linear(d_lang, d_cog, bias=False)
        self.w_gate_fwd = nn.Linear(d_lang, d_cog, bias=False)
        
        # Reverse Gateway (Cog -> Lang)
        self.norm_rev = RMSNorm(d_cog, eps=eps)
        self.w_up = nn.Linear(d_cog, d_lang, bias=False)
        self.w_gate_rev = nn.Linear(d_cog, d_lang, bias=False)
        
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        std = 0.02
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.normal_(p, mean=0.0, std=std)

    @property
    def total_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def route_to_cognition(self, h_lang: torch.Tensor) -> torch.Tensor:
        """
        Forward routing: Language representation -> Cognitive representation.
        h_lang: [B, T, 192] or [B, 192]
        Returns: [B, T, 96] or [B, 96]
        """
        h_norm = self.norm_fwd(h_lang)
        gate = torch.sigmoid(self.w_gate_fwd(h_norm))
        proj = self.w_down(h_norm)
        return gate * proj

    def route_to_language(self, h_cog: torch.Tensor) -> torch.Tensor:
        """
        Reverse routing: Cognitive representation -> Language representation.
        h_cog: [B, T, 96] or [B, 96]
        Returns: [B, T, 192] or [B, 192]
        """
        h_norm = self.norm_rev(h_cog)
        gate = torch.sigmoid(self.w_gate_rev(h_norm))
        proj = self.w_up(h_norm)
        return gate * proj
