"""
Contrastive learning loss functions for the Sovereign Semantic Encoder (Step 14).

Implements InfoNCE (Multiple Negatives Ranking Loss) for representation alignment
between semantic queries, positive documents, and hard/in-batch negatives.

Mathematical Formulation:
Given normalized query representations Q \\in \\mathbb{R}^{B \\times D} and
positive document representations P \\in \\mathbb{R}^{B \\times D}:

1. In-Batch Negatives Mode:
   The similarity matrix S \\in \\mathbb{R}^{B \\times B} is computed via:
       S_{i, j} = \\frac{Q_i \\cdot P_j}{\\tau}
   where \\tau > 0 is the temperature scaling hyperparameter.
   For each query i, positive index is i, and all other items j \\neq i in the batch
   serve as negative distractors. The objective is multi-class categorical cross-entropy:
       \\mathcal{L} = \\frac{1}{B} \\sum_{i=1}^B -\\log \\frac{\\exp(S_{i, i})}{\\sum_{j=1}^B \\exp(S_{i, j})}

2. Explicit Hard Negatives Mode:
   When explicit negative candidates N \\in \\mathbb{R}^{B \\times K \\times D} are provided,
   the logits for query i are formed by concatenating the positive similarity with the K
   negative similarities:
       \\text{logits}_i = \\left[ \\frac{Q_i \\cdot P_i}{\\tau}, \\frac{Q_i \\cdot N_{i, 1}}{\\tau}, \\dots, \\frac{Q_i \\cdot N_{i, K}}{\\tau} \\right]
   with ground-truth target label 0.
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    """
    InfoNCE / Multiple Negatives Ranking Loss (MNRL).
    
    Attributes:
        temperature: Softmax scaling constant \\tau (default: 0.05).
    """

    def __init__(self, temperature: float = 0.05) -> None:
        super().__init__()
        if temperature <= 0.0:
            raise ValueError(f"temperature must be positive, got {temperature}")
        self.temperature = temperature

    def forward(
        self,
        query_embeddings: torch.Tensor,
        positive_embeddings: torch.Tensor,
        negative_embeddings: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute InfoNCE contrastive loss.
        
        Args:
            query_embeddings: Tensor [B, D], L2 normalized.
            positive_embeddings: Tensor [B, D], L2 normalized.
            negative_embeddings: Optional Tensor [B, K, D] or [B * K, D], L2 normalized.
        Returns:
            Scalar cross-entropy loss tensor.
        """
        b = query_embeddings.size(0)
        device = query_embeddings.device

        if negative_embeddings is None:
            # 1. In-batch negatives: S[i, j] = (Q[i] . P[j]) / tau
            similarity_matrix = torch.matmul(query_embeddings, positive_embeddings.T) / self.temperature
            labels = torch.arange(b, dtype=torch.long, device=device)
            loss = F.cross_entropy(similarity_matrix, labels)
            return loss

        # 2. Explicit negatives provided
        if negative_embeddings.dim() == 2:
            # Reshape [B * K, D] -> [B, K, D] if evenly divisible
            k = negative_embeddings.size(0) // b
            neg_reshaped = negative_embeddings.view(b, k, -1)
        elif negative_embeddings.dim() == 3:
            neg_reshaped = negative_embeddings
        else:
            raise ValueError(f"Unsupported negative_embeddings shape: {negative_embeddings.shape}")

        # Compute positive similarity: [B, 1]
        pos_sim = torch.sum(query_embeddings * positive_embeddings, dim=-1, keepdim=True) / self.temperature

        # Compute negative similarities: [B, K]
        # Query: [B, 1, D], Negatives: [B, K, D] -> bmm -> [B, 1, K] -> squeeze -> [B, K]
        neg_sim = torch.bmm(neg_reshaped, query_embeddings.unsqueeze(-1)).squeeze(-1) / self.temperature

        # Concatenate: [B, 1 + K]
        logits = torch.cat([pos_sim, neg_sim], dim=-1)
        # Ground truth target is index 0 for all batch elements
        labels = torch.zeros(b, dtype=torch.long, device=device)
        return F.cross_entropy(logits, labels)
