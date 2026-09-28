"""
Evaluation metrics for the Sovereign Semantic Encoder (Step 14).

Computes information retrieval metrics:
- Recall@1
- Recall@5
- Recall@10
- Mean Reciprocal Rank (MRR)
- Cosine similarity distributions (positive vs negative pairs)
"""

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class SemanticRetrievalMetrics:
    """
    Standard information retrieval evaluation metrics.
    """
    recall_at_1: float
    recall_at_5: float
    recall_at_10: float
    mrr: float
    mean_positive_sim: float
    mean_negative_sim: float
    num_queries: int
    num_corpus_docs: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def evaluate_semantic_retrieval(
    query_embeddings: torch.Tensor,
    corpus_embeddings: torch.Tensor,
    ground_truth_doc_indices: List[int],
    top_k_list: Tuple[int, ...] = (1, 5, 10),
) -> SemanticRetrievalMetrics:
    """
    Evaluate ranking performance of dense vectors against ground truth indices.
    
    Args:
        query_embeddings: Tensor [num_queries, D], normalized.
        corpus_embeddings: Tensor [num_docs, D], normalized.
        ground_truth_doc_indices: List of target document index per query.
        top_k_list: Tuple of k values to calculate Recall@k.
    Returns:
        SemanticRetrievalMetrics with Recall@k, MRR, and similarity statistics.
    """
    num_queries = query_embeddings.size(0)
    num_docs = corpus_embeddings.size(0)

    if len(ground_truth_doc_indices) != num_queries:
        raise ValueError("Length mismatch between queries and ground truth indices.")

    # Compute full similarity matrix: [num_queries, num_docs]
    sim_matrix = torch.matmul(query_embeddings, corpus_embeddings.T).cpu().numpy()

    recalls = {k: 0.0 for k in top_k_list}
    reciprocal_ranks = []
    pos_sims = []
    neg_sims = []

    for i in range(num_queries):
        target_idx = ground_truth_doc_indices[i]
        scores = sim_matrix[i]

        # Positive similarity
        pos_sim = float(scores[target_idx])
        pos_sims.append(pos_sim)

        # Negative similarities (all other docs)
        for j in range(num_docs):
            if j != target_idx:
                neg_sims.append(float(scores[j]))

        # Rank documents descending by score with deterministic tie-breaking on index
        ranked_indices = sorted(range(num_docs), key=lambda idx: (-scores[idx], idx))

        # Find 1-based rank of target_idx
        rank = ranked_indices.index(target_idx) + 1
        reciprocal_ranks.append(1.0 / rank)

        for k in top_k_list:
            if rank <= k:
                recalls[k] += 1.0

    return SemanticRetrievalMetrics(
        recall_at_1=round(recalls.get(1, 0.0) / num_queries, 4),
        recall_at_5=round(recalls.get(5, 0.0) / num_queries, 4),
        recall_at_10=round(recalls.get(10, 0.0) / num_queries, 4),
        mrr=round(float(np.mean(reciprocal_ranks)), 4),
        mean_positive_sim=round(float(np.mean(pos_sims)), 4) if pos_sims else 0.0,
        mean_negative_sim=round(float(np.mean(neg_sims)), 4) if neg_sims else 0.0,
        num_queries=num_queries,
        num_corpus_docs=num_docs,
    )
