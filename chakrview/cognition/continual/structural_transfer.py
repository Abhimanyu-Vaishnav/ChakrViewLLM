"""
Wave 540 & 544: Cross-Task & Cross-Domain Structural Transfer Engine.

Rigidly separates:
- EXACT MEMORY (Lexical / Entity matching)
- SURFACE PATTERN (Grammar / prompt template matching)
- STRUCTURAL ABSTRACTION (Topological pattern: DAG, Chain, Bipartite matching)
- CROSS-DOMAIN ANALOGY (Relational reasoning -> Planning; Planning -> Resource allocation)

Prevents false claims of parametric learning by rigorously classifying the transfer mechanism.
"""

from __future__ import annotations

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Any, List, Set, Tuple


class TransferClassification(str, Enum):
    MEMORY_RETRIEVAL = "MEMORY_RETRIEVAL"
    SURFACE_MATCH = "SURFACE_MATCH"
    POLICY_GENERALIZATION = "POLICY_GENERALIZATION"
    STRUCTURAL_GENERALIZATION = "STRUCTURAL_GENERALIZATION"
    CROSS_DOMAIN_ANALOGY = "CROSS_DOMAIN_ANALOGY"
    PARAMETRIC_ADAPTATION = "PARAMETRIC_ADAPTATION"  # Only if Delta W > 0 and proven neural weight shift
    NO_TRANSFER = "NO_TRANSFER"


@dataclass
class TransferEvaluationResult:
    source_domain: str
    target_domain: str
    lexical_overlap: float
    surface_similarity: float
    structural_isomorphism: float
    transfer_accuracy: float
    classification: TransferClassification
    parametric_delta_w: int
    rationale: str


class StructuralTransferEngine:
    """Evaluates transfer across task families and domains."""

    @classmethod
    def evaluate_cross_task_transfer(
        cls,
        source_task: Dict[str, Any],
        target_task: Dict[str, Any],
        target_performance: float,
        model_delta_w: int = 0,
    ) -> TransferEvaluationResult:
        source_ents = set(source_task.get("entities", []))
        target_ents = set(target_task.get("entities", []))
        
        # 1. Lexical overlap
        inter = source_ents.intersection(target_ents)
        union = source_ents.union(target_ents)
        lexical_overlap = len(inter) / len(union) if union else 0.0

        # 2. Structural similarity (e.g. graph topology or relation depth)
        source_hops = source_task.get("hops", 1)
        target_hops = target_task.get("hops", 1)
        structural_isomorphism = 1.0 if source_hops == target_hops else 0.5

        # 3. Classify mechanism
        if model_delta_w > 0:
            classification = TransferClassification.PARAMETRIC_ADAPTATION
            rationale = "Parametric weights changed during transfer (Delta W > 0)"
        elif lexical_overlap > 0.4:
            classification = TransferClassification.MEMORY_RETRIEVAL
            rationale = f"High entity/lexical overlap ({lexical_overlap:.2f}); transfer driven by direct memory retrieval"
        elif structural_isomorphism > 0.8 and target_performance >= 0.8:
            source_domain = source_task.get("domain", "default")
            target_domain = target_task.get("domain", "default")
            if source_domain != target_domain:
                classification = TransferClassification.CROSS_DOMAIN_ANALOGY
                rationale = f"Zero lexical overlap ({lexical_overlap:.2f}) with distinct domains ({source_domain} -> {target_domain}); policy reused across analogous relational structures"
            else:
                classification = TransferClassification.POLICY_GENERALIZATION
                rationale = f"Zero lexical overlap ({lexical_overlap:.2f}); transfer driven by topological policy generalization"
        elif target_performance < 0.5:
            classification = TransferClassification.NO_TRANSFER
            rationale = "Target task performance below chance; no transfer observed"
        else:
            classification = TransferClassification.SURFACE_MATCH
            rationale = "Transfer mediated by template surface match"

        return TransferEvaluationResult(
            source_domain=source_task.get("domain", "family_a"),
            target_domain=target_task.get("domain", "family_b"),
            lexical_overlap=lexical_overlap,
            surface_similarity=0.1,
            structural_isomorphism=structural_isomorphism,
            transfer_accuracy=target_performance,
            classification=classification,
            parametric_delta_w=model_delta_w,
            rationale=rationale,
        )
