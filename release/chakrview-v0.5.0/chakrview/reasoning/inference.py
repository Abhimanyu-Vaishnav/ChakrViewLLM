"""
Structured Inference Engine for ChakrView Reasoning (Step 19).

Provides deterministic inference construction across deduction, induction,
comparison, constraint reasoning, temporal reasoning, and causal hypotheses.
"""

from dataclasses import dataclass, field
from enum import Enum
import time
from typing import Dict, List, Optional, Any
import uuid

from chakrview.reasoning.evidence import EvidenceItem, EvidenceStore


class InferenceType(str, Enum):
    """Formal taxonomy of inference mechanisms."""
    DEDUCTION = "DEDUCTION"                      # Strict logical entailment from valid premises
    INDUCTION = "INDUCTION"                      # Pattern generalization from repeated observations
    ANALOGY = "ANALOGY"                          # Structural mapping from a known domain
    COMPARISON = "COMPARISON"                    # Multi-attribute similarity / contrast evaluation
    CONSTRAINT_REASONING = "CONSTRAINT_REASONING"# Satisfaction verification against active bounds
    TEMPORAL_REASONING = "TEMPORAL_REASONING"    # Chronological ordering and temporal interval logic
    CAUSAL_HYPOTHESIS = "CAUSAL_HYPOTHESIS"      # Plausible causal linkage between events


@dataclass
class Inference:
    """
    Formally documented inference linking premises, reasoning rule, and conclusion.
    """
    inference_id: str = field(default_factory=lambda: f"inf_{uuid.uuid4().hex[:8]}")
    inference_type: InferenceType = InferenceType.DEDUCTION
    premises: List[str] = field(default_factory=list)  # IDs of evidence or prior inferences
    rule_or_method: str = ""
    conclusion: str = ""
    confidence: float = 0.8
    provenance: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "inference_id": self.inference_id,
            "inference_type": self.inference_type.value,
            "premises": list(self.premises),
            "rule_or_method": self.rule_or_method,
            "conclusion": self.conclusion,
            "confidence": self.confidence,
            "provenance": dict(self.provenance),
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Inference":
        d = dict(data)
        d["inference_type"] = InferenceType(d["inference_type"])
        return cls(**d)


class InferenceEngine:
    """
    Constructs and records structured inferences from premises with bounded propagation.
    """

    def deduce(
        self,
        premises: List[EvidenceItem],
        rule_or_method: str,
        conclusion: str,
    ) -> Inference:
        """
        Deductive inference: strictly derived from premises.
        Composite confidence is bounded by the minimum premise confidence.
        """
        premise_ids = [p.evidence_id for p in premises]
        if not premises:
            conf = 0.5
        else:
            conf = min(p.confidence * p.reliability for p in premises)

        return Inference(
            inference_type=InferenceType.DEDUCTION,
            premises=premise_ids,
            rule_or_method=rule_or_method,
            conclusion=conclusion,
            confidence=conf,
            provenance={"rule": rule_or_method, "premise_count": len(premises)},
        )

    def evaluate_constraints(
        self,
        candidate_name: str,
        parameters: Dict[str, Any],
        constraints: Dict[str, Any],
    ) -> Inference:
        """
        Constraint reasoning: evaluates candidate parameter against active limits.
        """
        violations = []
        for key, max_limit in constraints.items():
            if key in parameters:
                val = parameters[key]
                if isinstance(val, (int, float)) and isinstance(max_limit, (int, float)):
                    if val > max_limit:
                        violations.append(f"{key}={val} exceeds max {max_limit}")

        if violations:
            conclusion = f"Constraint violation for '{candidate_name}': " + "; ".join(violations)
            conf = 1.0
        else:
            conclusion = f"Candidate '{candidate_name}' satisfies all evaluated constraints."
            conf = 0.95

        return Inference(
            inference_type=InferenceType.CONSTRAINT_REASONING,
            premises=[],
            rule_or_method="boundary_check",
            conclusion=conclusion,
            confidence=conf,
            provenance={"violations": violations, "constraints": constraints},
        )

    def compare(
        self,
        item_a: Dict[str, Any],
        item_b: Dict[str, Any],
        attribute: str,
    ) -> Inference:
        """
        Comparative reasoning between two items along a target dimension.
        """
        val_a = item_a.get(attribute)
        val_b = item_b.get(attribute)

        if val_a == val_b:
            conclusion = f"Items are identical with respect to '{attribute}' ({val_a})."
        else:
            conclusion = f"Discrepancy in '{attribute}': item_a={val_a}, item_b={val_b}."

        return Inference(
            inference_type=InferenceType.COMPARISON,
            premises=[str(item_a.get("id", "a")), str(item_b.get("id", "b"))],
            rule_or_method="attribute_comparison",
            conclusion=conclusion,
            confidence=0.9,
            provenance={"attribute": attribute, "val_a": val_a, "val_b": val_b},
        )
