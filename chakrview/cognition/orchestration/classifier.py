"""
Deterministic Workload Classifier for Step 28 Adaptive Orchestration.

Maps cognitive tasks to formal WorkloadClass categories based on observable
structural properties, evidence state, and environmental constraints.

CRITICAL INVARIANTS:
1. DETERMINISTIC: Zero random sampling; identical inputs yield identical classifications.
2. EXPLAINABLE: Every classification produces public structural explanation metadata.
3. CPU-SAFE: Pure rule-based evaluation without neural inference or GPU dependencies.
"""

from typing import Dict, Any, Optional, Tuple

from chakrview.cognition.orchestration.models import WorkloadClass
from chakrview.cognition.orchestration.workload import extract_workload_features
from chakrview.cognition.adaptation.profiles import ResourceProfile


class DeterministicWorkloadClassifier:
    """
    Deterministic rule-based classifier assigning cognitive objectives to WorkloadClass.
    """

    @classmethod
    def classify(
        cls,
        objective: str,
        context: Optional[Dict[str, Any]] = None,
        resource_profile: ResourceProfile = ResourceProfile.STANDARD,
        contradiction_count: int = 0,
        has_unverified_claims: bool = False,
        node_health_degraded: bool = False,
    ) -> Tuple[WorkloadClass, Dict[str, Any]]:
        """
        Classify task objective into a WorkloadClass with deterministic justification.
        """
        context = context or {}
        is_resource_constrained = (
            resource_profile == ResourceProfile.LOW_RESOURCE and node_health_degraded
        ) or context.get("force_resource_constrained", False)

        features = extract_workload_features(
            objective=objective,
            context=context,
            contradiction_count=contradiction_count,
            has_unverified_claims=has_unverified_claims,
            is_resource_constrained=is_resource_constrained,
        )

        # 1. Resource constrained environment check
        if is_resource_constrained:
            return WorkloadClass.RESOURCE_CONSTRAINED, {
                "reason": "Host hardware is LOW_RESOURCE with degraded nodes or strict constraint",
                "features": features,
            }

        # 2. Contradiction / Conflict priority
        if features["has_contradiction"]:
            return WorkloadClass.CONFLICTED, {
                "reason": "Detected explicit contradiction markers or opposing evidence in context",
                "features": features,
            }

        # 3. Explicit verification required
        if features["has_verification"]:
            return WorkloadClass.VERIFICATION_REQUIRED, {
                "reason": "Task explicitly mandates safety, audit, compliance, or integrity verification",
                "features": features,
            }

        # 4. Ambiguity / Underspecified priority
        if features["has_ambiguity"]:
            return WorkloadClass.AMBIGUOUS, {
                "reason": "Objective contains epistemic uncertainty markers or is underspecified",
                "features": features,
            }

        # 5. Complex multi-part reasoning or mathematical/algorithmic logic
        if (features["has_math_or_logic"] and features["has_multi_part"]) or (
            features["word_count"] >= 35 and features["has_multi_part"]
        ) or context.get("requires_multi_step_planning", False):
            return WorkloadClass.COMPLEX, {
                "reason": "Multi-part compositional objective requiring structured analytical decomposition",
                "features": features,
            }

        # 6. Simple direct declarative task
        if (
            features["word_count"] <= 8
            and not features["has_math_or_logic"]
            and not features["has_capability"]
            and not features["has_analytical"]
        ):
            return WorkloadClass.SIMPLE, {
                "reason": "Short, single-clause declarative query requiring minimal lookup",
                "features": features,
            }

        # 7. Standard cognitive federation
        return WorkloadClass.STANDARD, {
            "reason": "Standard multi-agent analytical or informational task",
            "features": features,
        }
