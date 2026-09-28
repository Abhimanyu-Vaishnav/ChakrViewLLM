"""
Policy and Architectural Constraints for Step 28 Adaptive Orchestration.

Defines configurable thresholds, budget policies, and hard invariant ceilings.
"""

from dataclasses import dataclass
from typing import Dict, Any

from chakrview.cognition.orchestration.models import (
    WorkloadClass,
    MAX_ORCHESTRATION_AGENTS,
    MAX_ORCHESTRATION_NODES,
    MAX_DELIBERATION_ROUNDS,
    MAX_ORCHESTRATION_RETRIES,
)
from chakrview.cognition.adaptation.profiles import ResourceProfile


@dataclass
class AdaptiveOrchestrationPolicy:
    """
    Operational configuration for the AdaptiveCognitiveOrchestrator.
    All properties are strictly validated against hard ceilings.
    """
    # Ceilings
    max_agents: int = MAX_ORCHESTRATION_AGENTS
    max_nodes: int = MAX_ORCHESTRATION_NODES
    max_deliberation_rounds: int = MAX_DELIBERATION_ROUNDS
    max_retries: int = MAX_ORCHESTRATION_RETRIES

    # Deliberation & Verification triggers
    auto_escalate_conflicts: bool = True
    mandatory_verification_for_high_risk: bool = True
    early_termination_on_sufficiency: bool = True

    # Budgets
    default_timeout_ms: int = 5000
    low_resource_timeout_ms: int = 2500
    high_resource_timeout_ms: int = 8000

    def __post_init__(self) -> None:
        if self.max_agents > MAX_ORCHESTRATION_AGENTS:
            raise ValueError(f"max_agents exceeds hard ceiling of {MAX_ORCHESTRATION_AGENTS}")
        if self.max_nodes > MAX_ORCHESTRATION_NODES:
            raise ValueError(f"max_nodes exceeds hard ceiling of {MAX_ORCHESTRATION_NODES}")
        if self.max_deliberation_rounds > MAX_DELIBERATION_ROUNDS:
            raise ValueError(f"max_deliberation_rounds exceeds hard ceiling of {MAX_DELIBERATION_ROUNDS}")
        if self.max_retries > MAX_ORCHESTRATION_RETRIES:
            raise ValueError(f"max_retries exceeds hard ceiling of {MAX_ORCHESTRATION_RETRIES}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_agents": self.max_agents,
            "max_nodes": self.max_nodes,
            "max_deliberation_rounds": self.max_deliberation_rounds,
            "max_retries": self.max_retries,
            "auto_escalate_conflicts": self.auto_escalate_conflicts,
            "mandatory_verification_for_high_risk": self.mandatory_verification_for_high_risk,
            "early_termination_on_sufficiency": self.early_termination_on_sufficiency,
            "default_timeout_ms": self.default_timeout_ms,
        }
