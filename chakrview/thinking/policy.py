"""
Deliberation Policies for ChakrView (Step 21).

Defines bounded execution controls and safety thresholds for the thinking workspace:
- Hard ceilings on thought steps, revision cycles, hypotheses, and evidence items.
- Strict timeout boundaries preventing runaway loops.
- Configurable critique score thresholds and verification prerequisites.
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any


@dataclass
class ThinkingPolicy:
    """
    Configurable operational limits and governance rules for deliberation.
    """
    max_thought_steps: int = 10
    max_hypotheses: int = 5
    max_revision_cycles: int = 3
    max_evidence_items: int = 12
    max_workspace_tokens: int = 512
    max_deliberation_time_ms: float = 2500.0
    require_verification_to_stop: bool = True
    min_critique_score: float = 0.75
    allow_revisions: bool = True

    def __post_init__(self) -> None:
        if self.max_thought_steps <= 0:
            raise ValueError(f"max_thought_steps must be positive, got {self.max_thought_steps}")
        if self.max_revision_cycles < 0:
            raise ValueError(f"max_revision_cycles cannot be negative, got {self.max_revision_cycles}")
        if not (0.0 <= self.min_critique_score <= 1.0):
            raise ValueError(f"min_critique_score must be in [0.0, 1.0], got {self.min_critique_score}")
        if not (1 <= self.max_workspace_tokens <= 512):
            raise ValueError(f"max_workspace_tokens must be in [1, 512], got {self.max_workspace_tokens}")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ThinkingPolicy":
        return cls(**data)


def get_standard_policy() -> ThinkingPolicy:
    """Standard balanced deliberation policy."""
    return ThinkingPolicy(
        max_thought_steps=10,
        max_hypotheses=5,
        max_revision_cycles=3,
        max_evidence_items=12,
        max_workspace_tokens=512,
        max_deliberation_time_ms=2500.0,
        require_verification_to_stop=True,
        min_critique_score=0.75,
        allow_revisions=True,
    )


def get_strict_policy() -> ThinkingPolicy:
    """Strict policy for high-assurance scenarios."""
    return ThinkingPolicy(
        max_thought_steps=16,
        max_hypotheses=8,
        max_revision_cycles=4,
        max_evidence_items=16,
        max_workspace_tokens=512,
        max_deliberation_time_ms=5000.0,
        require_verification_to_stop=True,
        min_critique_score=0.85,
        allow_revisions=True,
    )


def get_fast_policy() -> ThinkingPolicy:
    """Low-latency policy bounded for minimal deliberation overhead."""
    return ThinkingPolicy(
        max_thought_steps=4,
        max_hypotheses=2,
        max_revision_cycles=1,
        max_evidence_items=6,
        max_workspace_tokens=512,
        max_deliberation_time_ms=800.0,
        require_verification_to_stop=False,
        min_critique_score=0.6,
        allow_revisions=True,
    )
