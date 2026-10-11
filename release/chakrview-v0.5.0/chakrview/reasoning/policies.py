"""
Configurable Reasoning Policies for ChakrView (Step 19).

Enforces strict bounds on recursion depth, iteration cycles, subproblem expansion,
and evidence accumulation to guarantee termination and resource safety.
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any


@dataclass
class ReasoningPolicy:
    """
    Operational parameters and safety boundaries governing a reasoning session.
    """
    max_depth: int = 4
    max_subproblems: int = 10
    max_iterations: int = 5
    max_revisions: int = 3
    max_evidence_items: int = 50
    max_actions: int = 5
    evidence_threshold: float = 0.5
    confidence_threshold: float = 0.6
    require_verification: bool = True
    allow_revisions: bool = True
    allow_capabilities: bool = True
    handle_contradictions_conservatively: bool = True
    timeout_seconds: float = 30.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReasoningPolicy":
        return cls(**data)


def get_standard_policy() -> ReasoningPolicy:
    """Default balanced reasoning policy for standard workloads."""
    return ReasoningPolicy()


def get_strict_policy() -> ReasoningPolicy:
    """High-assurance policy requiring high confidence and strict verification."""
    return ReasoningPolicy(
        max_depth=3,
        max_subproblems=6,
        max_iterations=4,
        max_revisions=2,
        max_evidence_items=40,
        max_actions=3,
        evidence_threshold=0.7,
        confidence_threshold=0.8,
        require_verification=True,
        allow_revisions=True,
        allow_capabilities=True,
        handle_contradictions_conservatively=True,
        timeout_seconds=20.0,
    )


def get_fast_policy() -> ReasoningPolicy:
    """Low-latency policy with minimal revisions for real-time edge responses."""
    return ReasoningPolicy(
        max_depth=2,
        max_subproblems=4,
        max_iterations=2,
        max_revisions=1,
        max_evidence_items=20,
        max_actions=2,
        evidence_threshold=0.4,
        confidence_threshold=0.5,
        require_verification=False,
        allow_revisions=False,
        allow_capabilities=True,
        handle_contradictions_conservatively=False,
        timeout_seconds=5.0,
    )
