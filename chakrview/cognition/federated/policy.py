"""
Federated Execution Policy & Hardware Adaptation (Step 26).

Maps hardware resource profiles to bounded multi-agent cooperative budgets
while strictly preserving the single frozen ChakrMicro v0.1 core.

CRITICAL INVARIANTS:
1. The exact same ChakrMicro core runs on all hardware profiles.
2. Hardware profiles scale cooperative budgets ONLY (agents, rounds, messages, delegation depth).
3. Hard architectural ceilings prevent unbounded agent creation or recursion on any hardware:
   - MAX_AGENTS <= 8
   - MAX_ROUNDS <= 8
   - MAX_MESSAGES <= 128
   - MAX_DELEGATION_DEPTH <= 4
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional

from chakrview.cognition.adaptation.profiles import ResourceProfile


# Hard architectural ceilings
HARD_CEILING_FEDERATED_AGENTS = 8
HARD_CEILING_FEDERATED_ROUNDS = 8
HARD_CEILING_FEDERATED_MESSAGES = 128
HARD_CEILING_DELEGATION_DEPTH = 4


@dataclass(frozen=True)
class FederatedExecutionPolicy:
    """
    Deterministic execution budget mapped to hardware resource capacity.
    """
    profile: ResourceProfile
    max_agents: int
    max_rounds: int
    max_messages_per_round: int
    max_total_messages: int
    max_delegation_depth: int
    max_candidate_outputs: int
    max_synthesis_attempts: int
    timeout_ms: float

    def __post_init__(self) -> None:
        # Enforce hard ceiling guarantees
        if self.max_agents > HARD_CEILING_FEDERATED_AGENTS:
            object.__setattr__(self, "max_agents", HARD_CEILING_FEDERATED_AGENTS)
        if self.max_rounds > HARD_CEILING_FEDERATED_ROUNDS:
            object.__setattr__(self, "max_rounds", HARD_CEILING_FEDERATED_ROUNDS)
        if self.max_total_messages > HARD_CEILING_FEDERATED_MESSAGES:
            object.__setattr__(self, "max_total_messages", HARD_CEILING_FEDERATED_MESSAGES)
        if self.max_delegation_depth > HARD_CEILING_DELEGATION_DEPTH:
            object.__setattr__(self, "max_delegation_depth", HARD_CEILING_DELEGATION_DEPTH)

    @classmethod
    def from_resource_profile(cls, profile: ResourceProfile) -> "FederatedExecutionPolicy":
        """Generate deterministic bounded federated policy from resource profile."""
        if profile == ResourceProfile.LOW_RESOURCE:
            return cls(
                profile=profile,
                max_agents=3,
                max_rounds=2,
                max_messages_per_round=6,
                max_total_messages=24,
                max_delegation_depth=2,
                max_candidate_outputs=2,
                max_synthesis_attempts=1,
                timeout_ms=5000.0,
            )
        elif profile == ResourceProfile.HIGH_RESOURCE:
            return cls(
                profile=profile,
                max_agents=7,
                max_rounds=6,
                max_messages_per_round=20,
                max_total_messages=120,
                max_delegation_depth=4,
                max_candidate_outputs=5,
                max_synthesis_attempts=3,
                timeout_ms=18000.0,
            )
        else:  # STANDARD
            return cls(
                profile=ResourceProfile.STANDARD,
                max_agents=5,
                max_rounds=4,
                max_messages_per_round=12,
                max_total_messages=64,
                max_delegation_depth=3,
                max_candidate_outputs=3,
                max_synthesis_attempts=2,
                timeout_ms=10000.0,
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "profile": self.profile.value,
            "max_agents": self.max_agents,
            "max_rounds": self.max_rounds,
            "max_messages_per_round": self.max_messages_per_round,
            "max_total_messages": self.max_total_messages,
            "max_delegation_depth": self.max_delegation_depth,
            "max_candidate_outputs": self.max_candidate_outputs,
            "max_synthesis_attempts": self.max_synthesis_attempts,
            "timeout_ms": self.timeout_ms,
        }
