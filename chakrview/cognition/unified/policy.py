"""
Unified Cognitive Execution Policy & Hardware Adaptation (Step 25).

Translates hardware capacity into bounded cognitive budgets while strictly
preserving the frozen ChakrMicro v0.1 core across all tiers.

CRITICAL INVARIANTS:
1. ChakrMicro v0.1 model parameters (3,443,136), vocabulary (4096),
   context length (512), and special tokens are identical on all hardware.
2. Hardware profiles scale execution budgets ONLY (cycles, steps, candidates, tokens).
3. Hard architectural ceilings prevent unbounded computation on any hardware.
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional

from chakrview.cognition.adaptation.profiles import ResourceProfile
from chakrview.cognition.adaptation.hardware import HardwareProfiler
from chakrview.cognition.adaptation.profiles import ResourceClassifier


# Hard architectural ceilings
HARD_CEILING_CONTEXT_TOKENS = 512      # Hard limit of ChakrMicro v0.1
HARD_CEILING_GENERATION_TOKENS = 512   # Hard limit of ChakrMicro v0.1
HARD_CEILING_THINKING_STEPS = 32
HARD_CEILING_REVISIONS = 6
HARD_CEILING_MEMORY_TOP_K = 25
HARD_CEILING_REASONING_ITERATIONS = 30


@dataclass(frozen=True)
class UnifiedCognitivePolicy:
    """
    Deterministic cognitive budget mapped to hardware resource capacity.
    """
    profile: ResourceProfile
    memory_top_k: int
    reasoning_max_iterations: int
    critical_hypotheses: int
    critical_max_alternatives: int
    thinking_steps: int
    max_revisions: int
    max_generation_tokens: int
    max_context_tokens: int

    def __post_init__(self) -> None:
        # Enforce hard ceiling guarantees
        if self.max_context_tokens > HARD_CEILING_CONTEXT_TOKENS:
            object.__setattr__(self, "max_context_tokens", HARD_CEILING_CONTEXT_TOKENS)
        if self.max_generation_tokens > HARD_CEILING_GENERATION_TOKENS:
            object.__setattr__(self, "max_generation_tokens", HARD_CEILING_GENERATION_TOKENS)
        if self.thinking_steps > HARD_CEILING_THINKING_STEPS:
            object.__setattr__(self, "thinking_steps", HARD_CEILING_THINKING_STEPS)
        if self.max_revisions > HARD_CEILING_REVISIONS:
            object.__setattr__(self, "max_revisions", HARD_CEILING_REVISIONS)
        if self.memory_top_k > HARD_CEILING_MEMORY_TOP_K:
            object.__setattr__(self, "memory_top_k", HARD_CEILING_MEMORY_TOP_K)
        if self.reasoning_max_iterations > HARD_CEILING_REASONING_ITERATIONS:
            object.__setattr__(self, "reasoning_max_iterations", HARD_CEILING_REASONING_ITERATIONS)

    @classmethod
    def low_resource(cls) -> "UnifiedCognitivePolicy":
        """Conservative execution budget for constrained/older hardware."""
        return cls(
            profile=ResourceProfile.LOW_RESOURCE,
            memory_top_k=3,
            reasoning_max_iterations=3,
            critical_hypotheses=2,
            critical_max_alternatives=1,
            thinking_steps=3,
            max_revisions=1,
            max_generation_tokens=64,
            max_context_tokens=384,
        )

    @classmethod
    def standard(cls) -> "UnifiedCognitivePolicy":
        """Balanced default execution budget for standard PCs and laptops."""
        return cls(
            profile=ResourceProfile.STANDARD,
            memory_top_k=6,
            reasoning_max_iterations=8,
            critical_hypotheses=3,
            critical_max_alternatives=2,
            thinking_steps=6,
            max_revisions=2,
            max_generation_tokens=128,
            max_context_tokens=448,
        )

    @classmethod
    def high_resource(cls) -> "UnifiedCognitivePolicy":
        """Expanded bounded budget for high-core/high-memory workstations."""
        return cls(
            profile=ResourceProfile.HIGH_RESOURCE,
            memory_top_k=12,
            reasoning_max_iterations=15,
            critical_hypotheses=5,
            critical_max_alternatives=3,
            thinking_steps=12,
            max_revisions=4,
            max_generation_tokens=256,
            max_context_tokens=480,
        )

    @classmethod
    def from_resource_profile(cls, profile: ResourceProfile) -> "UnifiedCognitivePolicy":
        """Factory from Step 23 ResourceProfile."""
        if profile == ResourceProfile.LOW_RESOURCE:
            return cls.low_resource()
        elif profile == ResourceProfile.HIGH_RESOURCE:
            return cls.high_resource()
        return cls.standard()

    @classmethod
    def detect_host_policy(cls) -> "UnifiedCognitivePolicy":
        """Probe host hardware safely and determine bounded cognitive policy."""
        try:
            snapshot = HardwareProfiler.probe()
            profile = ResourceClassifier.classify(snapshot)
            return cls.from_resource_profile(profile)
        except Exception:
            return cls.standard()

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["profile"] = self.profile.value
        return data
