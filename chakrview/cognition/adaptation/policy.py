"""
Adaptive Execution Policy for ChakrView Adaptation (Step 23).

Converts hardware profile classification into concrete, bounded execution budgets:
- max_thinking_steps
- max_revision_cycles
- max_evidence_items
- max_hypotheses
- max_generation_tokens
- batch_size
- gradient_accumulation
- worker_count
- memory_budget_mb
- verification_depth

CRITICAL BOUNDARY RULE:
Hardware detection must NEVER be allowed to create unbounded computation:
    detected_resources -> bounded policy -> execution budget
All parameters enforce hard architectural ceilings.
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional

from chakrview.cognition.adaptation.profiles import ResourceProfile
from chakrview.cognition.adaptation.hardware import HardwareProfileSnapshot, HardwareProfiler
from chakrview.cognition.adaptation.profiles import ResourceClassifier


# Hard architectural ceilings preventing unbounded execution on any hardware
HARD_CEILING_THINKING_STEPS = 32
HARD_CEILING_REVISION_CYCLES = 6
HARD_CEILING_EVIDENCE_ITEMS = 50
HARD_CEILING_HYPOTHESES = 12
HARD_CEILING_GENERATION_TOKENS = 512  # Bound by ChakrMicro max_seq_len (512)
HARD_CEILING_BATCH_SIZE = 8
HARD_CEILING_WORKERS = 16
HARD_CEILING_MEMORY_MB = 4096


@dataclass(frozen=True)
class AdaptiveExecutionPolicy:
    """
    Strict, deterministic execution budget tailored to hardware capacity.
    """
    profile: ResourceProfile
    max_thinking_steps: int
    max_revision_cycles: int
    max_evidence_items: int
    max_hypotheses: int
    max_generation_tokens: int
    batch_size: int
    gradient_accumulation: int
    worker_count: int
    memory_budget_mb: int
    verification_depth: str

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["profile"] = self.profile.value
        return data

    @classmethod
    def create_bounded(
        cls,
        profile: ResourceProfile,
        max_thinking_steps: Optional[int] = None,
        max_revision_cycles: Optional[int] = None,
        max_evidence_items: Optional[int] = None,
        max_hypotheses: Optional[int] = None,
        max_generation_tokens: Optional[int] = None,
        batch_size: Optional[int] = None,
        gradient_accumulation: Optional[int] = None,
        worker_count: Optional[int] = None,
        memory_budget_mb: Optional[int] = None,
        verification_depth: Optional[str] = None,
    ) -> "AdaptiveExecutionPolicy":
        """
        Factory method applying profile defaults while clamping all values to hard ceilings.
        """
        defaults = cls.get_profile_defaults(profile)

        return cls(
            profile=profile,
            max_thinking_steps=min(
                max(1, max_thinking_steps if max_thinking_steps is not None else defaults.max_thinking_steps),
                HARD_CEILING_THINKING_STEPS,
            ),
            max_revision_cycles=min(
                max(0, max_revision_cycles if max_revision_cycles is not None else defaults.max_revision_cycles),
                HARD_CEILING_REVISION_CYCLES,
            ),
            max_evidence_items=min(
                max(1, max_evidence_items if max_evidence_items is not None else defaults.max_evidence_items),
                HARD_CEILING_EVIDENCE_ITEMS,
            ),
            max_hypotheses=min(
                max(1, max_hypotheses if max_hypotheses is not None else defaults.max_hypotheses),
                HARD_CEILING_HYPOTHESES,
            ),
            max_generation_tokens=min(
                max(8, max_generation_tokens if max_generation_tokens is not None else defaults.max_generation_tokens),
                HARD_CEILING_GENERATION_TOKENS,
            ),
            batch_size=min(
                max(1, batch_size if batch_size is not None else defaults.batch_size),
                HARD_CEILING_BATCH_SIZE,
            ),
            gradient_accumulation=max(
                1, gradient_accumulation if gradient_accumulation is not None else defaults.gradient_accumulation
            ),
            worker_count=min(
                max(1, worker_count if worker_count is not None else defaults.worker_count),
                HARD_CEILING_WORKERS,
            ),
            memory_budget_mb=min(
                max(128, memory_budget_mb if memory_budget_mb is not None else defaults.memory_budget_mb),
                HARD_CEILING_MEMORY_MB,
            ),
            verification_depth=verification_depth or defaults.verification_depth,
        )

    @staticmethod
    def get_profile_defaults(profile: ResourceProfile) -> "AdaptiveExecutionPolicy":
        """Predefined safe profiles."""
        if profile == ResourceProfile.LOW_RESOURCE:
            return AdaptiveExecutionPolicy(
                profile=ResourceProfile.LOW_RESOURCE,
                max_thinking_steps=4,
                max_revision_cycles=1,
                max_evidence_items=5,
                max_hypotheses=2,
                max_generation_tokens=64,
                batch_size=1,
                gradient_accumulation=8,
                worker_count=1,
                memory_budget_mb=256,
                verification_depth="basic",
            )
        elif profile == ResourceProfile.HIGH_RESOURCE:
            return AdaptiveExecutionPolicy(
                profile=ResourceProfile.HIGH_RESOURCE,
                max_thinking_steps=16,
                max_revision_cycles=4,
                max_evidence_items=30,
                max_hypotheses=8,
                max_generation_tokens=256,
                batch_size=4,
                gradient_accumulation=2,
                worker_count=4,
                memory_budget_mb=1024,
                verification_depth="exhaustive",
            )
        else:  # STANDARD
            return AdaptiveExecutionPolicy(
                profile=ResourceProfile.STANDARD,
                max_thinking_steps=8,
                max_revision_cycles=2,
                max_evidence_items=15,
                max_hypotheses=4,
                max_generation_tokens=128,
                batch_size=2,
                gradient_accumulation=4,
                worker_count=2,
                memory_budget_mb=512,
                verification_depth="standard",
            )

    @classmethod
    def from_environment(cls) -> "AdaptiveExecutionPolicy":
        """Automatically detect host resources and generate bounded execution policy."""
        snapshot = HardwareProfiler.profile()
        profile = ResourceClassifier.classify(snapshot)
        return cls.get_profile_defaults(profile)
