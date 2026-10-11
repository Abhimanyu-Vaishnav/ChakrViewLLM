"""
Memory Execution Policy & Hardware Adaptation for ChakrView (Step 24).

Translates detected hardware resource profiles (LOW_RESOURCE, STANDARD, HIGH_RESOURCE)
into concrete, bounded cognitive memory execution budgets.

CRITICAL ARCHITECTURAL CONSTRAINTS:
1. Hardware profile NEVER modifies the neural core:
   ChakrMicro v0.1 remains 3,443,136 parameters, 4,096 vocab, 512 context, BOS=0, EOS=1, PAD=2
   regardless of whether host PC is weak or powerful.
2. Hardware only determines execution budgets (retrieval slots, scan depth, memory capacity).
3. Hard architectural ceilings prevent unbounded computation on any hardware tier.
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional

from chakrview.cognition.adaptation.profiles import ResourceProfile
from chakrview.cognition.adaptation.hardware import HardwareProfiler
from chakrview.cognition.adaptation.profiles import ResourceClassifier


# Hard architectural ceilings preventing runaway memory computation
HARD_CEILING_WORKING_CAPACITY = 100
HARD_CEILING_RETRIEVAL_CANDIDATES = 30
HARD_CEILING_STORAGE_SCAN = 1000
HARD_CEILING_CONTRADICTION_DEPTH = 50
HARD_CEILING_CONSOLIDATION_BATCH = 20


@dataclass(frozen=True)
class MemoryExecutionPolicy:
    """
    Deterministic memory execution limits mapped to system hardware resources.
    """
    profile: ResourceProfile
    working_memory_capacity: int
    max_retrieval_candidates: int
    storage_scan_limit: int
    contradiction_check_depth: int
    consolidation_batch_size: int

    def __post_init__(self) -> None:
        # Enforce hard ceiling guarantees
        if self.working_memory_capacity > HARD_CEILING_WORKING_CAPACITY:
            object.__setattr__(self, "working_memory_capacity", HARD_CEILING_WORKING_CAPACITY)
        if self.max_retrieval_candidates > HARD_CEILING_RETRIEVAL_CANDIDATES:
            object.__setattr__(self, "max_retrieval_candidates", HARD_CEILING_RETRIEVAL_CANDIDATES)
        if self.storage_scan_limit > HARD_CEILING_STORAGE_SCAN:
            object.__setattr__(self, "storage_scan_limit", HARD_CEILING_STORAGE_SCAN)
        if self.contradiction_check_depth > HARD_CEILING_CONTRADICTION_DEPTH:
            object.__setattr__(self, "contradiction_check_depth", HARD_CEILING_CONTRADICTION_DEPTH)
        if self.consolidation_batch_size > HARD_CEILING_CONSOLIDATION_BATCH:
            object.__setattr__(self, "consolidation_batch_size", HARD_CEILING_CONSOLIDATION_BATCH)

    @classmethod
    def low_resource(cls) -> "MemoryExecutionPolicy":
        """Conservative configuration for low-spec / embedded PCs."""
        return cls(
            profile=ResourceProfile.LOW_RESOURCE,
            working_memory_capacity=10,
            max_retrieval_candidates=3,
            storage_scan_limit=50,
            contradiction_check_depth=5,
            consolidation_batch_size=2,
        )

    @classmethod
    def standard(cls) -> "MemoryExecutionPolicy":
        """Balanced default configuration for standard workstations."""
        return cls(
            profile=ResourceProfile.STANDARD,
            working_memory_capacity=30,
            max_retrieval_candidates=8,
            storage_scan_limit=200,
            contradiction_check_depth=15,
            consolidation_batch_size=5,
        )

    @classmethod
    def high_resource(cls) -> "MemoryExecutionPolicy":
        """Expanded bounded budget for high-core/high-memory machines."""
        return cls(
            profile=ResourceProfile.HIGH_RESOURCE,
            working_memory_capacity=60,
            max_retrieval_candidates=15,
            storage_scan_limit=400,
            contradiction_check_depth=30,
            consolidation_batch_size=10,
        )

    @classmethod
    def from_resource_profile(cls, profile: ResourceProfile) -> "MemoryExecutionPolicy":
        """Factory from Step 23 ResourceProfile enum."""
        if profile == ResourceProfile.LOW_RESOURCE:
            return cls.low_resource()
        elif profile == ResourceProfile.HIGH_RESOURCE:
            return cls.high_resource()
        return cls.standard()

    @classmethod
    def detect_host_policy(cls) -> "MemoryExecutionPolicy":
        """Probe host hardware safely and determine bounded memory policy."""
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
