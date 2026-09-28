"""
ChakrView Hardware Adaptation Subsystem (Step 23).

Public exports for hardware probing, resource profiles, and adaptive execution policy.
"""

from chakrview.cognition.adaptation.hardware import (
    CPUInfo,
    MemoryInfo,
    DeviceInfo,
    HardwareProfileSnapshot,
    HardwareProfiler,
    UNKNOWN,
)
from chakrview.cognition.adaptation.profiles import (
    ResourceProfile,
    ResourceClassifier,
)
from chakrview.cognition.adaptation.policy import (
    AdaptiveExecutionPolicy,
    HARD_CEILING_THINKING_STEPS,
    HARD_CEILING_REVISION_CYCLES,
    HARD_CEILING_EVIDENCE_ITEMS,
    HARD_CEILING_HYPOTHESES,
    HARD_CEILING_GENERATION_TOKENS,
    HARD_CEILING_BATCH_SIZE,
    HARD_CEILING_WORKERS,
    HARD_CEILING_MEMORY_MB,
)

__all__ = [
    "CPUInfo",
    "MemoryInfo",
    "DeviceInfo",
    "HardwareProfileSnapshot",
    "HardwareProfiler",
    "UNKNOWN",
    "ResourceProfile",
    "ResourceClassifier",
    "AdaptiveExecutionPolicy",
    "HARD_CEILING_THINKING_STEPS",
    "HARD_CEILING_REVISION_CYCLES",
    "HARD_CEILING_EVIDENCE_ITEMS",
    "HARD_CEILING_HYPOTHESES",
    "HARD_CEILING_GENERATION_TOKENS",
    "HARD_CEILING_BATCH_SIZE",
    "HARD_CEILING_WORKERS",
    "HARD_CEILING_MEMORY_MB",
]
