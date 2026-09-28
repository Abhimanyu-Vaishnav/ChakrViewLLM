"""
ChakrView Self-Diagnostics & Safe Self-Healing Subsystem (Step 23).

Public exports for:
- CoreIntegrityGuard
- SystemDiagnosticsEngine
- SafeSelfHealingManager
- DiagnosticStatus, DiagnosticCheckResult, DiagnosticReport
- InvariantViolationError, WeightMutationError, UnrecoverableFaultError
"""

from chakrview.cognition.diagnostics.integrity import (
    CoreIntegrityGuard,
    IntegrityCheckResult,
    InvariantViolationError,
    WeightMutationError,
    EXPECTED_PARAMETERS,
    EXPECTED_VOCAB_SIZE,
    EXPECTED_MAX_SEQ_LEN,
    EXPECTED_BOS_ID,
    EXPECTED_EOS_ID,
    EXPECTED_PAD_ID,
)
from chakrview.cognition.diagnostics.diagnostics import (
    DiagnosticStatus,
    DiagnosticCheckResult,
    DiagnosticReport,
    SystemDiagnosticsEngine,
)
from chakrview.cognition.diagnostics.healing import (
    SafeSelfHealingManager,
    HealingEventRecord,
    UnrecoverableFaultError,
)

__all__ = [
    "CoreIntegrityGuard",
    "IntegrityCheckResult",
    "InvariantViolationError",
    "WeightMutationError",
    "EXPECTED_PARAMETERS",
    "EXPECTED_VOCAB_SIZE",
    "EXPECTED_MAX_SEQ_LEN",
    "EXPECTED_BOS_ID",
    "EXPECTED_EOS_ID",
    "EXPECTED_PAD_ID",
    "DiagnosticStatus",
    "DiagnosticCheckResult",
    "DiagnosticReport",
    "SystemDiagnosticsEngine",
    "SafeSelfHealingManager",
    "HealingEventRecord",
    "UnrecoverableFaultError",
]
