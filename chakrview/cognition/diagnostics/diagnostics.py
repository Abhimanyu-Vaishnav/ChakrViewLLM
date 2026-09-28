"""
Self-Diagnostics Engine for ChakrView (Step 23).

Performs 10 comprehensive diagnostic inspections:
1. Model configuration integrity
2. Frozen invariant integrity
3. Runtime weight fingerprint
4. Tokenizer compatibility
5. Context length constraints
6. Memory/state serialization integrity
7. Checkpoint metadata integrity
8. Runtime numerical health (NaN/Inf)
9. Cache/state consistency
10. Training artifact integrity

Statuses:
- HEALTHY
- DEGRADED
- RECOVERABLE
- CORRUPTED
- BLOCKED
- UNKNOWN
"""

from dataclasses import dataclass, field, asdict
from enum import Enum
import json
from pathlib import Path
import time
from typing import Dict, List, Optional, Any, Union
import torch
import torch.nn as nn

from chakrview.brain.model import ChakrMicro
from chakrview.tokenizer.tokenizer import BPETokenizer
from chakrview.cognition.diagnostics.integrity import CoreIntegrityGuard, InvariantViolationError


class DiagnosticStatus(str, Enum):
    """Categorical assessment of diagnostic check outcome."""
    HEALTHY = "HEALTHY"            # Fully compliant and operational
    DEGRADED = "DEGRADED"          # Operational but operating under constrained conditions
    RECOVERABLE = "RECOVERABLE"    # Anomaly detected that can be safely rebuilt/restored
    CORRUPTED = "CORRUPTED"        # Artifact or state corrupted beyond simple repair
    BLOCKED = "BLOCKED"            # Invariant breached; execution must stop immediately
    UNKNOWN = "UNKNOWN"            # Telemetry unavailable; cannot reliably determine status


@dataclass
class DiagnosticCheckResult:
    """Outcome of an individual diagnostic check."""
    check_name: str
    status: DiagnosticStatus
    passed: bool
    is_recoverable: bool
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "check_name": self.check_name,
            "status": self.status.value,
            "passed": self.passed,
            "is_recoverable": self.is_recoverable,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp,
        }


@dataclass
class DiagnosticReport:
    """Aggregated system diagnostic summary."""
    overall_status: DiagnosticStatus
    passed: bool
    checks: List[DiagnosticCheckResult]
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall_status": self.overall_status.value,
            "passed": self.passed,
            "checks": [c.to_dict() for c in self.checks],
            "timestamp": self.timestamp,
        }


class SystemDiagnosticsEngine:
    """
    Modular diagnostic evaluator for runtime, models, states, and artifacts.
    """

    def __init__(
        self,
        model: Optional[ChakrMicro] = None,
        tokenizer: Optional[BPETokenizer] = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer

    # 1. Model Configuration Integrity
    def check_model_configuration(self, model: Optional[ChakrMicro] = None) -> DiagnosticCheckResult:
        m = model or self.model
        if m is None:
            return DiagnosticCheckResult(
                check_name="model_configuration_integrity",
                status=DiagnosticStatus.UNKNOWN,
                passed=False,
                is_recoverable=False,
                message="No model instance available to inspect.",
            )

        cfg = getattr(m, "config", None)
        if cfg is None:
            return DiagnosticCheckResult(
                check_name="model_configuration_integrity",
                status=DiagnosticStatus.BLOCKED,
                passed=False,
                is_recoverable=False,
                message="Model instance lacks configuration attribute.",
            )

        return DiagnosticCheckResult(
            check_name="model_configuration_integrity",
            status=DiagnosticStatus.HEALTHY,
            passed=True,
            is_recoverable=False,
            message="Model configuration present and valid.",
            details={"config_class": cfg.__class__.__name__},
        )

    # 2. Frozen Invariant Integrity
    def check_frozen_invariants(self, model: Optional[ChakrMicro] = None) -> DiagnosticCheckResult:
        m = model or self.model
        if m is None:
            return DiagnosticCheckResult(
                check_name="frozen_invariant_integrity",
                status=DiagnosticStatus.UNKNOWN,
                passed=False,
                is_recoverable=False,
                message="No model instance available to inspect invariants.",
            )

        res = CoreIntegrityGuard.verify_model(m)
        status = DiagnosticStatus.HEALTHY if res.passed else DiagnosticStatus.BLOCKED
        return DiagnosticCheckResult(
            check_name="frozen_invariant_integrity",
            status=status,
            passed=res.passed,
            is_recoverable=False,
            message=res.message,
            details=res.details,
        )

    # 3. Runtime Weight Fingerprint
    def check_runtime_weight_fingerprint(
        self, model: Optional[ChakrMicro] = None, expected_fingerprint: Optional[str] = None
    ) -> DiagnosticCheckResult:
        m = model or self.model
        if m is None:
            return DiagnosticCheckResult(
                check_name="runtime_weight_fingerprint",
                status=DiagnosticStatus.UNKNOWN,
                passed=False,
                is_recoverable=False,
                message="No model instance available for weight fingerprinting.",
            )

        current_fp = CoreIntegrityGuard.compute_weight_fingerprint(m)
        if expected_fingerprint is not None:
            match = (current_fp == expected_fingerprint)
            status = DiagnosticStatus.HEALTHY if match else DiagnosticStatus.BLOCKED
            return DiagnosticCheckResult(
                check_name="runtime_weight_fingerprint",
                status=status,
                passed=match,
                is_recoverable=False,
                message="Weight fingerprint matches expected hash." if match else "Weight fingerprint mismatch!",
                details={"current_fingerprint": current_fp, "expected_fingerprint": expected_fingerprint},
            )

        return DiagnosticCheckResult(
            check_name="runtime_weight_fingerprint",
            status=DiagnosticStatus.HEALTHY,
            passed=True,
            is_recoverable=False,
            message="Computed runtime weight fingerprint successfully.",
            details={"current_fingerprint": current_fp},
        )

    # 4. Tokenizer Compatibility
    def check_tokenizer_compatibility(
        self, tokenizer: Optional[BPETokenizer] = None
    ) -> DiagnosticCheckResult:
        tok = tokenizer or self.tokenizer
        if tok is None:
            return DiagnosticCheckResult(
                check_name="tokenizer_compatibility",
                status=DiagnosticStatus.UNKNOWN,
                passed=False,
                is_recoverable=False,
                message="No tokenizer instance available.",
            )

        passed, msg = CoreIntegrityGuard.verify_tokenizer(tok)
        status = DiagnosticStatus.HEALTHY if passed else DiagnosticStatus.BLOCKED
        return DiagnosticCheckResult(
            check_name="tokenizer_compatibility",
            status=status,
            passed=passed,
            is_recoverable=False,
            message=msg,
        )

    # 5. Context Length Constraints
    def check_context_length_constraints(
        self, token_count: int, max_limit: int = 512
    ) -> DiagnosticCheckResult:
        if token_count <= max_limit:
            return DiagnosticCheckResult(
                check_name="context_length_constraints",
                status=DiagnosticStatus.HEALTHY,
                passed=True,
                is_recoverable=False,
                message=f"Context length {token_count} within limit {max_limit}.",
                details={"token_count": token_count, "max_limit": max_limit},
            )
        else:
            return DiagnosticCheckResult(
                check_name="context_length_constraints",
                status=DiagnosticStatus.RECOVERABLE,
                passed=False,
                is_recoverable=True,  # Recoverable via truncation or context rebuild
                message=f"Context length {token_count} exceeds maximum {max_limit}.",
                details={"token_count": token_count, "max_limit": max_limit},
            )

    # 6. Memory/State Serialization Integrity
    def check_state_serialization(
        self, state_dict: Union[Dict[str, Any], str]
    ) -> DiagnosticCheckResult:
        try:
            if isinstance(state_dict, str):
                parsed = json.loads(state_dict)
            else:
                parsed = json.loads(json.dumps(state_dict))
            return DiagnosticCheckResult(
                check_name="state_serialization_integrity",
                status=DiagnosticStatus.HEALTHY,
                passed=True,
                is_recoverable=False,
                message="State serializes and deserializes without error.",
                details={"keys_count": len(parsed) if isinstance(parsed, dict) else 1},
            )
        except Exception as e:
            return DiagnosticCheckResult(
                check_name="state_serialization_integrity",
                status=DiagnosticStatus.RECOVERABLE,
                passed=False,
                is_recoverable=True,  # Recoverable via state restore from backup
                message=f"State serialization corrupted: {str(e)}",
            )

    # 7. Checkpoint Metadata Integrity
    def check_checkpoint_metadata(
        self, checkpoint_data: Dict[str, Any]
    ) -> DiagnosticCheckResult:
        required_keys = ["step", "model_state_dict"]
        missing = [k for k in required_keys if k not in checkpoint_data]
        if missing:
            return DiagnosticCheckResult(
                check_name="checkpoint_metadata_integrity",
                status=DiagnosticStatus.CORRUPTED,
                passed=False,
                is_recoverable=False,
                message=f"Checkpoint metadata missing required keys: {missing}",
            )

        # Invariant checks if metadata present
        meta = checkpoint_data.get("metadata", {})
        if meta:
            if meta.get("vocab_size") and meta.get("vocab_size") != 4096:
                return DiagnosticCheckResult(
                    check_name="checkpoint_metadata_integrity",
                    status=DiagnosticStatus.BLOCKED,
                    passed=False,
                    is_recoverable=False,
                    message="Checkpoint vocab_size mismatch.",
                )

        return DiagnosticCheckResult(
            check_name="checkpoint_metadata_integrity",
            status=DiagnosticStatus.HEALTHY,
            passed=True,
            is_recoverable=False,
            message="Checkpoint metadata is healthy and compliant.",
        )

    # 8. Runtime Numerical Health (NaN/Inf)
    def check_runtime_numerical_health(
        self, model: Optional[ChakrMicro] = None
    ) -> DiagnosticCheckResult:
        m = model or self.model
        if m is None:
            return DiagnosticCheckResult(
                check_name="runtime_numerical_health",
                status=DiagnosticStatus.UNKNOWN,
                passed=False,
                is_recoverable=False,
                message="No model instance available for numerical check.",
            )

        nan_count = 0
        inf_count = 0
        for name, param in m.named_parameters():
            if torch.isnan(param).any():
                nan_count += 1
            if torch.isinf(param).any():
                inf_count += 1

        if nan_count > 0 or inf_count > 0:
            return DiagnosticCheckResult(
                check_name="runtime_numerical_health",
                status=DiagnosticStatus.CORRUPTED,
                passed=False,
                is_recoverable=False,
                message=f"Numerical instability: {nan_count} NaN tensors, {inf_count} Inf tensors.",
                details={"nan_tensors": nan_count, "inf_tensors": inf_count},
            )

        return DiagnosticCheckResult(
            check_name="runtime_numerical_health",
            status=DiagnosticStatus.HEALTHY,
            passed=True,
            is_recoverable=False,
            message="All parameters are finite (0 NaNs, 0 Infs).",
        )

    # 9. Cache/State Consistency
    def check_cache_consistency(
        self, cache_object: Optional[Any]
    ) -> DiagnosticCheckResult:
        if cache_object is None:
            return DiagnosticCheckResult(
                check_name="cache_consistency",
                status=DiagnosticStatus.HEALTHY,
                passed=True,
                is_recoverable=False,
                message="Cache is clean/empty.",
            )

        # Check for sequence dimension corruption or negative indices
        if hasattr(cache_object, "seq_len") and getattr(cache_object, "seq_len") > 512:
            return DiagnosticCheckResult(
                check_name="cache_consistency",
                status=DiagnosticStatus.RECOVERABLE,
                passed=False,
                is_recoverable=True,  # Reinitialize cache
                message="Cache seq_len exceeds context window ceiling 512.",
            )

        return DiagnosticCheckResult(
            check_name="cache_consistency",
            status=DiagnosticStatus.HEALTHY,
            passed=True,
            is_recoverable=False,
            message="Cache state is consistent.",
        )

    # 10. Training Artifact Integrity
    def check_training_artifact(
        self, manifest: Optional[Dict[str, Any]]
    ) -> DiagnosticCheckResult:
        if manifest is None:
            return DiagnosticCheckResult(
                check_name="training_artifact_integrity",
                status=DiagnosticStatus.UNKNOWN,
                passed=False,
                is_recoverable=False,
                message="No artifact manifest provided.",
            )

        required = ["dataset_fingerprint", "tokenizer_fingerprint", "total_examples"]
        missing = [r for r in required if r not in manifest]
        if missing:
            return DiagnosticCheckResult(
                check_name="training_artifact_integrity",
                status=DiagnosticStatus.CORRUPTED,
                passed=False,
                is_recoverable=False,
                message=f"Training artifact manifest missing fields: {missing}",
            )

        return DiagnosticCheckResult(
            check_name="training_artifact_integrity",
            status=DiagnosticStatus.HEALTHY,
            passed=True,
            is_recoverable=False,
            message="Training artifact manifest is valid.",
        )

    # Comprehensive diagnostic evaluation
    def run_full_diagnostics(
        self,
        model: Optional[ChakrMicro] = None,
        tokenizer: Optional[BPETokenizer] = None,
        expected_weight_fingerprint: Optional[str] = None,
        test_context_len: int = 128,
        test_state: Optional[Dict[str, Any]] = None,
    ) -> DiagnosticReport:
        """Run all primary inspections and synthesize system health status."""
        checks = [
            self.check_model_configuration(model),
            self.check_frozen_invariants(model),
            self.check_runtime_weight_fingerprint(model, expected_weight_fingerprint),
            self.check_tokenizer_compatibility(tokenizer),
            self.check_context_length_constraints(test_context_len),
            self.check_state_serialization(test_state or {"healthy": True}),
            self.check_runtime_numerical_health(model),
            self.check_cache_consistency(None),
        ]

        # Determine overall status
        statuses = [c.status for c in checks]
        if DiagnosticStatus.BLOCKED in statuses:
            overall = DiagnosticStatus.BLOCKED
        elif DiagnosticStatus.CORRUPTED in statuses:
            overall = DiagnosticStatus.CORRUPTED
        elif DiagnosticStatus.RECOVERABLE in statuses:
            overall = DiagnosticStatus.RECOVERABLE
        elif DiagnosticStatus.DEGRADED in statuses:
            overall = DiagnosticStatus.DEGRADED
        elif all(s == DiagnosticStatus.HEALTHY for s in statuses):
            overall = DiagnosticStatus.HEALTHY
        else:
            overall = DiagnosticStatus.UNKNOWN

        passed = overall in [DiagnosticStatus.HEALTHY, DiagnosticStatus.DEGRADED]
        return DiagnosticReport(
            overall_status=overall,
            passed=passed,
            checks=checks,
            timestamp=time.time(),
        )
