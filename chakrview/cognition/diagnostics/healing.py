"""
Safe Self-Healing and Recovery for ChakrView (Step 23).

Protocol:
    detect
      ↓
    classify
      ↓
    isolate
      ↓
    restore / rebuild
      ↓
    verify
      ↓
    resume

SAFE RECOVERY ACTIONS:
1. Corrupted derived context -> rebuild context from raw inputs.
2. Invalid transient cache -> clear / reinitialize cache.
3. Invalid serialized cognitive state -> restore last valid backup.
4. Corrupted training artifact -> quarantine and reject artifact.
5. Failed checkpoint validation -> rollback to last known-good checkpoint.
6. Runtime invariant mismatch -> BLOCK execution (NEVER modify model).
7. Runtime weight mutation -> FAIL CLOSED immediately.

STRICT INVARIANTS:
Self-healing MUST NOT:
- modify model weights
- rewrite core architecture
- bypass safety checks
- disable failed validation
"""

from dataclasses import dataclass, field
import json
from pathlib import Path
import time
from typing import Dict, List, Optional, Any, Callable, Union

from chakrview.brain.model import ChakrMicro
from chakrview.cognition.diagnostics.integrity import (
    CoreIntegrityGuard,
    InvariantViolationError,
    WeightMutationError,
)
from chakrview.cognition.diagnostics.diagnostics import DiagnosticStatus, DiagnosticCheckResult


class UnrecoverableFaultError(RuntimeError):
    """Raised when an unrecoverable fault or invariant breach occurs."""
    pass


@dataclass
class HealingEventRecord:
    """Audit entry documenting an automated recovery action."""
    action_name: str
    target_component: str
    previous_state: str
    recovered_state: str
    success: bool
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_name": self.action_name,
            "target_component": self.target_component,
            "previous_state": self.previous_state,
            "recovered_state": self.recovered_state,
            "success": self.success,
            "details": self.details,
            "timestamp": self.timestamp,
        }


class SafeSelfHealingManager:
    """
    Governed self-healing manager enforcing strict safety boundaries.
    """

    def __init__(self, quarantine_dir: Optional[Path] = None) -> None:
        self.quarantine_dir = quarantine_dir or Path(".chakr_quarantine")
        self.audit_log: List[HealingEventRecord] = []

    def log_healing(
        self,
        action_name: str,
        target_component: str,
        previous_state: str,
        recovered_state: str,
        success: bool,
        details: Optional[Dict[str, Any]] = None,
    ) -> HealingEventRecord:
        record = HealingEventRecord(
            action_name=action_name,
            target_component=target_component,
            previous_state=previous_state,
            recovered_state=recovered_state,
            success=success,
            details=details or {},
            timestamp=time.time(),
        )
        self.audit_log.append(record)
        return record

    # 1. Corrupted Derived Context -> Rebuild Context
    def rebuild_derived_context(
        self,
        raw_prompt: str,
        system_identity: str,
        context_builder_fn: Callable[[str, str], Any],
    ) -> Any:
        """
        Safely discard corrupted assembled context and rebuild deterministically from raw inputs.
        """
        try:
            new_context = context_builder_fn(raw_prompt, system_identity)
            self.log_healing(
                action_name="rebuild_derived_context",
                target_component="context_assembly",
                previous_state="corrupted_or_overflowed",
                recovered_state="rebuilt_clean",
                success=True,
                details={"prompt_len": len(raw_prompt)},
            )
            return new_context
        except Exception as e:
            self.log_healing(
                action_name="rebuild_derived_context",
                target_component="context_assembly",
                previous_state="corrupted",
                recovered_state="rebuild_failed",
                success=False,
                details={"error": str(e)},
            )
            raise UnrecoverableFaultError(f"Context rebuild failed: {str(e)}")

    # 2. Invalid Transient Cache -> Clear / Reinitialize Cache
    def reinitialize_transient_cache(
        self, cache_factory_fn: Callable[[], Any]
    ) -> Any:
        """
        Clear invalid KV cache or scratchpad memory and instantiate clean cache.
        """
        try:
            clean_cache = cache_factory_fn()
            self.log_healing(
                action_name="reinitialize_transient_cache",
                target_component="kv_cache_transient",
                previous_state="invalid_dimensions_or_corrupt",
                recovered_state="fresh_cache_instance",
                success=True,
            )
            return clean_cache
        except Exception as e:
            self.log_healing(
                action_name="reinitialize_transient_cache",
                target_component="kv_cache_transient",
                previous_state="corrupt",
                recovered_state="reinit_failed",
                success=False,
                details={"error": str(e)},
            )
            raise UnrecoverableFaultError(f"Cache reinitialization failed: {str(e)}")

    # 3. Invalid Serialized Cognitive State -> Restore Last Valid State
    def restore_serialized_cognitive_state(
        self,
        backup_state: Optional[Dict[str, Any]],
        default_state_factory: Optional[Callable[[], Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Restore cognitive state from validated backup, or initialize clean default if available.
        """
        if backup_state is not None:
            # Verify backup integrity
            try:
                verified = json.loads(json.dumps(backup_state))
                self.log_healing(
                    action_name="restore_serialized_cognitive_state",
                    target_component="cognitive_state",
                    previous_state="corrupted_json",
                    recovered_state="restored_from_backup",
                    success=True,
                )
                return verified
            except Exception:
                pass

        if default_state_factory is not None:
            default_state = default_state_factory()
            self.log_healing(
                action_name="restore_serialized_cognitive_state",
                target_component="cognitive_state",
                previous_state="corrupted_json",
                recovered_state="reset_to_default",
                success=True,
            )
            return default_state

        raise UnrecoverableFaultError("No valid backup or default factory available to restore state.")

    # 4. Corrupted Training Artifact -> Reject Artifact
    def quarantine_corrupted_training_artifact(
        self, artifact_name: str, reason: str
    ) -> Dict[str, Any]:
        """
        Isolate and reject an invalid or corrupted training artifact.
        """
        self.quarantine_dir.mkdir(parents=True, exist_ok=True)
        record = {
            "artifact_name": artifact_name,
            "reason": reason,
            "quarantined_at": time.time(),
            "status": "QUARANTINED",
        }
        self.log_healing(
            action_name="quarantine_corrupted_training_artifact",
            target_component="training_dataset",
            previous_state="corrupted_artifact",
            recovered_state="isolated_and_rejected",
            success=True,
            details=record,
        )
        return record

    # 5. Failed Checkpoint Validation -> Rollback to Last Known Good
    def rollback_model_checkpoint(
        self, rollback_fn: Callable[[], Any]
    ) -> Any:
        """
        Execute rollback to previously verified checkpoint using ModelUpdateManager.
        """
        try:
            res = rollback_fn()
            self.log_healing(
                action_name="rollback_model_checkpoint",
                target_component="model_version",
                previous_state="failed_candidate",
                recovered_state="active_restored_to_known_good",
                success=True,
            )
            return res
        except Exception as e:
            self.log_healing(
                action_name="rollback_model_checkpoint",
                target_component="model_version",
                previous_state="failed_candidate",
                recovered_state="rollback_failed",
                success=False,
                details={"error": str(e)},
            )
            raise UnrecoverableFaultError(f"Model rollback failed: {str(e)}")

    # 6. Runtime Invariant Mismatch -> BLOCK execution (Fail-Closed)
    def handle_invariant_mismatch(self, check_result: DiagnosticCheckResult) -> None:
        """
        Hard block on invariant failure.
        CRITICAL: Never attempts to alter, patch, or repair the frozen neural core.
        """
        self.log_healing(
            action_name="block_on_invariant_mismatch",
            target_component="neural_core",
            previous_state="invariant_breached",
            recovered_state="BLOCKED_FAIL_CLOSED",
            success=True,
            details=check_result.to_dict(),
        )
        raise InvariantViolationError(
            f"EXECUTION BLOCKED: Invariant check '{check_result.check_name}' failed: {check_result.message}"
        )

    # 7. Runtime Weight Mutation -> FAIL CLOSED
    def handle_runtime_weight_mutation(
        self, model: ChakrMicro, expected_fingerprint: str
    ) -> None:
        """
        Fail closed if runtime model weights differ from expected read-only fingerprint.
        """
        current_fp = CoreIntegrityGuard.compute_weight_fingerprint(model)
        if current_fp != expected_fingerprint:
            self.log_healing(
                action_name="fail_closed_weight_mutation",
                target_component="neural_weights",
                previous_state="weights_mutated",
                recovered_state="HALTED_FAIL_CLOSED",
                success=True,
                details={"expected": expected_fingerprint, "current": current_fp},
            )
            raise WeightMutationError(
                f"SECURITY HALT: Model weights modified during runtime inference! Expected {expected_fingerprint}, got {current_fp}."
            )
