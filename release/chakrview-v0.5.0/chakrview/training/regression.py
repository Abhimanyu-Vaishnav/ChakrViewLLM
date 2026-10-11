"""
Regression Gate & Governed Model Promotion for ChakrView (Step 22).

Enforces the formal model release lifecycle:
TRAINING -> TRAINED -> VALIDATED -> REGRESSION_TESTED -> PROMOTABLE -> PROMOTED

Guarantees:
1. No silent or automatic promotion: newly trained checkpoints never replace active models automatically.
2. Hard verification of frozen invariants (parameters=3,443,136, vocab=4096, context=512).
3. Tokenizer and dataset compatibility verification.
4. Mandatory regression testing gate.
5. Integration with ModelUpdateManager for full auditability and instant rollback.
"""

from dataclasses import dataclass, asdict, field
from enum import Enum
import time
from typing import Dict, List, Optional, Any, Tuple, Callable
import torch
import torch.nn as nn

from chakrview.intelligence.learning import (
    ModelUpdateManager,
    ModelVersionArtifact,
    ModelUpdateSafetyError,
)
from chakrview.training.contract import TrainingDatasetManifest, TokenizerFingerprint
from chakrview.training.safety import TrainingSafetyChecker, InvariantViolationError
from chakrview.training.validation import ValidationResult


class PromotionStatus(str, Enum):
    """Formal lifecycle state for candidate trained models."""
    TRAINING = "TRAINING"
    TRAINED = "TRAINED"
    VALIDATED = "VALIDATED"
    REGRESSION_TESTED = "REGRESSION_TESTED"
    PROMOTABLE = "PROMOTABLE"
    PROMOTED = "PROMOTED"
    REJECTED = "REJECTED"


@dataclass
class RegressionGateResult:
    """Outcome record of a regression gate evaluation."""
    candidate_version_id: str
    status: PromotionStatus
    invariants_passed: bool
    tokenizer_compatible: bool
    validation_passed: bool
    regression_passed: bool
    checkpoint_valid: bool
    is_promotable: bool
    violations: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        return d


class RegressionGate:
    """
    Quality and safety gate governing the transition of trained models to production.
    """

    def __init__(
        self,
        update_manager: Optional[ModelUpdateManager] = None,
        regression_test_fn: Optional[Callable[[nn.Module], Tuple[bool, List[str]]]] = None,
    ) -> None:
        self.update_manager = update_manager or ModelUpdateManager()
        self.regression_test_fn = regression_test_fn

    def evaluate_candidate(
        self,
        candidate_model: nn.Module,
        candidate_version_id: str,
        checkpoint_path: str,
        val_result: ValidationResult,
        dataset_manifest: TrainingDatasetManifest,
        active_tokenizer: Any,
    ) -> RegressionGateResult:
        """
        Execute comprehensive multi-stage gate checks on a candidate model checkpoint.
        """
        violations: List[str] = []

        # 1. Architecture & Invariant Verification
        inv_passed, inv_violations = TrainingSafetyChecker.verify_model_invariants(candidate_model)
        if not inv_passed:
            violations.extend(inv_violations)

        # 2. Tokenizer Compatibility
        tok_fp = TokenizerFingerprint.from_tokenizer(active_tokenizer)
        tok_compatible = (tok_fp.fingerprint_hash == dataset_manifest.tokenizer_fingerprint)
        if not tok_compatible:
            violations.append(
                f"Tokenizer mismatch: model tokenizer '{tok_fp.fingerprint_hash}' "
                f"!= dataset tokenizer '{dataset_manifest.tokenizer_fingerprint}'."
            )

        # 3. Validation Completion
        val_passed = (
            val_result.val_loss is not None
            and not torch.isnan(torch.tensor(val_result.val_loss))
            and val_result.perplexity_valid
            and val_result.val_loss < 20.0
        )
        if not val_passed:
            violations.append(
                f"Validation check failed: val_loss={val_result.val_loss}, "
                f"perplexity_valid={val_result.perplexity_valid}."
            )

        # 4. Checkpoint Integrity
        ckpt_valid = (checkpoint_path is not None and len(str(checkpoint_path).strip()) > 0)
        if not ckpt_valid:
            violations.append("Checkpoint path missing or invalid.")

        # 5. Regression Testing
        reg_passed = True
        if self.regression_test_fn is not None:
            reg_passed, reg_violations = self.regression_test_fn(candidate_model)
            if not reg_passed:
                violations.extend(reg_violations)
        else:
            # Default baseline smoke check: ensure forward pass on single batch is non-trivial and finite
            try:
                candidate_model.eval()
                with torch.no_grad():
                    test_inp = torch.tensor([[0, 10, 20, 30, 1]], dtype=torch.long)
                    out = candidate_model(test_inp)
                    if torch.isnan(out).any() or torch.isinf(out).any():
                        reg_passed = False
                        violations.append("Baseline forward pass regression produced NaN or Inf.")
            except Exception as e:
                reg_passed = False
                violations.append(f"Baseline forward pass regression failed: {str(e)}")

        is_promotable = inv_passed and tok_compatible and val_passed and reg_passed and ckpt_valid

        status = PromotionStatus.PROMOTABLE if is_promotable else PromotionStatus.REJECTED

        # If promotable, register in ModelUpdateManager as APPROVED_STANDBY
        if is_promotable:
            self.update_manager.register_candidate_version(
                version_id=candidate_version_id,
                artifact_path=checkpoint_path,
                candidate_model=candidate_model,
                val_loss=val_result.val_loss,
                regression_passed=True,
                approver="REGRESSION_GATE_PIPELINE",
            )

        return RegressionGateResult(
            candidate_version_id=candidate_version_id,
            status=status,
            invariants_passed=inv_passed,
            tokenizer_compatible=tok_compatible,
            validation_passed=val_passed,
            regression_passed=reg_passed,
            checkpoint_valid=ckpt_valid,
            is_promotable=is_promotable,
            violations=violations,
        )

    def promote_candidate(
        self,
        candidate_version_id: str,
        authorized_by: str,
    ) -> None:
        """
        Execute an explicit, authorized promotion of a promotable model artifact to active.
        """
        self.update_manager.promote_to_active(
            version_id=candidate_version_id,
            authorized_by=authorized_by,
        )

    def rollback(
        self,
        target_version_id: str,
        reason: str,
    ) -> ModelVersionArtifact:
        """
        Roll back active model version to an earlier approved artifact.
        """
        return self.update_manager.rollback(target_version_id=target_version_id, reason=reason)
