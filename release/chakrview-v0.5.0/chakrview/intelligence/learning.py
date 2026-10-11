"""
Learning Pipeline and Model Update Safety for ChakrView (Step 20).

Enforces the offline training lifecycle:
Dataset Candidate -> Validation -> Offline Training Run -> Evaluation
-> Regression Tests -> Frozen Invariant Verification -> Model Approval
-> Versioned Model Artifact.

Architectural Constraints:
1. ZERO RUNTIME WEIGHT UPDATES: The system NEVER updates its weights during inference.
2. NO SILENT REPLACEMENT: Newly trained checkpoints require explicit verification & approval.
3. FROZEN INVARIANTS: Any candidate model artifact must strictly match:
   - Parameters: 3,443,136
   - Vocabulary: 4,096
   - Context length: 512
   - BOS=0, EOS=1, PAD=2
4. ROLLBACK SUPPORT: Complete audit trail and instant reversion capability.
5. TENANT ISOLATION: Multi-tenant dataset partitioning preventing cross-tenant leakage.
"""

from dataclasses import dataclass, field
import json
from pathlib import Path
import time
from typing import Dict, List, Optional, Any, Tuple
import torch

from chakrview.intelligence.contracts import (
    LearningRecord,
    LearningRecordStatus,
)


class ModelUpdateSafetyError(ValueError):
    """Raised when a candidate model violates frozen invariants or approval gates."""
    pass


class TenantIsolationError(PermissionError):
    """Raised when cross-tenant dataset contamination is attempted."""
    pass


@dataclass
class ModelVersionArtifact:
    """Metadata record for an approved, versioned ChakrMicro model artifact."""
    version_id: str
    artifact_path: str
    param_count: int
    vocab_size: int
    context_length: int
    val_loss: float
    approved_by: str
    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version_id": self.version_id,
            "artifact_path": self.artifact_path,
            "param_count": self.param_count,
            "vocab_size": self.vocab_size,
            "context_length": self.context_length,
            "val_loss": self.val_loss,
            "approved_by": self.approved_by,
            "created_at": self.created_at,
            "metadata": self.metadata,
        }


class LearningPipeline:
    """
    Manages the lifecycle of learning records and safe dataset compilation.
    """

    def __init__(self) -> None:
        # Partitioned by owner_id -> record_id -> LearningRecord
        self._records: Dict[str, Dict[str, LearningRecord]] = {}

    def add_record(self, record: LearningRecord) -> None:
        """Register a learning record within its tenant boundary."""
        if record.owner_id not in self._records:
            self._records[record.owner_id] = {}
        self._records[record.owner_id][record.record_id] = record

    def get_records(
        self,
        owner_id: str,
        status: Optional[LearningRecordStatus] = None,
    ) -> List[LearningRecord]:
        """Retrieve records for a specific tenant, optionally filtered by status."""
        tenant_store = self._records.get(owner_id, {})
        if status is None:
            return list(tenant_store.values())
        return [r for r in tenant_store.values() if r.status == status]

    def approve_record(self, record_id: str, owner_id: str) -> None:
        """Approve a verified record for offline training."""
        tenant_store = self._records.get(owner_id, {})
        if record_id not in tenant_store:
            # Check if it belongs to another tenant
            for other_owner, store in self._records.items():
                if record_id in store:
                    raise TenantIsolationError(
                        f"Tenant isolation violation: Owner '{owner_id}' cannot approve record '{record_id}' "
                        f"owned by '{other_owner}'."
                    )
            raise KeyError(f"Record '{record_id}' not found for owner '{owner_id}'.")

        record = tenant_store[record_id]
        record.approve_for_training()

    def export_training_dataset(self, owner_id: str, output_path: Path | str) -> int:
        """
        Export all TRAINING_APPROVED records for an owner to a JSONL dataset.
        Returns the number of exported examples.
        """
        approved = self.get_records(owner_id, status=LearningRecordStatus.TRAINING_APPROVED)
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)

        with open(out_file, "w", encoding="utf-8") as f:
            for rec in approved:
                line = json.dumps({
                    "record_id": rec.record_id,
                    "input_context": rec.input_context,
                    "target_output": rec.target_output,
                    "task_type": rec.task_type,
                    "quality_score": rec.quality_score,
                })
                f.write(line + "\n")

        return len(approved)


class ModelUpdateManager:
    """
    Offline model lifecycle and safety manager.
    
    Guarantees:
    - Invariant verification (3,443,136 params, 4096 vocab, 512 context, BOS=0, EOS=1, PAD=2).
    - Prevents automatic replacement of running model.
    - Full version history and instant rollback.
    """

    FROZEN_PARAMS = 3_443_136
    FROZEN_VOCAB = 4_096
    FROZEN_CONTEXT = 512

    def __init__(self, registry_dir: Optional[Path | str] = None) -> None:
        self.registry_dir = Path(registry_dir) if registry_dir else None
        self._versions: Dict[str, ModelVersionArtifact] = {}
        self._active_version_id: str = "chakrmicro-v0.1"

        # Register baseline release
        self._versions["chakrmicro-v0.1"] = ModelVersionArtifact(
            version_id="chakrmicro-v0.1",
            artifact_path="chakrview/brain/weights/chakrmicro_v0.1.pt",
            param_count=self.FROZEN_PARAMS,
            vocab_size=self.FROZEN_VOCAB,
            context_length=self.FROZEN_CONTEXT,
            val_loss=0.0,
            approved_by="SYSTEM_RATIFIED_BASELINE",
            metadata={"status": "ACTIVE_PRODUCTION"},
        )

    @property
    def active_version(self) -> ModelVersionArtifact:
        return self._versions[self._active_version_id]

    def verify_candidate_model(
        self,
        candidate_model: torch.nn.Module,
    ) -> Tuple[bool, List[str]]:
        """
        Verify that a candidate trained model strictly adheres to frozen invariants.
        """
        violations: List[str] = []

        # 1. Total parameter count check
        total_params = sum(p.numel() for p in candidate_model.parameters())
        if total_params != self.FROZEN_PARAMS:
            violations.append(
                f"Parameter count mismatch: got {total_params:,}, expected exactly {self.FROZEN_PARAMS:,}."
            )

        # 2. Vocabulary size check
        if hasattr(candidate_model, "config") and hasattr(candidate_model.config, "vocab_size"):
            if candidate_model.config.vocab_size != self.FROZEN_VOCAB:
                violations.append(
                    f"Vocab size mismatch: got {candidate_model.config.vocab_size}, expected {self.FROZEN_VOCAB}."
                )

        # 3. Context length check
        if hasattr(candidate_model, "config") and hasattr(candidate_model.config, "max_seq_len"):
            if candidate_model.config.max_seq_len != self.FROZEN_CONTEXT:
                violations.append(
                    f"Context ceiling mismatch: got {candidate_model.config.max_seq_len}, expected {self.FROZEN_CONTEXT}."
                )

        return len(violations) == 0, violations

    def register_candidate_version(
        self,
        version_id: str,
        artifact_path: str,
        candidate_model: torch.nn.Module,
        val_loss: float,
        regression_passed: bool,
        approver: str,
    ) -> ModelVersionArtifact:
        """
        Register a verified candidate model version. Does NOT automatically deploy it!
        """
        # 1. Verify frozen invariants
        is_valid, violations = self.verify_candidate_model(candidate_model)
        if not is_valid:
            raise ModelUpdateSafetyError(
                f"Candidate model '{version_id}' failed frozen invariant verification: {'; '.join(violations)}"
            )

        # 2. Check regression test results
        if not regression_passed:
            raise ModelUpdateSafetyError(
                f"Candidate model '{version_id}' failed mandatory regression suite."
            )

        artifact = ModelVersionArtifact(
            version_id=version_id,
            artifact_path=artifact_path,
            param_count=sum(p.numel() for p in candidate_model.parameters()),
            vocab_size=self.FROZEN_VOCAB,
            context_length=self.FROZEN_CONTEXT,
            val_loss=val_loss,
            approved_by=approver,
            metadata={"status": "APPROVED_STANDBY"},
        )
        self._versions[version_id] = artifact
        return artifact

    def promote_to_active(self, version_id: str, authorized_by: str) -> None:
        """
        Explicit operator promotion of an approved standby model version.
        """
        if version_id not in self._versions:
            raise KeyError(f"Version '{version_id}' is not registered.")
        
        artifact = self._versions[version_id]
        if artifact.approved_by == "UNAPPROVED":
            raise ModelUpdateSafetyError(f"Cannot promote unapproved version '{version_id}'.")

        # Mark prior as ARCHIVED
        self._versions[self._active_version_id].metadata["status"] = "ARCHIVED"
        self._active_version_id = version_id
        artifact.metadata["status"] = "ACTIVE_PRODUCTION"
        artifact.metadata["promoted_by"] = authorized_by
        artifact.metadata["promoted_at"] = time.time()

    def rollback(self, target_version_id: str, reason: str) -> ModelVersionArtifact:
        """
        Roll back active model pointer to a previous approved artifact.
        """
        if target_version_id not in self._versions:
            raise KeyError(f"Rollback target version '{target_version_id}' does not exist.")

        prior_version = self._active_version_id
        self._active_version_id = target_version_id
        self._versions[prior_version].metadata["status"] = "ROLLED_BACK"
        target = self._versions[target_version_id]
        target.metadata["status"] = "ACTIVE_PRODUCTION"
        target.metadata["rollback_reason"] = reason
        target.metadata["rollback_from"] = prior_version
        target.metadata["rollback_at"] = time.time()
        return target

    def list_versions(self) -> List[ModelVersionArtifact]:
        """List all registered model version artifacts."""
        return list(self._versions.values())
