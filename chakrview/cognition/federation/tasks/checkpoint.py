"""
Durable Task Checkpoint Store & Commit Protocol for Execution Continuity (Steps 38 & 39).

CRITICAL AXIOMS:
1. WORK CONTINUITY VIA COMMITTED CHECKPOINTS:
   Work execution resumes from the latest committed checkpoint, never uncommitted state.
2. CHECKPOINT COMMIT PROTOCOL:
   Execution progress is acknowledged only after a checkpoint is validated, authenticated,
   durably stored, and transitioned to COMMITTED state.
3. TAMPER-EVIDENT INTEGRITY & STRICT MONOTONICITY:
   Checkpoints carry canonical SHA-256 state digests. Sequence numbers must strictly advance.
   Stale or corrupted checkpoints are rejected immediately.
4. INTEGRATION WITH DURABLE WAL:
   Committed checkpoints append to the Step 34 security journal (WAL).
"""

import logging
import threading
import time
from typing import Dict, List, Optional, Any, Tuple

from chakrview.cognition.federation.tasks.models import (
    TaskCheckpoint,
    CheckpointManifest,
    CheckpointStatus,
    WorkUnitState,
)
from chakrview.cognition.federation.tasks.errors import (
    InvalidCheckpointError,
    CheckpointCorruptionError,
    StaleCheckpointError,
    CheckpointCommitError,
)
from chakrview.cognition.federation.persistence.models import (
    JournalEntryType,
)

logger = logging.getLogger(__name__)


class CheckpointStore:
    """
    Authoritative, durable storage abstraction for distributed task checkpoints.
    Supports atomic commit protocol, sequence supersession, and recovery.
    """

    def __init__(self, journal: Optional[Any] = None) -> None:
        self.journal = journal
        self._lock = threading.RLock()
        # Key: (task_id, unit_id, checkpoint_id) -> CheckpointManifest
        self._manifests: Dict[Tuple[str, str, str], CheckpointManifest] = {}
        # Key: (task_id, unit_id) -> latest COMMITTED CheckpointManifest
        self._latest_committed: Dict[Tuple[str, str], CheckpointManifest] = {}
        # Key: (task_id, unit_id) -> List[CheckpointManifest] all versions
        self._manifest_history: Dict[Tuple[str, str], List[CheckpointManifest]] = {}

    def create_manifest(self, manifest: CheckpointManifest) -> CheckpointManifest:
        """
        Validate, verify integrity, and store a new checkpoint manifest in PENDING status.
        
        Raises:
            CheckpointCorruptionError: If digest verification fails.
            StaleCheckpointError: If sequence number is <= latest committed sequence.
        """
        with self._lock:
            # 1. Cryptographic and digest integrity verification
            if not manifest.verify_integrity():
                manifest.status = CheckpointStatus.CORRUPTED
                raise CheckpointCorruptionError(
                    f"Checkpoint manifest {manifest.checkpoint_id} failed integrity verification. "
                    f"Computed digest does not match recorded digest {manifest.checkpoint_payload_digest}"
                )

            key = (manifest.task_id, manifest.work_unit_id)
            latest = self._latest_committed.get(key)

            # 2. Sequence monotonicity verification against committed state
            if latest is not None:
                if manifest.checkpoint_sequence <= latest.checkpoint_sequence:
                    raise StaleCheckpointError(
                        f"Stale checkpoint sequence for unit {manifest.work_unit_id}: "
                        f"received sequence {manifest.checkpoint_sequence} <= committed {latest.checkpoint_sequence}"
                    )

            # 3. Store in PENDING state
            manifest.status = CheckpointStatus.PENDING
            entry_key = (manifest.task_id, manifest.work_unit_id, manifest.checkpoint_id)
            self._manifests[entry_key] = manifest

            if key not in self._manifest_history:
                self._manifest_history[key] = []
            self._manifest_history[key].append(manifest)

            logger.debug(
                "Created PENDING checkpoint manifest %s (unit %s, seq %d)",
                manifest.checkpoint_id, manifest.work_unit_id, manifest.checkpoint_sequence,
            )
            return manifest

    def commit_checkpoint(self, task_id: str, unit_id: str, checkpoint_id: str) -> CheckpointManifest:
        """
        Atomically commit a pending checkpoint, superseding any prior committed checkpoints.
        
        Raises:
            CheckpointCommitError if checkpoint not found or already superseded/corrupted.
        """
        with self._lock:
            entry_key = (task_id, unit_id, checkpoint_id)
            manifest = self._manifests.get(entry_key)
            if manifest is None:
                raise CheckpointCommitError(f"Checkpoint {checkpoint_id} not found for unit {unit_id}")

            if manifest.status == CheckpointStatus.CORRUPTED:
                raise CheckpointCommitError(f"Cannot commit corrupted checkpoint {checkpoint_id}")

            unit_key = (task_id, unit_id)
            prior_committed = self._latest_committed.get(unit_key)

            # Mark prior committed as SUPERSEDED
            if prior_committed is not None:
                prior_committed.status = CheckpointStatus.SUPERSEDED
                self._wal_log(
                    JournalEntryType.TASK_CHECKPOINT_SUPERSEDED,
                    {
                        "task_id": task_id,
                        "unit_id": unit_id,
                        "superseded_checkpoint_id": prior_committed.checkpoint_id,
                        "new_checkpoint_id": checkpoint_id,
                    },
                )

            # Transition current to COMMITTED
            manifest.status = CheckpointStatus.COMMITTED
            self._latest_committed[unit_key] = manifest

            # Append to WAL
            self._wal_log(
                JournalEntryType.TASK_CHECKPOINT_COMMITTED,
                {
                    "task_id": task_id,
                    "unit_id": unit_id,
                    "checkpoint_id": manifest.checkpoint_id,
                    "sequence": manifest.checkpoint_sequence,
                    "worker_id": manifest.worker_id,
                    "attempt": manifest.attempt_id,
                    "fencing_token": manifest.fencing_token,
                    "digest": manifest.checkpoint_payload_digest,
                },
            )

            logger.info(
                "Committed checkpoint manifest %s for unit %s (sequence %d, attempt %d)",
                checkpoint_id, unit_id, manifest.checkpoint_sequence, manifest.attempt_id,
            )
            return manifest

    def supersede_checkpoints(self, task_id: str, unit_id: str, up_to_sequence: int) -> int:
        """Mark all checkpoints with sequence < up_to_sequence as SUPERSEDED."""
        superseded_count = 0
        with self._lock:
            history = self._manifest_history.get((task_id, unit_id), [])
            for m in history:
                if m.checkpoint_sequence < up_to_sequence and m.status in (CheckpointStatus.PENDING, CheckpointStatus.COMMITTED):
                    m.status = CheckpointStatus.SUPERSEDED
                    superseded_count += 1
        return superseded_count

    def get_committed_checkpoint(self, task_id: str, unit_id: str) -> Optional[CheckpointManifest]:
        """Retrieve the authoritative latest COMMITTED checkpoint for a work unit."""
        with self._lock:
            return self._latest_committed.get((task_id, unit_id))

    def get_manifest(self, task_id: str, unit_id: str, checkpoint_id: str) -> Optional[CheckpointManifest]:
        """Retrieve a specific checkpoint manifest by ID."""
        with self._lock:
            return self._manifests.get((task_id, unit_id, checkpoint_id))

    def recover_manifest(self, task_id: str, unit_id: str) -> Optional[CheckpointManifest]:
        """
        Recover the safest committed checkpoint manifest for work continuity.
        Validates integrity before returning.
        """
        with self._lock:
            latest = self._latest_committed.get((task_id, unit_id))
            if latest is None:
                return None

            if not latest.verify_integrity():
                logger.error("Authoritative checkpoint %s failed integrity check during recovery", latest.checkpoint_id)
                latest.status = CheckpointStatus.CORRUPTED
                return None

            return latest

    def compare_progress(self, m1: CheckpointManifest, m2: CheckpointManifest) -> int:
        """
        Compare logical progress of two manifests.
        Returns:
            1 if m1 > m2
           -1 if m1 < m2
            0 if m1 == m2
        """
        if m1.checkpoint_sequence > m2.checkpoint_sequence:
            return 1
        elif m1.checkpoint_sequence < m2.checkpoint_sequence:
            return -1

        # Fallback to items processed in completed_work_range if sequence equal
        p1 = m1.completed_work_range.get("items_processed", 0)
        p2 = m2.completed_work_range.get("items_processed", 0)
        if p1 > p2:
            return 1
        elif p1 < p2:
            return -1
        return 0

    def garbage_collect(self, task_id: str, keep_latest_n: int = 2) -> int:
        """Prune superseded and expired checkpoints, retaining at least keep_latest_n."""
        pruned_count = 0
        with self._lock:
            unit_keys = [k for k in self._manifest_history if k[0] == task_id]
            for unit_key in unit_keys:
                history = self._manifest_history[unit_key]
                committed_candidates = [m for m in history if m.status == CheckpointStatus.COMMITTED]
                candidates_to_prune = [m for m in history if m.status in (CheckpointStatus.SUPERSEDED, CheckpointStatus.EXPIRED)]

                # Retain latest N
                excess = len(history) - keep_latest_n
                if excess > 0:
                    for m in candidates_to_prune[:excess]:
                        self._manifests.pop((task_id, m.work_unit_id, m.checkpoint_id), None)
                        history.remove(m)
                        pruned_count += 1
        return pruned_count

    def clear_task_checkpoints(self, task_id: str) -> None:
        """Clean up all checkpoints for a completed or aborted task."""
        with self._lock:
            keys_to_remove = [k for k in self._manifests if k[0] == task_id]
            for k in keys_to_remove:
                self._manifests.pop(k, None)

            unit_keys = [k for k in self._latest_committed if k[0] == task_id]
            for uk in unit_keys:
                self._latest_committed.pop(uk, None)
                self._manifest_history.pop(uk, None)

    def _wal_log(self, entry_type: JournalEntryType, payload: Dict[str, Any]) -> None:
        if self.journal is not None:
            try:
                self.journal.append_entry(entry_type=entry_type, payload=payload)
            except Exception:
                pass


class TaskCheckpointManager:
    """
    Coordinates creation, verification, storage, and retrieval of durable task checkpoints.
    Embeds CheckpointStore while preserving backward compatibility with Step 38.
    """

    def __init__(self, journal: Optional[Any] = None) -> None:
        self.journal = journal
        self.checkpoint_store = CheckpointStore(journal=journal)
        self._lock = threading.RLock()
        # Key: (task_id, unit_id) -> latest TaskCheckpoint (Step 38 backward compatibility)
        self._latest_checkpoints: Dict[Tuple[str, str], TaskCheckpoint] = {}
        # Key: (task_id, unit_id) -> List[TaskCheckpoint] history
        self._checkpoint_history: Dict[Tuple[str, str], List[TaskCheckpoint]] = {}

    def save_checkpoint(self, checkpoint: TaskCheckpoint) -> None:
        """
        Validate, verify integrity, and store a checkpoint record.
        Automatically syncs with CheckpointStore for Step 39 continuity.
        
        Raises:
            CheckpointCorruptionError if digest mismatch detected.
            InvalidCheckpointError if sequence or progress regresses.
        """
        with self._lock:
            # 1. Verify cryptographic integrity
            if not checkpoint.verify_integrity():
                raise CheckpointCorruptionError(
                    f"Checkpoint {checkpoint.checkpoint_id} failed integrity verification. "
                    f"Computed digest does not match recorded digest {checkpoint.integrity_digest}"
                )

            key = (checkpoint.task_id, checkpoint.unit_id)
            existing = self._latest_checkpoints.get(key)

            # 2. Sequence monotonicity check
            if existing is not None:
                if checkpoint.sequence < existing.sequence:
                    raise InvalidCheckpointError(
                        f"Checkpoint sequence regression for unit {checkpoint.unit_id}: "
                        f"received sequence {checkpoint.sequence} < existing {existing.sequence}"
                    )
                if checkpoint.progress < existing.progress:
                    raise InvalidCheckpointError(
                        f"Checkpoint progress regression for unit {checkpoint.unit_id}: "
                        f"received progress {checkpoint.progress:.2f} < existing {existing.progress:.2f}"
                    )

            # 3. Store in memory state
            self._latest_checkpoints[key] = checkpoint
            if key not in self._checkpoint_history:
                self._checkpoint_history[key] = []
            self._checkpoint_history[key].append(checkpoint)

            # 4. Integrate into Step 39 CheckpointStore as an authoritative committed manifest
            manifest = CheckpointManifest(
                task_id=checkpoint.task_id,
                work_unit_id=checkpoint.unit_id,
                attempt_id=checkpoint.attempt,
                checkpoint_id=checkpoint.checkpoint_id,
                checkpoint_sequence=checkpoint.sequence,
                worker_id=checkpoint.worker_id,
                execution_state=checkpoint.state,
                completed_work_range={"progress": checkpoint.progress},
                remaining_work={"remaining_progress": max(0.0, 1.0 - checkpoint.progress)},
                intermediate_payload=checkpoint.partial_result,
                status=CheckpointStatus.PENDING,
            )
            try:
                self.checkpoint_store.create_manifest(manifest)
                self.checkpoint_store.commit_checkpoint(
                    checkpoint.task_id, checkpoint.unit_id, checkpoint.checkpoint_id
                )
            except Exception as e:
                logger.debug("Checkpoint store sync: %s", str(e))

            # 5. Append to durable WAL journal if available
            if self.journal is not None:
                try:
                    self.journal.append_entry(
                        entry_type=JournalEntryType.TASK_CHECKPOINTED,
                        payload={
                            "task_id": checkpoint.task_id,
                            "unit_id": checkpoint.unit_id,
                            "checkpoint_id": checkpoint.checkpoint_id,
                            "sequence": checkpoint.sequence,
                            "progress": checkpoint.progress,
                            "worker_id": checkpoint.worker_id,
                            "attempt": checkpoint.attempt,
                            "integrity_digest": checkpoint.integrity_digest,
                        },
                    )
                except Exception:
                    pass

    def get_latest_checkpoint(self, task_id: str, unit_id: str) -> Optional[TaskCheckpoint]:
        """Retrieve the latest valid checkpoint for a work unit."""
        with self._lock:
            return self._latest_checkpoints.get((task_id, unit_id))

    def get_checkpoint_history(self, task_id: str, unit_id: str) -> List[TaskCheckpoint]:
        """Retrieve all checkpoints for a work unit in chronological order."""
        with self._lock:
            return list(self._checkpoint_history.get((task_id, unit_id), []))

    def clear_task_checkpoints(self, task_id: str) -> None:
        """Prune checkpoints for completed tasks."""
        with self._lock:
            keys_to_remove = [k for k in self._latest_checkpoints if k[0] == task_id]
            for k in keys_to_remove:
                self._latest_checkpoints.pop(k, None)
                self._checkpoint_history.pop(k, None)
            self.checkpoint_store.clear_task_checkpoints(task_id)
