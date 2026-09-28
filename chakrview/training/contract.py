"""
Training Data Contracts for ChakrView (Step 22).

Defines strongly typed training contracts that enforce:
RAW USER TEXT != VERIFIED TRAINING DATA

Guarantees:
1. Only TRAINING_APPROVED learning records may enter training datasets.
2. Full provenance tracking (source record, owner, task, timestamps).
3. Explicit sequence token accounting and truncation logging (never silent truncation).
4. Deterministic hashing and fingerprinting for reproducible datasets.
"""

from dataclasses import dataclass, field, asdict
import hashlib
import json
import time
from typing import Dict, List, Optional, Any

from chakrview.intelligence.contracts import LearningRecord, LearningRecordStatus


class TrainingRecordEligibilityError(ValueError):
    """Raised when an unapproved, unverified, or quarantined record is submitted for training."""
    pass


@dataclass
class TokenizerFingerprint:
    """Deterministic cryptographic fingerprint of a tokenizer configuration."""
    vocab_size: int
    num_merges: int
    bos_id: int
    eos_id: int
    pad_id: int
    fingerprint_hash: str

    @classmethod
    def from_tokenizer(cls, tokenizer: Any) -> "TokenizerFingerprint":
        vocab_size = getattr(tokenizer, "vocab_size", 4096)
        merges = getattr(tokenizer, "merges", {})
        num_merges = len(merges)
        bos_id = 0
        eos_id = 1
        pad_id = 2

        hasher = hashlib.sha256()
        hasher.update(f"vocab:{vocab_size}:merges:{num_merges}:bos:{bos_id}:eos:{eos_id}:pad:{pad_id}".encode("utf-8"))
        if merges:
            sorted_items = sorted(list(merges.items()), key=lambda x: x[1])[:50]
            hasher.update(str(sorted_items).encode("utf-8"))

        return cls(
            vocab_size=vocab_size,
            num_merges=num_merges,
            bos_id=bos_id,
            eos_id=eos_id,
            pad_id=pad_id,
            fingerprint_hash=hasher.hexdigest()[:16],
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TrainingExample:
    """
    Formally validated, tokenized training example derived from an approved LearningRecord.
    
    Preserves end-to-end provenance:
    source_record_id -> input/target tokens -> dataset manifest -> training run.
    """
    example_id: str
    source_record_id: str
    owner_id: str
    session_id: str
    input_text: str
    target_text: str
    input_token_ids: List[int]
    target_token_ids: List[int]
    sequence_token_ids: List[int]
    input_token_count: int
    target_token_count: int
    total_token_count: int
    is_truncated: bool
    truncation_notes: Optional[str] = None
    task_type: str = "general"
    quality_score: float = 1.0
    tokenizer_fingerprint: str = ""
    dataset_version: str = "v1.0"
    created_at: float = field(default_factory=time.time)
    source_provenance: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingExample":
        return cls(**data)


@dataclass
class TrainingDatasetManifest:
    """
    Cryptographically verifiable manifest for an offline training dataset.
    """
    manifest_id: str
    dataset_version: str
    dataset_fingerprint: str
    tokenizer_fingerprint: str
    total_examples: int
    total_tokens: int
    train_examples_count: int
    val_examples_count: int
    test_examples_count: int
    max_sequence_length: int = 512
    created_at: float = field(default_factory=time.time)
    provenance_summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingDatasetManifest":
        return cls(**data)


def validate_learning_record_for_training(record: LearningRecord) -> None:
    """
    Verify that a learning record is eligible to enter a training dataset.
    
    Hard constraints:
    1. Must have status == LearningRecordStatus.TRAINING_APPROVED.
    2. Must NOT be CANDIDATE, VERIFIED (unapproved), REJECTED, or QUARANTINED.
    3. Must have non-empty input and target texts.
    """
    if record.status == LearningRecordStatus.QUARANTINED:
        raise TrainingRecordEligibilityError(
            f"Record '{record.record_id}' is QUARANTINED (reason: {record.quarantine_reason}). "
            f"Quarantined data cannot enter training datasets."
        )

    if record.status == LearningRecordStatus.REJECTED:
        raise TrainingRecordEligibilityError(
            f"Record '{record.record_id}' is REJECTED (reason: {record.rejection_reason}). "
            f"Rejected records cannot enter training datasets."
        )

    if record.status == LearningRecordStatus.CANDIDATE:
        raise TrainingRecordEligibilityError(
            f"Record '{record.record_id}' is in CANDIDATE status. "
            f"Raw runtime observations must be verified and approved before training."
        )

    if record.status == LearningRecordStatus.VERIFIED:
        raise TrainingRecordEligibilityError(
            f"Record '{record.record_id}' is VERIFIED but not yet TRAINING_APPROVED. "
            f"Explicit signoff is required."
        )

    if record.status != LearningRecordStatus.TRAINING_APPROVED:
        raise TrainingRecordEligibilityError(
            f"Record '{record.record_id}' has unrecognized or unapproved status: {record.status}."
        )

    if not record.input_context.strip():
        raise TrainingRecordEligibilityError(f"Record '{record.record_id}' has empty input_context.")

    if not record.target_output.strip():
        raise TrainingRecordEligibilityError(f"Record '{record.record_id}' has empty target_output.")
