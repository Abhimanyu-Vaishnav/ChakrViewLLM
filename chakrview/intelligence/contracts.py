"""
Neural Reasoning Integration Contracts for ChakrView (Step 20).

Defines the formal boundaries and data structures connecting:
- ChakrMicro Neural Core
- Cognitive State
- Governed Reasoning
- Memory & Knowledge
- Capability & Environment
- Feedback & Offline Learning

Architectural Principles:
1. ChakrMicro v0.1 is FROZEN: 3,443,136 params, 4096 vocab, 512 context, BOS=0, EOS=1, PAD=2.
2. RAW USER TEXT != VERIFIED TRAINING DATA: Explicit validation lifecycle required.
3. DATA != AUTHORITY: Retrieved facts, memory, and neural candidates never confer authority.
4. NO FAKE INTELLIGENCE: Explicitly distinguish learned probability from deterministic logic.
   Do not fabricate calibrated uncertainty when unavailable.
5. ZERO RUNTIME WEIGHT UPDATES: Inference and training are strictly disjoint lifecycles.
"""

from dataclasses import dataclass, field, asdict
from enum import Enum
import time
from typing import Dict, List, Optional, Any
import uuid


class LearningRecordStatus(str, Enum):
    """
    Lifecycle stages for candidate training examples.
    
    Guarantees that raw observations or unverified neural outputs cannot
    directly enter the training corpus without rigorous validation.
    """
    CANDIDATE = "CANDIDATE"                   # Newly captured from runtime execution
    VERIFIED = "VERIFIED"                     # Passed rule/reasoning verification check
    REJECTED = "REJECTED"                     # Failed verification or safety constraints
    QUARANTINED = "QUARANTINED"               # Suspected prompt injection or contamination
    TRAINING_APPROVED = "TRAINING_APPROVED"   # Formally signed off for offline training dataset


@dataclass
class UncertaintyMetric:
    """
    Honest representation of neural uncertainty.
    
    CRITICAL CONSTRAINT:
    Do NOT fabricate calibrated confidence scores. If mathematically calibrated
    probabilities are not computed, is_available MUST be False and is_calibrated MUST be False.
    """
    entropy: Optional[float] = None
    top_token_margin: Optional[float] = None
    is_calibrated: bool = False
    is_available: bool = False
    notes: str = "Uncalibrated / raw logits"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def unavailable(cls, reason: str = "Uncertainty calculation not requested or model uncalibrated") -> "UncertaintyMetric":
        return cls(entropy=None, top_token_margin=None, is_calibrated=False, is_available=False, notes=reason)


@dataclass
class NeuralInferenceRequest:
    """
    Structured request for ChakrMicro neural token generation.
    """
    prompt_text: str
    prompt_tokens: Optional[List[int]] = None
    max_new_tokens: int = 64
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.95
    stop_token_ids: List[int] = field(default_factory=lambda: [1])  # EOS = 1
    compute_uncertainty: bool = False
    context_provenance: Dict[str, Any] = field(default_factory=dict)
    owner_id: str = "default_user"
    session_id: str = "default_session"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class NeuralInferenceResult:
    """
    Formal neural output contract for ChakrMicro.
    
    Guarantees:
    - Zero runtime weight mutation: weights_modified is permanently False.
    - Full provenance: ties output to input context and model version.
    - Honest uncertainty: flags uncalibrated or unavailable confidence.
    """
    text: str
    token_ids: List[int]
    prompt_tokens_count: int
    generated_tokens_count: int
    total_tokens_count: int
    latency_ms: float
    stop_reason: str
    uncertainty: UncertaintyMetric = field(default_factory=UncertaintyMetric.unavailable)
    model_version: str = "chakrmicro-v0.1"
    context_provenance: Dict[str, Any] = field(default_factory=dict)
    weights_modified: bool = False  # Architectural invariant: ALWAYS False at runtime

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["uncertainty"] = self.uncertainty.to_dict()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NeuralInferenceResult":
        data = dict(data)
        if "uncertainty" in data and isinstance(data["uncertainty"], dict):
            data["uncertainty"] = UncertaintyMetric(**data["uncertainty"])
        return cls(**data)


@dataclass
class LearningRecord:
    """
    Governed training example for offline model improvement.
    
    Preserves strict separation between runtime execution and model updates:
    RAW USER TEXT != VERIFIED TRAINING DATA
    """
    record_id: str
    owner_id: str
    session_id: str
    input_context: str
    target_output: str
    task_type: str = "general"
    input_token_ids: Optional[List[int]] = None
    target_token_ids: Optional[List[int]] = None
    source_provenance: Dict[str, Any] = field(default_factory=dict)
    evidence_references: List[str] = field(default_factory=list)
    verification_status: str = "UNVERIFIED"  # "PASS", "FAIL", "UNVERIFIED"
    quality_score: float = 0.0               # 0.0 to 1.0
    status: LearningRecordStatus = LearningRecordStatus.CANDIDATE
    rejection_reason: Optional[str] = None
    quarantine_reason: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    model_version: str = "chakrmicro-v0.1"

    @classmethod
    def create_candidate(
        cls,
        input_context: str,
        target_output: str,
        owner_id: str = "default_user",
        session_id: str = "default_session",
        task_type: str = "general",
        source_provenance: Optional[Dict[str, Any]] = None,
        evidence_references: Optional[List[str]] = None,
    ) -> "LearningRecord":
        """Factory for a new candidate learning record."""
        now = time.time()
        return cls(
            record_id=f"rec_{uuid.uuid4().hex[:12]}",
            owner_id=owner_id,
            session_id=session_id,
            input_context=input_context,
            target_output=target_output,
            task_type=task_type,
            source_provenance=source_provenance or {},
            evidence_references=evidence_references or [],
            status=LearningRecordStatus.CANDIDATE,
            verification_status="UNVERIFIED",
            quality_score=0.0,
            created_at=now,
            updated_at=now,
        )

    def mark_verified(self, quality_score: float = 1.0) -> None:
        """Mark record as having passed structured verification."""
        if self.status == LearningRecordStatus.QUARANTINED:
            raise ValueError(f"Cannot verify quarantined record '{self.record_id}'.")
        self.status = LearningRecordStatus.VERIFIED
        self.verification_status = "PASS"
        self.quality_score = max(0.0, min(1.0, quality_score))
        self.updated_at = time.time()

    def mark_rejected(self, reason: str) -> None:
        """Reject record from future training."""
        self.status = LearningRecordStatus.REJECTED
        self.verification_status = "FAIL"
        self.rejection_reason = reason
        self.updated_at = time.time()

    def mark_quarantined(self, reason: str) -> None:
        """Quarantine record due to safety, injection, or contamination suspicion."""
        self.status = LearningRecordStatus.QUARANTINED
        self.verification_status = "QUARANTINED"
        self.quarantine_reason = reason
        self.updated_at = time.time()

    def approve_for_training(self) -> None:
        """Approve verified record for inclusion in offline dataset."""
        if self.status != LearningRecordStatus.VERIFIED:
            raise ValueError(
                f"Cannot approve record '{self.record_id}' for training: "
                f"current status is {self.status.value}, expected VERIFIED."
            )
        self.status = LearningRecordStatus.TRAINING_APPROVED
        self.updated_at = time.time()

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["status"] = self.status.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LearningRecord":
        data = dict(data)
        if "status" in data and isinstance(data["status"], str):
            data["status"] = LearningRecordStatus(data["status"])
        return cls(**data)
