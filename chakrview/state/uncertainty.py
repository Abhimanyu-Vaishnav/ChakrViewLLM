"""
Explicit Uncertainty State Representation for ChakrView (Step 18).

Provides formal structures to track, quantify, and inspect uncertainty
associated with decisions, beliefs, sensors, or task outcomes.

Architectural Rule:
DO NOT FABRICATE CONFIDENCE VALUES.
If the system has no verifiable basis for a confidence metric, it must
explicitly flag is_uncalibrated=True and state the empirical reason.
"""

from dataclasses import dataclass, field, asdict
import time
from typing import Dict, List, Optional, Any
import uuid


@dataclass
class Uncertainty:
    """
    Explicit, auditable representation of epistemic or aleatoric uncertainty.
    """
    confidence: float
    reason: str
    source: str
    uncertainty_id: str = field(default_factory=lambda: f"unc_{uuid.uuid4().hex[:8]}")
    timestamp: float = field(default_factory=time.time)
    evidence_refs: List[str] = field(default_factory=list)
    is_uncalibrated: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"Confidence must be in [0.0, 1.0], got {self.confidence}")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Uncertainty":
        return cls(**data)


@dataclass
class UncertaintyState:
    """
    State container tracking active uncertainties across tasks, knowledge, or sensors.
    """
    uncertainties: Dict[str, Uncertainty] = field(default_factory=dict)

    def record_uncertainty(
        self,
        key: str,
        confidence: float,
        reason: str,
        source: str,
        evidence_refs: Optional[List[str]] = None,
        is_uncalibrated: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Uncertainty:
        """
        Record a quantified uncertainty for an entity or assertion key.
        """
        unc = Uncertainty(
            confidence=confidence,
            reason=reason,
            source=source,
            evidence_refs=evidence_refs or [],
            is_uncalibrated=is_uncalibrated,
            metadata=metadata or {},
        )
        self.uncertainties[key] = unc
        return unc

    def get_uncertainty(self, key: str) -> Optional[Uncertainty]:
        """Query uncertainty for a given key."""
        return self.uncertainties.get(key)

    def has_uncertainty(self, key: str) -> bool:
        """Check whether an uncertainty is recorded for key."""
        return key in self.uncertainties

    def clear_uncertainty(self, key: str) -> Optional[Uncertainty]:
        """Remove uncertainty record once resolved or updated."""
        return self.uncertainties.pop(key, None)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "uncertainties": {k: v.to_dict() for k, v in self.uncertainties.items()}
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UncertaintyState":
        uncertainties = {
            k: Uncertainty.from_dict(v)
            for k, v in data.get("uncertainties", {}).items()
        }
        return cls(uncertainties=uncertainties)
