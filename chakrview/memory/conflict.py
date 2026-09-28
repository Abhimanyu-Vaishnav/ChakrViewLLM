"""
Memory Conflict Model & Resolution Foundation for ChakrView (Step 16).

Detects and tracks divergent, contradictory, or updated factual statements
without silently deleting historical records.
Example:
Memory A: "Company revenue = ₹20 lakh" (Turn 1, Oct 2025)
Memory B: "Company revenue = ₹32 lakh" (Turn 8, Jan 2026)
Preserves provenance and status, enabling governed policy resolution.
"""

from dataclasses import dataclass, field, asdict
from enum import Enum
import re
import time
from typing import Dict, List, Optional, Any, Tuple
import uuid

from chakrview.memory.record import MemoryRecord, MemoryValidity


class ConflictStatus(str, Enum):
    """Lifecycle state of a detected factual conflict."""
    DETECTED = "detected"       # Initial divergence identified
    FLAGGED = "flagged"         # Marked for explicit user/policy disambiguation
    RESOLVED = "resolved"       # Resolved via supersession or explicit confirmation


@dataclass
class MemoryConflict:
    """
    Representation of a factual conflict between an existing memory and a new candidate.
    """
    conflict_id: str
    owner_id: str
    existing_memory_id: str
    candidate_content: str
    conflict_type: str = "attribute_divergence"
    status: ConflictStatus = ConflictStatus.DETECTED
    detected_at: float = field(default_factory=time.time)
    resolution_strategy: Optional[str] = None
    resolved_memory_id: Optional[str] = None
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "conflict_id": self.conflict_id,
            "owner_id": self.owner_id,
            "existing_memory_id": self.existing_memory_id,
            "candidate_content": self.candidate_content,
            "conflict_type": self.conflict_type,
            "status": self.status.value,
            "detected_at": self.detected_at,
            "resolution_strategy": self.resolution_strategy,
            "resolved_memory_id": self.resolved_memory_id,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryConflict":
        d = dict(data)
        d["status"] = ConflictStatus(d["status"])
        return cls(**d)


class ConflictDetector:
    """
    Identifies contradictory facts across active memories.
    """

    @staticmethod
    def extract_subject_attribute_pairs(text: str) -> List[Tuple[str, str]]:
        """
        Simple deterministic entity/property key extraction.
        E.g. "Company revenue = 20 lakh" -> ("company revenue", "20 lakh")
        """
        pairs: List[Tuple[str, str]] = []
        # Pattern: <key> is / = / : / has <value>
        matches = re.findall(
            r"([a-zA-Z\s]{3,30})\s*(?:=|\bis\b|\bhas\b|:)\s*([0-9a-zA-Z\s₹$%.,]+)",
            text,
            re.IGNORECASE,
        )
        for k, v in matches:
            pairs.append((k.strip().lower(), v.strip().lower()))
        return pairs

    @classmethod
    def detect_conflict(
        cls,
        candidate_content: str,
        existing_records: List[MemoryRecord],
    ) -> Optional[MemoryConflict]:
        """
        Check if candidate_content contradicts any active record in existing_records.
        """
        cand_pairs = cls.extract_subject_attribute_pairs(candidate_content)
        if not cand_pairs:
            return None

        for rec in existing_records:
            if rec.validity != MemoryValidity.ACTIVE:
                continue

            rec_pairs = cls.extract_subject_attribute_pairs(rec.content)
            for cand_k, cand_v in cand_pairs:
                for rec_k, rec_v in rec_pairs:
                    # If keys are highly overlapping / identical but values clearly diverge
                    if (cand_k in rec_k or rec_k in cand_k) and cand_v != rec_v:
                        return MemoryConflict(
                            conflict_id=f"conf_{uuid.uuid4().hex[:8]}",
                            owner_id=rec.owner_id,
                            existing_memory_id=rec.memory_id,
                            candidate_content=candidate_content,
                            conflict_type="attribute_divergence",
                            status=ConflictStatus.DETECTED,
                            notes=f"Conflicting values for '{cand_k}': '{rec_v}' vs '{cand_v}'",
                        )

        return None
