"""
ChakrView Step 58: Memory Consolidation Engine.

Defines:
- SemanticMemoryEntry: Dataclass representing a consolidated, evidence-backed reusable pattern.
- MemoryConsolidator: Offline consolidation pipeline that aggregates raw verified ExperienceRecords,
  validates admission criteria, deduplicates, and promotes generalized patterns requiring >= 2 verified episodes.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
import hashlib
import time
from typing import Any, Dict, List, Optional, Sequence

from chakrview.learning.experience import ExperienceRecord


@dataclass
class SemanticMemoryEntry:
    """
    Consolidated, generalized pattern synthesized from multiple verified episodes.
    Requires >= 2 independent verified experiences as empirical evidence.
    """
    pattern_id: str
    pattern_name: str
    task_family: str
    problem_archetype: str
    solution_strategy: str
    supporting_evidence: List[str] = field(default_factory=list)  # List of experience_ids
    evidence_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    boundary_conditions: List[str] = field(default_factory=list)
    timestamp_utc: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SemanticMemoryEntry:
        return cls(
            pattern_id=data["pattern_id"],
            pattern_name=data["pattern_name"],
            task_family=data.get("task_family", "general"),
            problem_archetype=data["problem_archetype"],
            solution_strategy=data["solution_strategy"],
            supporting_evidence=list(data.get("supporting_evidence", [])),
            evidence_count=data.get("evidence_count", 0),
            success_count=data.get("success_count", 0),
            failure_count=data.get("failure_count", 0),
            boundary_conditions=list(data.get("boundary_conditions", [])),
            timestamp_utc=data.get("timestamp_utc", time.time()),
            metadata=dict(data.get("metadata", {})),
        )


class MemoryConsolidator:
    """
    Offline Memory Consolidator.
    Consolidates verified ExperienceRecords into generalized SemanticMemoryEntries.
    Enforces strict admission:
      - Experience must be verified (100% test pass in ChakrKshetra)
      - Minimum evidence threshold (>= 2 independent verified experiences) to promote to semantic pattern
      - Deduplication across similar diagnostic patterns
    """

    def __init__(self, min_evidence_threshold: int = 2) -> None:
        self.min_evidence_threshold = min_evidence_threshold
        self.episodic_store: Dict[str, ExperienceRecord] = {}
        self.semantic_store: Dict[str, SemanticMemoryEntry] = {}
        self.unconsolidated_candidates: Dict[str, List[ExperienceRecord]] = {}

    def admit_experience(self, record: ExperienceRecord) -> bool:
        """
        Admit a verified ExperienceRecord into the episodic store.
        Rejects unverified or incomplete records.
        """
        if not record.verified:
            return False
        if not record.what_worked or record.what_worked == "Did not converge to verified solution":
            return False

        self.episodic_store[record.experience_id] = record

        # Cluster by task_family and normalized diagnostic root
        cluster_key = self._compute_cluster_key(record)
        if cluster_key not in self.unconsolidated_candidates:
            self.unconsolidated_candidates[cluster_key] = []

        # Avoid exact duplicate experience insertion
        if not any(e.experience_id == record.experience_id for e in self.unconsolidated_candidates[cluster_key]):
            self.unconsolidated_candidates[cluster_key].append(record)

        return True

    def _compute_cluster_key(self, record: ExperienceRecord) -> str:
        """Group experiences by task_family and general category."""
        family = record.task_family.lower().strip()
        # Diagnostic archetype extraction
        diag = (record.diagnosis or "").lower()
        if "assert" in diag or "mismatch" in diag or "error" in diag:
            arch = "defect_repair"
        elif "transition" in family or "state" in family:
            arch = "state_transition"
        else:
            arch = "general"
        return f"{family}::{arch}"

    def consolidate(self) -> List[SemanticMemoryEntry]:
        """
        Run consolidation over unconsolidated candidate clusters.
        Promotes clusters meeting or exceeding the minimum evidence threshold.
        """
        newly_promoted: List[SemanticMemoryEntry] = []

        for cluster_key, records in list(self.unconsolidated_candidates.items()):
            # Filter only verified records
            verified_records = [r for r in records if r.verified]
            if len(verified_records) >= self.min_evidence_threshold:
                # Synthesize semantic pattern
                pattern = self._synthesize_pattern(cluster_key, verified_records)
                self.semantic_store[pattern.pattern_id] = pattern
                newly_promoted.append(pattern)

        return newly_promoted

    def _synthesize_pattern(self, cluster_key: str, records: List[ExperienceRecord]) -> SemanticMemoryEntry:
        """Deterministically synthesize a SemanticMemoryEntry from supporting experiences."""
        family, arch = cluster_key.split("::", 1)
        evidence_ids = [r.experience_id for r in records]

        # Extract representative problem archetype and solution strategy
        first_diag = records[0].diagnosis or "Execution discrepancy"
        archetype = f"Targeted defect in {family} leading to: {first_diag}"[:120]

        # Consolidated solution strategy
        strategies = [r.reusable_pattern for r in records if r.reusable_pattern]
        strategy = f"Verified repair pattern for {family}: {strategies[0]}" if strategies else "Execute verified correction"
        strategy = strategy[:150]

        pattern_id = f"sem_pat_{hashlib.sha256(cluster_key.encode('utf-8')).hexdigest()[:12]}"
        pattern_name = f"Consolidated_{family.title()}_{arch.title()}"

        boundaries = [
            f"Restricted to task family '{family}'",
            "Requires deterministic test validation in ChakrKshetra",
            "Do not apply if state schema does not match",
        ]

        return SemanticMemoryEntry(
            pattern_id=pattern_id,
            pattern_name=pattern_name,
            task_family=family,
            problem_archetype=archetype,
            solution_strategy=strategy,
            supporting_evidence=evidence_ids,
            evidence_count=len(evidence_ids),
            success_count=len(evidence_ids),
            failure_count=0,
            boundary_conditions=boundaries,
            timestamp_utc=time.time(),
        )

    def get_semantic_patterns(self, task_family: Optional[str] = None) -> List[SemanticMemoryEntry]:
        """Return all or family-filtered semantic patterns."""
        if task_family:
            return [p for p in self.semantic_store.values() if p.task_family == task_family]
        return list(self.semantic_store.values())

    def get_episodic_records(self, task_family: Optional[str] = None) -> List[ExperienceRecord]:
        """Return all or family-filtered raw episodic records."""
        if task_family:
            return [e for e in self.episodic_store.values() if e.task_family == task_family]
        return list(self.episodic_store.values())
