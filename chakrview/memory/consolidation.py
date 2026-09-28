"""
Memory Consolidation Engine for ChakrView (Step 16).

Transforms fragmented, multi-session episodic or semantic memories into
coherent, structured long-term knowledge candidates while strictly
preserving source memories and citation provenance.

Workflow:
Raw Fragmented Memories -> Clustering -> Consolidation Candidate -> Policy -> Consolidated Memory
"""

from dataclasses import dataclass, field
import re
import time
from typing import Dict, List, Optional, Any, Set
import uuid

from chakrview.memory.record import MemoryRecord, MemoryType, MemoryProvenance, MemoryValidity


@dataclass
class ConsolidationCandidate:
    """
    Proposed consolidated memory synthesized from multiple source records.
    """
    candidate_id: str
    owner_id: str
    source_memory_ids: List[str]
    consolidated_content: str
    target_type: MemoryType = MemoryType.SEMANTIC
    importance: float = 0.8
    confidence: float = 0.85
    rationale: str = ""
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "owner_id": self.owner_id,
            "source_memory_ids": self.source_memory_ids,
            "consolidated_content": self.consolidated_content,
            "target_type": self.target_type.value,
            "importance": self.importance,
            "confidence": self.confidence,
            "rationale": self.rationale,
            "created_at": self.created_at,
        }


class MemoryConsolidator:
    """
    Deterministic consolidation engine clustering and synthesizing related memories.
    """

    @staticmethod
    def extract_keywords(text: str) -> Set[str]:
        """Extract alphanumeric words with length >= 3."""
        return set(re.findall(r"\b[a-zA-Z0-9]{3,}\b", text.lower()))

    def cluster_memories(
        self,
        records: List[MemoryRecord],
        min_overlap: int = 2,
    ) -> List[List[MemoryRecord]]:
        """
        Group memories that share significant keyword or tag overlap.
        """
        active_records = [r for r in records if r.validity == MemoryValidity.ACTIVE]
        if len(active_records) < 2:
            return []

        clusters: List[List[MemoryRecord]] = []
        assigned: Set[str] = set()

        for i, rec1 in enumerate(active_records):
            if rec1.memory_id in assigned:
                continue

            current_cluster = [rec1]
            words1 = self.extract_keywords(rec1.content)

            for j, rec2 in enumerate(active_records[i + 1:], start=i + 1):
                if rec2.memory_id in assigned:
                    continue

                words2 = self.extract_keywords(rec2.content)
                overlap = words1.intersection(words2)
                # Also check shared tags
                tag_overlap = set(rec1.tags).intersection(set(rec2.tags))

                if len(overlap) >= min_overlap or len(tag_overlap) >= 1:
                    current_cluster.append(rec2)
                    assigned.add(rec2.memory_id)

            if len(current_cluster) >= 2:
                assigned.add(rec1.memory_id)
                clusters.append(current_cluster)

        return clusters

    def generate_candidate(
        self,
        cluster: List[MemoryRecord],
        owner_id: str,
    ) -> Optional[ConsolidationCandidate]:
        """
        Synthesize a coherent consolidation candidate from a cluster of records.
        """
        if len(cluster) < 2:
            return None

        source_ids = [r.memory_id for r in cluster]
        # Cleanly join statements into a coherent synthesized summary
        cleaned_statements = [r.content.strip().rstrip(".") for r in cluster]
        consolidated_text = "; ".join(cleaned_statements) + "."

        # Compute combined importance & confidence
        avg_imp = sum(r.importance for r in cluster) / len(cluster)
        avg_conf = sum(r.confidence for r in cluster) / len(cluster)
        # Consolidation provides mutual corroboration, slightly boosting confidence
        boosted_conf = min(1.0, avg_conf + 0.05)
        boosted_imp = min(1.0, avg_imp + 0.10)

        return ConsolidationCandidate(
            candidate_id=f"cons_{uuid.uuid4().hex[:8]}",
            owner_id=owner_id,
            source_memory_ids=source_ids,
            consolidated_content=consolidated_text,
            target_type=MemoryType.SEMANTIC,
            importance=boosted_imp,
            confidence=boosted_conf,
            rationale=f"Consolidated from {len(cluster)} related records: {source_ids}",
        )

    def create_consolidated_record(
        self,
        candidate: ConsolidationCandidate,
    ) -> MemoryRecord:
        """
        Convert a validated consolidation candidate into a permanent MemoryRecord.
        Preserves source memory IDs in provenance metadata.
        """
        return MemoryRecord(
            memory_id=f"mem_{uuid.uuid4().hex[:8]}",
            memory_type=candidate.target_type,
            content=candidate.consolidated_content,
            owner_id=candidate.owner_id,
            importance=candidate.importance,
            confidence=candidate.confidence,
            provenance=MemoryProvenance(
                source_type="consolidation",
                source_id=candidate.candidate_id,
                user_id=candidate.owner_id,
                metadata={"consolidated_from": candidate.source_memory_ids},
            ),
            tags=["consolidated"],
            metadata={"rationale": candidate.rationale},
        )
