"""
Adapters for Working Memory & Knowledge Coexistence in ChakrView (Step 16).

Bridges:
1. WorkingMemoryAdapter: Promotes volatile multi-turn session working memory into long-term Persistent Memory.
2. KnowledgeMemoryAdapter: Exposes persistent memory to UnifiedRetriever alongside knowledge chunks while keeping provenance distinct.
"""

from typing import Dict, List, Optional, Any
import uuid

from chakrview.memory.record import MemoryRecord, MemoryType, MemoryProvenance
from chakrview.memory.store import MemoryStore
from chakrview.runtime.memory import MemoryItem as RuntimeWorkingMemoryItem, WorkingMemory as RuntimeWorkingMemory
from chakrview.runtime.knowledge import KnowledgeChunk, KnowledgeProvenance
from chakrview.runtime.retrieval import RetrievalCandidate, RetrievalSourceType


class WorkingMemoryAdapter:
    """
    Adapter promoting session working memory items to persistent long-term storage.
    """

    @staticmethod
    def promote_to_persistent(
        item: RuntimeWorkingMemoryItem,
        owner_id: str,
        session_id: Optional[str] = None,
        importance: float = 0.7,
        confidence: float = 0.85,
    ) -> MemoryRecord:
        """
        Convert a session WorkingMemory item into a persistent MemoryRecord.
        """
        return MemoryRecord(
            memory_id=f"mem_{uuid.uuid4().hex[:8]}",
            memory_type=MemoryType.SEMANTIC,
            content=item.content,
            owner_id=owner_id,
            importance=importance,
            confidence=confidence,
            provenance=MemoryProvenance(
                source_type="working_memory_promotion",
                source_id=item.source_turn_id or item.memory_id,
                session_id=session_id,
                user_id=owner_id,
                metadata={"original_type": item.memory_type.value},
            ),
            tags=[item.memory_type.value],
        )

    @classmethod
    def promote_all_from_working_memory(
        cls,
        working_memory: RuntimeWorkingMemory,
        owner_id: str,
        store: MemoryStore,
        session_id: Optional[str] = None,
        min_importance: float = 0.5,
    ) -> List[str]:
        """
        Promote all high-importance working memory items into persistent storage.
        """
        persisted_ids: List[str] = []
        for item in working_memory.get_all():
            if item.importance >= min_importance:
                rec = cls.promote_to_persistent(item, owner_id=owner_id, session_id=session_id)
                mid = store.add(rec)
                persisted_ids.append(mid)
        return persisted_ids


class KnowledgeMemoryAdapter:
    """
    Adapter presenting persistent memory records as retrieval candidates
    compatible with UnifiedRetriever.
    """

    @staticmethod
    def record_to_retrieval_candidate(record: MemoryRecord, score: float = 0.5) -> RetrievalCandidate:
        """
        Map MemoryRecord to a runtime RetrievalCandidate.
        """
        return RetrievalCandidate(
            candidate_id=record.memory_id,
            text=record.content,
            source_type=RetrievalSourceType.MEMORY,
            source_id=record.provenance.source_id or record.memory_id,
            score=score,
            retrieval_method="hybrid",
            metadata={
                "memory_id": record.memory_id,
                "owner_id": record.owner_id,
                "memory_type": record.memory_type.value,
                "importance": record.importance,
                "confidence": record.confidence,
                "source_type": record.provenance.source_type,
            },
            provenance={
                "memory_id": record.memory_id,
                "owner_id": record.owner_id,
                "memory_type": record.memory_type.value,
                "importance": record.importance,
                "confidence": record.confidence,
                "source_type": record.provenance.source_type,
            },
        )
