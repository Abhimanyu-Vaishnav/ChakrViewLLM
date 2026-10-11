"""
Personal Memory Manager for ChakrView (Step 16).

Central orchestration facade managing:
- Lifecycle: Candidate -> Validate -> Score -> Deduplicate -> Store -> Retrieve -> Consolidate -> Archive
- Multi-tier retrieval (lexical, semantic, recency, importance, confidence)
- Cross-conversation recall queries
- Case/Document comparison
- Controlled learning feedback integration
- Absolute privacy isolation and non-authority boundaries
"""

from typing import Dict, List, Optional, Any, Tuple, Union
import uuid

from chakrview.memory.record import MemoryRecord, MemoryType, MemoryValidity, MemoryProvenance
from chakrview.memory.store import MemoryStore, InMemoryMemoryStore
from chakrview.memory.scoring import MemoryScorer
from chakrview.memory.deduplication import MemoryDeduplicator, DeduplicationResult, MatchLevel
from chakrview.memory.conflict import ConflictDetector, MemoryConflict
from chakrview.memory.consolidation import MemoryConsolidator, ConsolidationCandidate
from chakrview.memory.temporal import TemporalMemoryManager
from chakrview.memory.retriever import PersistentMemoryRetriever, MemoryRetrievalCandidate
from chakrview.memory.comparison import CaseComparator, CaseComparisonResult
from chakrview.memory.learning import LearningFeedbackManager, LearningCategory, FeedbackSignal, LearningCandidate
from chakrview.memory.security import MemorySecurityPolicy, SecurityViolationError


class PersonalMemoryManager:
    """
    Unified manager for personal persistent memory and governed learning.
    """

    def __init__(
        self,
        store: Optional[MemoryStore] = None,
        retriever: Optional[PersistentMemoryRetriever] = None,
        deduplicator: Optional[MemoryDeduplicator] = None,
        consolidator: Optional[MemoryConsolidator] = None,
        learning_manager: Optional[LearningFeedbackManager] = None,
        embedding_provider: Optional[Any] = None,
    ) -> None:
        self.store = store or InMemoryMemoryStore()
        self.embedding_provider = embedding_provider
        self.retriever = retriever or PersistentMemoryRetriever(embedding_provider=self.embedding_provider)
        self.deduplicator = deduplicator or MemoryDeduplicator()
        self.consolidator = consolidator or MemoryConsolidator()
        self.learning_manager = learning_manager or LearningFeedbackManager()

    def store_memory(
        self,
        content: str,
        memory_type: MemoryType,
        owner_id: str,
        provenance: Optional[MemoryProvenance] = None,
        tags: Optional[List[str]] = None,
        check_duplicates: bool = True,
        detect_conflicts: bool = True,
        occurrence_count: int = 1,
    ) -> Tuple[str, Optional[DeduplicationResult], Optional[MemoryConflict]]:
        """
        Store a memory candidate through the full governed lifecycle.
        
        Returns:
            Tuple of (memory_id, deduplication_result, detected_conflict)
        """
        if not content or not content.strip():
            raise ValueError("Memory content cannot be empty.")
        if not owner_id:
            raise ValueError("owner_id cannot be empty.")

        # 1. Sanitize content for privacy
        clean_content = MemorySecurityPolicy.sanitize_content(content.strip())

        # 2. Existing records for owner
        existing_records = self.store.list_records(owner_id=owner_id, limit=500)

        # 3. Deduplication Check
        dedup_result: Optional[DeduplicationResult] = None
        cand_embedding = None
        if self.embedding_provider is not None:
            try:
                emb = self.embedding_provider.embed_query(clean_content)
                cand_embedding = emb.tolist() if hasattr(emb, "tolist") else emb
            except Exception:
                cand_embedding = None

        if check_duplicates and existing_records:
            dedup_result = self.deduplicator.check_duplicate(
                candidate_content=clean_content,
                existing_records=existing_records,
                candidate_embedding=cand_embedding,
            )
            if dedup_result.is_duplicate and dedup_result.match_level in (MatchLevel.EXACT, MatchLevel.NORMALIZED):
                # Return existing record ID rather than duplicating
                return dedup_result.matched_record_id, dedup_result, None

        # 4. Conflict Detection
        conflict: Optional[MemoryConflict] = None
        if detect_conflicts and existing_records:
            conflict = ConflictDetector.detect_conflict(
                candidate_content=clean_content,
                existing_records=existing_records,
            )

        # 5. Dual-Metric Scoring (Importance vs Confidence)
        score_res = MemoryScorer.score_memory(
            content=clean_content,
            memory_type=memory_type,
            provenance=provenance,
            occurrence_count=occurrence_count,
        )

        # 6. Construct MemoryRecord
        memory_id = f"mem_{uuid.uuid4().hex[:8]}"
        record = MemoryRecord(
            memory_id=memory_id,
            memory_type=memory_type,
            content=clean_content,
            owner_id=owner_id,
            importance=score_res.importance,
            confidence=score_res.confidence,
            provenance=provenance or MemoryProvenance(user_id=owner_id),
            tags=tags or [],
            embedding=cand_embedding,
            metadata={"scoring": {
                "importance_reasons": score_res.importance_reasons,
                "confidence_reasons": score_res.confidence_reasons,
            }},
        )

        # 7. Check Data != Authority boundary
        MemorySecurityPolicy.assert_no_tool_authority(record)

        # 8. Persist
        self.store.add(record)
        return memory_id, dedup_result, conflict

    def retrieve(
        self,
        query: str,
        owner_id: str,
        top_k: int = 5,
        memory_type: Optional[MemoryType] = None,
        as_of_timestamp: Optional[float] = None,
    ) -> List[MemoryRetrievalCandidate]:
        """
        Execute weighted hybrid retrieval scoped strictly to owner_id.
        """
        return self.retriever.retrieve(
            query=query,
            owner_id=owner_id,
            store=self.store,
            top_k=top_k,
            memory_type=memory_type,
            as_of_timestamp=as_of_timestamp,
        )

    def recall_cross_conversation(
        self,
        query: str,
        owner_id: str,
        top_k: int = 3,
    ) -> Dict[str, Any]:
        """
        Recall facts or context from prior conversations and sessions.
        Returns honest structured response if no records are found.
        """
        candidates = self.retrieve(query=query, owner_id=owner_id, top_k=top_k)
        relevant_candidates = [
            c for c in candidates
            if c.lexical_score > 0.0 or c.semantic_score > 0.0
        ]
        if not relevant_candidates:
            return {
                "found": False,
                "query": query,
                "owner_id": owner_id,
                "message": "No historical memory matches this query.",
                "memories": [],
            }

        return {
            "found": True,
            "query": query,
            "owner_id": owner_id,
            "count": len(relevant_candidates),
            "memories": [
                {
                    "memory_id": c.memory_id,
                    "content": c.content,
                    "memory_type": c.memory_type.value,
                    "combined_score": c.combined_score,
                    "importance": c.importance,
                    "confidence": c.confidence,
                    "provenance": c.provenance,
                }
                for c in relevant_candidates
            ],
        }

    def consolidate(self, owner_id: str) -> List[ConsolidationCandidate]:
        """
        Cluster related memories for an owner and produce consolidation candidates.
        """
        records = self.store.list_records(owner_id=owner_id, limit=500)
        clusters = self.consolidator.cluster_memories(records)
        candidates: List[ConsolidationCandidate] = []

        for cluster in clusters:
            cand = self.consolidator.generate_candidate(cluster, owner_id)
            if cand:
                candidates.append(cand)
                # Store the consolidated memory
                cons_rec = self.consolidator.create_consolidated_record(cand)
                self.store.add(cons_rec)

        return candidates

    def compare_case(
        self,
        current_text: str,
        historical_memory_id: str,
        owner_id: str,
    ) -> CaseComparisonResult:
        """
        Compare current task/document context against an existing memory record.
        """
        record = self.store.get(historical_memory_id, owner_id=owner_id)
        if record is None:
            raise ValueError(f"Historical memory '{historical_memory_id}' not found for owner '{owner_id}'.")
        return CaseComparator.compare(current_text, record)

    def record_feedback(
        self,
        category: LearningCategory,
        signal: FeedbackSignal,
        observed_outcome: str,
        feedback_text: str,
        task_id: Optional[str] = None,
        proposed_adjustment: str = "",
    ) -> LearningCandidate:
        """
        Record a governed learning candidate signal.
        """
        return self.learning_manager.record_feedback(
            category=category,
            signal=signal,
            observed_outcome=observed_outcome,
            feedback_text=feedback_text,
            task_id=task_id,
            proposed_adjustment=proposed_adjustment,
        )
