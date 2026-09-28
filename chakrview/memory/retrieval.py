"""
Memory Retrieval Engine for ChakrView Continual Cognition (Step 24).

Implements bounded, deterministic, CPU-first memory retrieval without external
vector database dependencies.

SCORING FORMULA:
    retrieval_score = relevance × confidence × verification_factor × recency_factor × contradiction_factor

CRITICAL PRINCIPLES:
1. DATA != AUTHORITY, MEMORY != AUTHORITY.
   Retrieved memories provide contextual evidence, never sovereign authority.
2. Candidate, Quarantined, and Rejected memories are filtered out from trusted retrieval.
3. Strict tenant isolation: Tenant A queries NEVER return Tenant B memories.
4. Bounded execution: Retrieval strictly respects MemoryExecutionPolicy limits.
"""

import math
import re
import time
from typing import Dict, List, Optional, Any, Set

from chakrview.memory.models import (
    Episode,
    SemanticMemory,
    MemoryRetrievalQuery,
    MemoryRetrievalCandidate,
    MemoryRetrievalResult,
    MemoryVerificationState,
    MemoryLifecycleStatus,
    MemoryProvenanceSource,
)
from chakrview.memory.episodic import EpisodicMemoryStore
from chakrview.memory.semantic import SemanticMemoryStore
from chakrview.memory.policy import MemoryExecutionPolicy


# Verification factor weights
VERIFICATION_FACTORS: Dict[MemoryVerificationState, float] = {
    MemoryVerificationState.VERIFIED: 1.0,
    MemoryVerificationState.UNVERIFIED: 0.6,
    MemoryVerificationState.CONTRADICTED: 0.3,
    MemoryVerificationState.ARCHIVED: 0.2,
    MemoryVerificationState.CANDIDATE: 0.0,    # Blocked from trusted retrieval
    MemoryVerificationState.QUARANTINED: 0.0,  # Blocked
    MemoryVerificationState.REJECTED: 0.0,     # Blocked
}


class ContinualMemoryRetriever:
    """
    Deterministic memory retriever scoring candidates across episodic and semantic stores.
    """

    def __init__(
        self,
        episodic_store: EpisodicMemoryStore,
        semantic_store: SemanticMemoryStore,
        policy: Optional[MemoryExecutionPolicy] = None,
    ) -> None:
        self.episodic_store = episodic_store
        self.semantic_store = semantic_store
        self.policy = policy or MemoryExecutionPolicy.standard()

    @staticmethod
    def _tokenize(text: str) -> Set[str]:
        """Simple deterministic word tokenizer splitting on all non-alphanumeric delimiters."""
        cleaned = re.sub(r"[^a-zA-Z0-9]+", " ", text.lower())
        return {w for w in cleaned.split() if len(w) >= 2}

    def _compute_relevance(self, query_tokens: Set[str], query_raw: str, content: str) -> float:
        """
        Compute lexical relevance score in [0.0, 1.0].
        Note: Clearly designated as heuristic lexical scoring.
        """
        if not query_tokens:
            return 0.0

        content_tokens = self._tokenize(content)
        if not content_tokens:
            return 0.0

        overlap = query_tokens.intersection(content_tokens)
        token_score = len(overlap) / len(query_tokens)

        # Substring bonus
        exact_bonus = 0.2 if query_raw.lower() in content.lower() else 0.0
        return min(1.0, token_score + exact_bonus)

    @staticmethod
    def _compute_recency(timestamp: float, half_life_days: float = 30.0) -> float:
        """
        Exponential recency decay bounded in [0.2, 1.0].
        """
        age_seconds = max(0.0, time.time() - timestamp)
        half_life_seconds = half_life_days * 86400.0
        decay = math.exp(-0.693147 * (age_seconds / half_life_seconds))
        return max(0.2, min(1.0, decay))

    def retrieve(self, query: MemoryRetrievalQuery) -> MemoryRetrievalResult:
        """
        Execute bounded multi-factor retrieval strictly within tenant boundary.
        """
        start_time = time.time()
        query_tokens = self._tokenize(query.query_text)
        candidates: List[MemoryRetrievalCandidate] = []
        scanned_count = 0

        # Enforce policy bounds
        effective_limit = min(query.top_k, self.policy.max_retrieval_candidates)
        scan_limit = self.policy.storage_scan_limit

        # 1. Scan Semantic Memories
        if not query.allowed_types or "semantic" in query.allowed_types:
            semantic_mems = self.semantic_store.list_memories(
                tenant_id=query.tenant_id,
                limit=scan_limit,
                active_only=True,
            )
            for mem in semantic_mems:
                scanned_count += 1
                if scanned_count > scan_limit:
                    break

                # Session filter if requested and not cross-session
                if not query.include_cross_session and query.session_id:
                    if mem.session_id != query.session_id and mem.session_id != "default_session":
                        continue

                # Verification state check
                ver_factor = VERIFICATION_FACTORS.get(mem.verification_status, 0.0)
                if query.trusted_only and mem.verification_status in (
                    MemoryVerificationState.CANDIDATE,
                    MemoryVerificationState.QUARANTINED,
                    MemoryVerificationState.REJECTED,
                ):
                    continue

                if mem.lifecycle_status in (MemoryLifecycleStatus.QUARANTINED, MemoryLifecycleStatus.DELETED):
                    continue

                content = mem.statement
                relevance = self._compute_relevance(query_tokens, query.query_text, content)
                if relevance < 0.1:
                    continue

                recency = self._compute_recency(mem.updated_at)
                contra_factor = 0.5 if mem.contradiction_refs else 1.0
                score = relevance * mem.confidence * ver_factor * recency * contra_factor

                if score >= query.min_score and mem.confidence >= query.min_confidence:
                    candidates.append(
                        MemoryRetrievalCandidate(
                            memory_id=mem.memory_id,
                            memory_type="semantic",
                            content=content,
                            score=score,
                            relevance_score=relevance,
                            confidence=mem.confidence,
                            verification_factor=ver_factor,
                            recency_factor=recency,
                            contradiction_factor=contra_factor,
                            provenance=mem.provenance,
                            verification_status=mem.verification_status,
                            metadata={"version": mem.version, "subject": mem.subject, "predicate": mem.predicate},
                        )
                    )

        # 2. Scan Episodic Memories
        if scanned_count < scan_limit and (not query.allowed_types or "episodic" in query.allowed_types):
            episodes = self.episodic_store.list_episodes(
                tenant_id=query.tenant_id,
                session_id=query.session_id if not query.include_cross_session else None,
                limit=scan_limit - scanned_count,
            )
            for ep in episodes:
                scanned_count += 1
                if scanned_count > scan_limit:
                    break

                ver_factor = VERIFICATION_FACTORS.get(ep.verification_status, 0.0)
                if query.trusted_only and ep.verification_status in (
                    MemoryVerificationState.CANDIDATE,
                    MemoryVerificationState.QUARANTINED,
                    MemoryVerificationState.REJECTED,
                ):
                    continue

                if ep.lifecycle_status in (MemoryLifecycleStatus.QUARANTINED, MemoryLifecycleStatus.DELETED):
                    continue

                content = f"Situation: {ep.situation} | Response: {ep.action_or_response} | Outcome: {ep.outcome}"
                relevance = self._compute_relevance(query_tokens, query.query_text, content)
                if relevance < 0.1:
                    continue

                recency = self._compute_recency(ep.timestamp)
                contra_factor = 1.0
                score = relevance * ep.confidence * ver_factor * recency * contra_factor

                if score >= query.min_score and ep.confidence >= query.min_confidence:
                    candidates.append(
                        MemoryRetrievalCandidate(
                            memory_id=ep.episode_id,
                            memory_type="episodic",
                            content=content,
                            score=score,
                            relevance_score=relevance,
                            confidence=ep.confidence,
                            verification_factor=ver_factor,
                            recency_factor=recency,
                            contradiction_factor=contra_factor,
                            provenance=ep.provenance,
                            verification_status=ep.verification_status,
                            metadata={"task_id": ep.task_id, "evidence_refs": ep.evidence_refs},
                        )
                    )

        # Rank candidates deterministically by score descending
        candidates.sort(key=lambda c: (c.score, c.confidence), reverse=True)
        top_candidates = candidates[:effective_limit]

        latency_ms = (time.time() - start_time) * 1000.0

        return MemoryRetrievalResult(
            query=query.query_text,
            tenant_id=query.tenant_id,
            candidates=top_candidates,
            total_scanned=scanned_count,
            latency_ms=latency_ms,
            budget_used=len(top_candidates),
            execution_profile=self.policy.profile.value,
        )
