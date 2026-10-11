"""
Persistent Cognitive Memory Adapter for Federated Cognitive Orchestration (Step 43).

Bridges Step 24/25 Continual Memory Subsystem:
    - ContinualMemoryRetriever
    - EpisodicMemoryStore
    - SemanticMemoryStore
    - ContinualMemoryStorage
into Step 42 Federated Cognitive Orchestration:
    - CognitiveContextEnvelope (context pre-population)
    - CognitiveEpisode (episode completion consolidation)

AXIOMS:
- DATA != AUTHORITY: Retrieved memories provide context, never execution authority.
- MEMORY != AUTHORITY: Memories are advisory evidence.
- TENANT_ISOLATION: Queries, stores, and envelopes strictly enforce tenant boundaries.
- CONTINUAL INTEGRITY: Consolidation uses schema version "24.1" atomic persistence.
"""

from __future__ import annotations

import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional
import uuid

from chakrview.memory.models import (
    Episode,
    SemanticMemory,
    MemoryRetrievalQuery,
    MemoryRetrievalCandidate,
    MemoryRetrievalResult,
    MemoryProvenanceSource,
    MemoryVerificationState,
    MemoryLifecycleStatus,
)
from chakrview.memory.episodic import EpisodicMemoryStore
from chakrview.memory.semantic import SemanticMemoryStore
from chakrview.memory.retrieval import ContinualMemoryRetriever
from chakrview.memory.storage import ContinualMemoryStorage
from chakrview.memory.policy import MemoryExecutionPolicy

from chakrview.cognition.federation.cognitive.models import (
    CognitiveEpisode,
    CognitiveEpisodeState,
    CognitiveContextEnvelope,
)

logger = logging.getLogger("chakrview.federation.cognitive.memory")


class PersistentCognitiveMemoryAdapter:
    """
    Adapter integrating persistent episodic & semantic memory into the federated
    cognitive orchestration layer.

    Capabilities:
    1. Context Retrieval: Queries verified memories for an objective and populates
       CognitiveContextEnvelope with prior relevant knowledge.
    2. Episode Consolidation: Takes a COMMITTED CognitiveEpisode and records it
       into EpisodicMemoryStore and SemanticMemoryStore.
    3. State Persistence: Atomically saves and restores memory snapshots using
       ContinualMemoryStorage (schema "24.1").
    """

    def __init__(
        self,
        episodic_store: Optional[EpisodicMemoryStore] = None,
        semantic_store: Optional[SemanticMemoryStore] = None,
        storage_path: Optional[str] = None,
        policy: Optional[MemoryExecutionPolicy] = None,
    ) -> None:
        self.episodic_store = episodic_store or EpisodicMemoryStore()
        self.semantic_store = semantic_store or SemanticMemoryStore()
        self.policy = policy or MemoryExecutionPolicy.standard()
        self.retriever = ContinualMemoryRetriever(
            episodic_store=self.episodic_store,
            semantic_store=self.semantic_store,
            policy=self.policy,
        )
        self.storage_path = storage_path

        # If storage path exists on disk, load previous state
        if self.storage_path and Path(self.storage_path).exists():
            self.load_from_storage(self.storage_path)

    # ─────────────────────────────────────────────────────────────────────────
    # Context Retrieval
    # ─────────────────────────────────────────────────────────────────────────

    def retrieve_context(
        self,
        objective: str,
        tenant_id: str,
        session_id: Optional[str] = None,
        top_k: int = 5,
        min_confidence: float = 0.4,
        trusted_only: bool = True,
        include_cross_session: bool = True,
    ) -> List[str]:
        """
        Query persistent episodic and semantic stores for information relevant
        to the given objective, scoped strictly to the tenant.

        Returns:
            List of formatted context strings suitable for CognitiveContextEnvelope.
        """
        if not tenant_id:
            raise ValueError("tenant_id is required for memory retrieval.")

        query = MemoryRetrievalQuery(
            query_text=objective,
            tenant_id=tenant_id,
            session_id=session_id,
            top_k=top_k,
            min_confidence=min_confidence,
            trusted_only=trusted_only,
            include_cross_session=include_cross_session,
        )

        result: MemoryRetrievalResult = self.retriever.retrieve(query)
        context_items: List[str] = []

        for candidate in result.candidates:
            # Format clean, bounded context string
            tag = f"[MEMORY:{candidate.memory_type.upper()}]"
            score_tag = f"(rel={candidate.relevance_score:.2f})"
            item = f"{tag} {score_tag} {candidate.content}"
            context_items.append(item)

        logger.debug(
            "Retrieved %d memory context items for tenant '%s' (objective: '%s')",
            len(context_items),
            tenant_id,
            objective[:60],
        )
        return context_items

    # ─────────────────────────────────────────────────────────────────────────
    # Episode Consolidation
    # ─────────────────────────────────────────────────────────────────────────

    def consolidate_episode(
        self,
        episode: CognitiveEpisode,
        extract_semantic: bool = True,
    ) -> Dict[str, Any]:
        """
        Consolidate a completed CognitiveEpisode into persistent memory.

        Records:
        1. An Episode in EpisodicMemoryStore capturing situation, action, outcome,
           and evidence references.
        2. A candidate SemanticMemory proposition if a synthesis conclusion exists.
        3. Atomically syncs state to disk if storage_path is configured.

        Args:
            episode: CognitiveEpisode (must be in COMMITTED state).
            extract_semantic: Whether to extract a semantic proposition.

        Returns:
            Dict containing consolidated episode_id and optional semantic memory_id.
        """
        if not episode.tenant_id:
            raise ValueError("Cannot consolidate episode without tenant_id.")

        # 1. Build Episodic Record
        situation = episode.objective
        if episode.synthesis_result:
            action_or_response = episode.synthesis_result.synthesized_conclusion
            evidence_refs = list(episode.synthesis_result.participating_nodes)
        else:
            action_or_response = episode.error or "Episode execution completed."
            evidence_refs = []

        outcome = episode.state.value

        ep_record = self.episodic_store.record_episode(
            tenant_id=episode.tenant_id,
            session_id=episode.session_id,
            situation=situation,
            action_or_response=action_or_response,
            outcome=outcome,
            task_id=episode.episode_id,
            evidence_refs=evidence_refs,
            confidence=0.85 if episode.state == CognitiveEpisodeState.COMMITTED else 0.5,
            provenance=MemoryProvenanceSource.SYSTEM_OBSERVED,
            verification_status=(
                MemoryVerificationState.UNVERIFIED
                if episode.state == CognitiveEpisodeState.COMMITTED
                else MemoryVerificationState.CANDIDATE
            ),
            lifecycle_status=MemoryLifecycleStatus.ACTIVE,
            tags=["federated_cognitive_episode"],
            metadata={
                "consensus_proposal_id": episode.consensus_proposal_id,
                "has_minority_evidence": (
                    episode.synthesis_result.has_minority_evidence
                    if episode.synthesis_result
                    else False
                ),
            },
        )

        consolidated: Dict[str, Any] = {
            "episode_id": episode.episode_id,
            "recorded_episode_id": ep_record.episode_id,
            "tenant_id": episode.tenant_id,
            "semantic_memory_id": None,
        }

        # 2. Extract Semantic Proposition
        if (
            extract_semantic
            and episode.state == CognitiveEpisodeState.COMMITTED
            and episode.synthesis_result
        ):
            # Safe proposition extraction: subject = objective prefix, predicate = 'concluded', object = conclusion
            subject = episode.objective.strip()[:60]
            predicate = "concluded"
            object_value = episode.synthesis_result.synthesized_conclusion.strip()[:160]

            sem_record = self.semantic_store.add_memory(
                tenant_id=episode.tenant_id,
                subject=subject,
                predicate=predicate,
                object_value=object_value,
                session_id=episode.session_id,
                provenance=MemoryProvenanceSource.REASONING_DERIVED,
                confidence=0.75,
                verification_status=MemoryVerificationState.CANDIDATE,
                lifecycle_status=MemoryLifecycleStatus.ACTIVE,
                tags=["episode_synthesis"],
                metadata={
                    "origin_episode_id": episode.episode_id,
                    "consensus_proposal_id": episode.consensus_proposal_id,
                },
            )
            consolidated["semantic_memory_id"] = sem_record.memory_id

        # 3. Atomic Disk Persistence
        if self.storage_path:
            self.save_to_storage(self.storage_path)

        logger.info(
            "Consolidated episode '%s' into persistent memory (tenant: '%s')",
            episode.episode_id,
            episode.tenant_id,
        )
        return consolidated

    # ─────────────────────────────────────────────────────────────────────────
    # Persistence
    # ─────────────────────────────────────────────────────────────────────────

    def save_to_storage(self, filepath: str) -> None:
        """Atomically persist active episodic and semantic memory to disk."""
        data = ContinualMemoryStorage.export_state(
            episodic_store=self.episodic_store,
            semantic_store=self.semantic_store,
        )
        ContinualMemoryStorage.save_to_file(filepath, data)
        logger.debug("Persisted memory state to '%s'", filepath)

    def load_from_storage(self, filepath: str) -> None:
        """Load and validate memory state from disk into active stores."""
        data = ContinualMemoryStorage.load_from_file(filepath)
        ContinualMemoryStorage.import_state(
            data=data,
            target_episodic_store=self.episodic_store,
            target_semantic_store=self.semantic_store,
        )
        logger.debug("Loaded memory state from '%s'", filepath)
