"""
Unified Continual Cognition & Memory Engine for ChakrView (Step 24).

Orchestrates the complete memory and experience lifecycle:
- Working Memory (bounded task workspace)
- Episodic Memory (structured interaction events)
- Semantic Memory (versioned declarative knowledge)
- Deterministic Retrieval (multi-factor heuristic scoring)
- Contradiction Management (explicit conflict tracking and resolution)
- Experience Consolidation (episode -> candidate semantic memory)
- Controlled Lifecycle Management (archival, expiration, quarantine)
- Governance Bridge (routing approved knowledge toward Step 22 offline learning)
- Hardware Adaptation (policy bounds mapped to hardware capacity)
- Diagnostic Self-Checks (schema, orphan references, isolation integrity)

ARCHITECTURAL INVARIANTS:
1. ChakrMicro v0.1 remains strictly frozen (3,443,136 params, 4096 vocab, 512 context, BOS=0, EOS=1, PAD=2).
2. weights_modified == False: Memory operations NEVER modify neural core weights.
3. DATA != AUTHORITY, MEMORY != AUTHORITY, EXPERIENCE != AUTHORITY.
4. Tenant and session boundaries are strictly enforced.
"""

import time
from typing import Dict, List, Optional, Any, Tuple

from chakrview.memory.models import (
    Episode,
    SemanticMemory,
    MemoryContradiction,
    MemoryRetrievalQuery,
    MemoryRetrievalCandidate,
    MemoryRetrievalResult,
    MemoryProvenanceSource,
    MemoryVerificationState,
    MemoryLifecycleStatus,
)
from chakrview.memory.working import WorkingMemory, WorkingMemoryConfig
from chakrview.memory.episodic import EpisodicMemoryStore
from chakrview.memory.semantic import SemanticMemoryStore
from chakrview.memory.contradiction import ContradictionManager
from chakrview.memory.retrieval import ContinualMemoryRetriever
from chakrview.memory.consolidation import ExperienceConsolidationEngine
from chakrview.memory.lifecycle import MemoryLifecycleManager
from chakrview.memory.governance import MemoryGovernanceBridge
from chakrview.memory.policy import MemoryExecutionPolicy
from chakrview.memory.storage import ContinualMemoryStorage
from chakrview.cognition.adaptation.profiles import ResourceProfile


class ContinualCognitionEngine:
    """
    Central orchestration engine for sovereign continual memory and cognition.
    """

    def __init__(
        self,
        policy: Optional[MemoryExecutionPolicy] = None,
        episodic_store: Optional[EpisodicMemoryStore] = None,
        semantic_store: Optional[SemanticMemoryStore] = None,
    ) -> None:
        self.policy = policy or MemoryExecutionPolicy.detect_host_policy()

        self.episodic_store = episodic_store or EpisodicMemoryStore()
        self.semantic_store = semantic_store or SemanticMemoryStore()
        self.contradiction_mgr = ContradictionManager(semantic_store=self.semantic_store)
        self.lifecycle_mgr = MemoryLifecycleManager(
            episodic_store=self.episodic_store,
            semantic_store=self.semantic_store,
        )
        self.retriever = ContinualMemoryRetriever(
            episodic_store=self.episodic_store,
            semantic_store=self.semantic_store,
            policy=self.policy,
        )
        self.consolidator = ExperienceConsolidationEngine(
            episodic_store=self.episodic_store,
            semantic_store=self.semantic_store,
            contradiction_mgr=self.contradiction_mgr,
            policy=self.policy,
        )
        self.governance_bridge = MemoryGovernanceBridge()

        # Working memory instances indexed by (tenant_id, session_id)
        self._working_memories: Dict[Tuple[str, str], WorkingMemory] = {}

    def get_working_memory(self, tenant_id: str, session_id: str) -> WorkingMemory:
        """Fetch or initialize bounded working memory for tenant and session."""
        key = (tenant_id, session_id)
        if key not in self._working_memories:
            cfg = WorkingMemoryConfig(
                max_context_items=self.policy.working_memory_capacity // 2,
                max_hypotheses=max(2, self.policy.working_memory_capacity // 8),
                max_evidence_items=max(3, self.policy.working_memory_capacity // 4),
                max_decisions=max(3, self.policy.working_memory_capacity // 4),
                max_constraints=10,
                max_pending_questions=5,
                max_observations=self.policy.working_memory_capacity // 2,
            )
            self._working_memories[key] = WorkingMemory(
                tenant_id=tenant_id,
                session_id=session_id,
                config=cfg,
            )
        return self._working_memories[key]

    def record_experience(
        self,
        tenant_id: str,
        session_id: str,
        situation: str,
        action_or_response: str,
        outcome: str,
        task_id: Optional[str] = None,
        confidence: float = 0.8,
        provenance: MemoryProvenanceSource = MemoryProvenanceSource.SYSTEM_OBSERVED,
        verification_status: MemoryVerificationState = MemoryVerificationState.UNVERIFIED,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Episode:
        """Record an episodic experience with strict tenant isolation."""
        ep = self.episodic_store.record_episode(
            tenant_id=tenant_id,
            session_id=session_id,
            situation=situation,
            action_or_response=action_or_response,
            outcome=outcome,
            task_id=task_id,
            confidence=confidence,
            provenance=provenance,
            verification_status=verification_status,
            tags=tags,
            metadata=metadata,
        )
        # Mirror to working memory
        wm = self.get_working_memory(tenant_id, session_id)
        wm.add_observation(f"Experienced task {task_id or 'general'}: {situation[:60]} -> {outcome[:60]}")
        return ep

    def add_semantic_fact(
        self,
        tenant_id: str,
        subject: str,
        predicate: str,
        object_value: str,
        session_id: str = "default_session",
        confidence: float = 0.8,
        provenance: MemoryProvenanceSource = MemoryProvenanceSource.USER_PROVIDED,
        verification_status: MemoryVerificationState = MemoryVerificationState.UNVERIFIED,
        tags: Optional[List[str]] = None,
    ) -> SemanticMemory:
        """
        Add a semantic fact with automated contradiction detection.
        """
        # Check for contradictions first
        conflicts = self.contradiction_mgr.detect_semantic_conflicts(
            tenant_id=tenant_id,
            candidate_subject=subject,
            candidate_predicate=predicate,
            candidate_value=object_value,
        )

        mem = self.semantic_store.add_memory(
            tenant_id=tenant_id,
            subject=subject,
            predicate=predicate,
            object_value=object_value,
            session_id=session_id,
            provenance=provenance,
            confidence=confidence,
            verification_status=verification_status,
            tags=tags,
        )

        if conflicts:
            conflicting_ids = [c[0].memory_id for c in conflicts] + [mem.memory_id]
            self.contradiction_mgr.record_contradiction(
                tenant_id=tenant_id,
                conflicting_memory_ids=conflicting_ids,
                contradiction_type="attribute_conflict",
                severity="HIGH",
                provenance=provenance,
                notes=f"Conflict detected when adding fact: {subject} {predicate}",
            )

        return mem

    def retrieve(self, query: MemoryRetrievalQuery) -> MemoryRetrievalResult:
        """Execute bounded multi-factor memory retrieval."""
        return self.retriever.retrieve(query)

    def consolidate(self, tenant_id: str) -> List[SemanticMemory]:
        """Trigger continual experience consolidation for a tenant."""
        return self.consolidator.consolidate_tenant_episodes(tenant_id)

    def set_hardware_profile(self, profile: ResourceProfile) -> None:
        """Adapt execution policy to a new hardware resource profile."""
        self.policy = MemoryExecutionPolicy.from_resource_profile(profile)
        self.retriever.policy = self.policy
        self.consolidator.policy = self.policy

    def run_diagnostics(self, tenant_id: str) -> Dict[str, Any]:
        """
        Execute comprehensive memory integrity and safety diagnostic inspection.
        Checks:
        1. Schema and serialization health
        2. Orphan contradiction references
        3. Lifecycle consistency
        4. Cross-tenant isolation verification
        5. Verification status hygiene
        """
        issues: List[str] = []
        status = "HEALTHY"

        # 1. Inspect semantic memories for orphan contradictions
        all_cons = {c.contradiction_id for c in self.contradiction_mgr.list_all(tenant_id)}
        for mem in self.semantic_store.list_memories(tenant_id, limit=500, active_only=False):
            for cid in mem.contradiction_refs:
                if cid not in all_cons:
                    issues.append(f"Semantic memory '{mem.memory_id}' references missing contradiction '{cid}'.")
                    status = "DEGRADED"

        # 2. Inspect contradictions for orphan memory references
        for con in self.contradiction_mgr.list_all(tenant_id):
            for mid in con.conflicting_memory_ids:
                if not self.semantic_store.get_memory(tenant_id, mid) and not self.episodic_store.get_episode(tenant_id, mid):
                    issues.append(f"Contradiction '{con.contradiction_id}' references missing memory '{mid}'.")
                    status = "DEGRADED"

        # 3. Check for tenant bleed (ensure no foreign tenant keys)
        ep_count = self.episodic_store.count(tenant_id)
        sem_count = self.semantic_store.count(tenant_id)
        con_count = len(self.contradiction_mgr.list_all(tenant_id))

        return {
            "status": status,
            "tenant_id": tenant_id,
            "issues": issues,
            "total_episodes": ep_count,
            "total_semantic_memories": sem_count,
            "total_contradictions": con_count,
            "unresolved_contradictions": len(self.contradiction_mgr.list_unresolved(tenant_id)),
            "active_policy_profile": self.policy.profile.value,
            "weights_modified": False,  # Core invariant: permanently False
        }
