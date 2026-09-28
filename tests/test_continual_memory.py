"""
Comprehensive Test Suite for ChakrView Step 24:
Memory, Experience & Continual Cognition Foundation.

Verifies:
1.  Working-memory creation
2.  Working-memory bounds & FIFO eviction
3.  Episodic memory creation
4.  Semantic memory creation
5.  Provenance preservation
6.  Verification-state enforcement
7.  Candidate memory rejection from trusted retrieval
8.  Quarantined memory rejection
9.  Deterministic retrieval scoring
10. Recency decay behavior
11. Confidence weighting behavior
12. Contradiction detection
13. Contradiction preservation (no silent overwrite)
14. Contradiction resolution lifecycle
15. Experience consolidation
16. Learning-candidate generation
17. Step 22 learning pipeline integration
18. Runtime weights remain unchanged
19. Frozen model invariants (3,443,136 params, 4096 vocab, 512 context)
20. Tenant isolation
21. Session isolation
22. Corrupted-memory detection
23. Safe memory recovery
24. Hardware LOW_RESOURCE behavior
25. Hardware STANDARD behavior
26. Hardware HIGH_RESOURCE behavior
27. Hard ceiling bounded execution
28. Serialization round-trip
29. Malformed record rejection
30. End-to-end experience -> memory -> retrieval flow
31. End-to-end memory -> learning candidate flow
32. Capability gate cannot be bypassed through memory
"""

import json
import pytest
import time
import torch

from chakrview.brain.model import ChakrMicro
from chakrview.brain.config import ModelConfig
from chakrview.capability.gate import CapabilityGate, CapabilityAuthorizationError
from chakrview.capability.registry import CapabilityRegistry
from chakrview.capability.contract import CapabilityRequest, RiskClassification
from chakrview.cognition.diagnostics.integrity import CoreIntegrityGuard, InvariantViolationError
from chakrview.cognition.adaptation.profiles import ResourceProfile
from chakrview.intelligence.contracts import LearningRecordStatus
from chakrview.memory.models import (
    Episode,
    SemanticMemory,
    MemoryContradiction,
    MemoryProvenanceSource,
    MemoryVerificationState,
    MemoryLifecycleStatus,
    ContradictionResolutionState,
    MemoryRetrievalQuery,
)
from chakrview.memory.working import WorkingMemory, WorkingMemoryConfig
from chakrview.memory.episodic import EpisodicMemoryStore
from chakrview.memory.semantic import SemanticMemoryStore
from chakrview.memory.contradiction import ContradictionManager
from chakrview.memory.retrieval import ContinualMemoryRetriever
from chakrview.memory.consolidation import ExperienceConsolidationEngine
from chakrview.memory.lifecycle import MemoryLifecycleManager
from chakrview.memory.policy import MemoryExecutionPolicy
from chakrview.memory.governance import MemoryGovernanceBridge, MemoryGovernanceError
from chakrview.memory.storage import ContinualMemoryStorage, MemoryStorageSchemaError
from chakrview.memory.engine import ContinualCognitionEngine


@pytest.fixture
def frozen_model() -> ChakrMicro:
    """Fixture providing a deterministic ChakrMicro v0.1 instance."""
    config = ModelConfig(
        vocab_size=4096,
        d_model=192,
        n_layers=6,
        n_heads=6,
        hidden_dim=512,
        max_seq_len=512,
        pad_token_id=2,
    )
    model = ChakrMicro(config)
    model.eval()
    return model


# ============================================================================
# 1. Working Memory Tests
# ============================================================================

def test_01_working_memory_creation():
    """Verify working memory initialization and slot management."""
    wm = WorkingMemory(tenant_id="tenant_alpha", session_id="session_1")
    wm.set_objective("Evaluate Q3 financial balance sheet")
    wm.add_context("Balance sheet retrieved from auditor")
    wm.add_hypothesis({"claim": "Operating profit rose 12%", "status": "candidate"})
    wm.add_evidence({"source": "P&L doc", "metric": 0.12})
    wm.add_decision({"action": "Verify tax liabilities"})
    wm.add_constraint("Do not reveal confidential payroll data")
    wm.add_pending_question("What is the tax rate applied?")
    wm.add_observation("Auditor stamp verified")

    assert wm.objective == "Evaluate Q3 financial balance sheet"
    assert len(wm.active_context) == 1
    assert len(wm.current_hypotheses) == 1
    assert len(wm.relevant_evidence) == 1
    assert len(wm.intermediate_decisions) == 1
    assert len(wm.active_constraints) == 1
    assert len(wm.pending_questions) == 1
    assert len(wm.recent_observations) == 1


def test_02_working_memory_bounds():
    """Verify strict capacity bounding and FIFO eviction."""
    cfg = WorkingMemoryConfig(max_context_items=3, max_hypotheses=2)
    wm = WorkingMemory(tenant_id="tenant_alpha", session_id="session_1", config=cfg)

    for i in range(5):
        wm.add_context(f"Context line {i}")
    for i in range(4):
        wm.add_hypothesis({"id": i, "claim": f"Hypothesis {i}"})

    assert len(wm.active_context) == 3
    assert wm.active_context == ["Context line 2", "Context line 3", "Context line 4"]
    assert len(wm.current_hypotheses) == 2
    assert wm.current_hypotheses[0]["id"] == 2
    assert wm.current_hypotheses[1]["id"] == 3
    assert wm.eviction_count == 4


# ============================================================================
# 2. Episodic Memory Tests
# ============================================================================

def test_03_episodic_memory_creation():
    """Verify structured episodic experience recording."""
    store = EpisodicMemoryStore()
    ep = store.record_episode(
        tenant_id="tenant_alpha",
        session_id="session_1",
        situation="User asked about server uptime",
        action_or_response="Queried status monitor and reported 99.98%",
        outcome="User accepted answer without further clarification",
        task_id="task_server_status",
        confidence=0.95,
        provenance=MemoryProvenanceSource.SYSTEM_OBSERVED,
    )

    assert ep.episode_id.startswith("ep_")
    assert ep.tenant_id == "tenant_alpha"
    assert ep.session_id == "session_1"
    assert ep.situation == "User asked about server uptime"
    assert ep.action_or_response == "Queried status monitor and reported 99.98%"
    assert ep.outcome == "User accepted answer without further clarification"
    assert ep.confidence == 0.95
    assert ep.verification_status == MemoryVerificationState.UNVERIFIED
    assert ep.lifecycle_status == MemoryLifecycleStatus.ACTIVE


def test_04_semantic_memory_creation():
    """Verify versioned semantic proposition creation."""
    store = SemanticMemoryStore()
    mem = store.add_memory(
        tenant_id="tenant_alpha",
        subject="DatabaseCluster",
        predicate="primary_region",
        object_value="ap-south-1",
        provenance=MemoryProvenanceSource.USER_PROVIDED,
        confidence=0.9,
    )

    assert mem.memory_id.startswith("sem_")
    assert mem.subject == "DatabaseCluster"
    assert mem.predicate == "primary_region"
    assert mem.object_value == "ap-south-1"
    assert mem.statement == "DatabaseCluster primary_region ap-south-1"
    assert mem.version == 1
    assert mem.verification_status == MemoryVerificationState.CANDIDATE


# ============================================================================
# 3. Provenance & Verification Tests
# ============================================================================

def test_05_provenance_preservation():
    """Verify explicit provenance source tagging."""
    store = SemanticMemoryStore()
    mem = store.add_memory(
        tenant_id="tenant_alpha",
        subject="Compiler",
        predicate="target_arch",
        object_value="x86_64",
        provenance=MemoryProvenanceSource.CRITICAL_THINKING_DERIVED,
    )
    assert mem.provenance == MemoryProvenanceSource.CRITICAL_THINKING_DERIVED

    d = mem.to_dict()
    assert d["provenance"] == "CRITICAL_THINKING_DERIVED"
    mem_restored = SemanticMemory.from_dict(d)
    assert mem_restored.provenance == MemoryProvenanceSource.CRITICAL_THINKING_DERIVED


def test_06_verification_state_enforcement():
    """Verify verification states and transitions."""
    store = EpisodicMemoryStore()
    ep = store.record_episode(
        tenant_id="tenant_alpha",
        session_id="session_1",
        situation="Test situation",
        action_or_response="Test action",
        outcome="Test outcome",
        verification_status=MemoryVerificationState.UNVERIFIED,
    )

    assert ep.verification_status == MemoryVerificationState.UNVERIFIED
    store.update_verification(ep.tenant_id, ep.episode_id, MemoryVerificationState.VERIFIED)
    assert ep.verification_status == MemoryVerificationState.VERIFIED


def test_07_candidate_memory_rejection_from_trusted_retrieval():
    """Verify that CANDIDATE memories are filtered from trusted retrieval."""
    ep_store = EpisodicMemoryStore()
    sem_store = SemanticMemoryStore()
    sem_store.add_memory(
        tenant_id="tenant_alpha",
        subject="System",
        predicate="status",
        object_value="online",
        verification_status=MemoryVerificationState.CANDIDATE,
    )

    retriever = ContinualMemoryRetriever(ep_store, sem_store)
    query = MemoryRetrievalQuery(
        query_text="System status",
        tenant_id="tenant_alpha",
        trusted_only=True,
    )
    result = retriever.retrieve(query)
    assert len(result.candidates) == 0


def test_08_quarantined_memory_rejection():
    """Verify that QUARANTINED memories are excluded from all retrieval."""
    ep_store = EpisodicMemoryStore()
    sem_store = SemanticMemoryStore()
    sem_store.add_memory(
        tenant_id="tenant_alpha",
        subject="Server",
        predicate="admin_credential",
        object_value="root_password_secret",
        verification_status=MemoryVerificationState.QUARANTINED,
        lifecycle_status=MemoryLifecycleStatus.QUARANTINED,
    )

    retriever = ContinualMemoryRetriever(ep_store, sem_store)
    query = MemoryRetrievalQuery(
        query_text="Server admin_credential",
        tenant_id="tenant_alpha",
        trusted_only=False,
    )
    result = retriever.retrieve(query)
    assert len(result.candidates) == 0


# ============================================================================
# 4. Retrieval Scoring & Ranking Tests
# ============================================================================

def test_09_deterministic_retrieval():
    """Verify deterministic ranking based on lexical relevance."""
    ep_store = EpisodicMemoryStore()
    sem_store = SemanticMemoryStore()

    sem_store.add_memory(
        tenant_id="tenant_alpha",
        subject="Apollo",
        predicate="spacecraft_mission",
        object_value="lunar_landing",
        confidence=0.95,
        verification_status=MemoryVerificationState.VERIFIED,
    )
    sem_store.add_memory(
        tenant_id="tenant_alpha",
        subject="Gemini",
        predicate="orbital_capsule",
        object_value="earth_orbit",
        confidence=0.95,
        verification_status=MemoryVerificationState.VERIFIED,
    )

    retriever = ContinualMemoryRetriever(ep_store, sem_store)
    query = MemoryRetrievalQuery(query_text="Apollo lunar mission", tenant_id="tenant_alpha")
    res = retriever.retrieve(query)

    assert len(res.candidates) >= 1
    assert "Apollo" in res.candidates[0].content
    assert res.candidates[0].score > 0.0


def test_10_recency_behavior():
    """Verify that fresher memories receive higher recency factors."""
    now = time.time()
    sem_store = SemanticMemoryStore()
    mem_old = sem_store.add_memory(
        tenant_id="tenant_alpha",
        subject="Weather",
        predicate="reading",
        object_value="sunny_old",
        confidence=0.9,
        verification_status=MemoryVerificationState.VERIFIED,
    )
    mem_old.updated_at = now - (86400 * 60)  # 60 days old

    mem_new = sem_store.add_memory(
        tenant_id="tenant_alpha",
        subject="Weather",
        predicate="reading",
        object_value="sunny_new",
        confidence=0.9,
        verification_status=MemoryVerificationState.VERIFIED,
    )
    mem_new.updated_at = now

    retriever = ContinualMemoryRetriever(EpisodicMemoryStore(), sem_store)
    query = MemoryRetrievalQuery(query_text="Weather reading", tenant_id="tenant_alpha")
    res = retriever.retrieve(query)

    assert len(res.candidates) == 2
    # Newer memory should rank first due to higher recency factor
    assert "sunny_new" in res.candidates[0].content
    assert res.candidates[0].recency_factor > res.candidates[1].recency_factor


def test_11_confidence_behavior():
    """Verify that higher confidence scales retrieval score linearly."""
    sem_store = SemanticMemoryStore()
    sem_store.add_memory(
        tenant_id="tenant_alpha",
        subject="Spec",
        predicate="diameter",
        object_value="10mm_low_conf",
        confidence=0.4,
        verification_status=MemoryVerificationState.VERIFIED,
    )
    sem_store.add_memory(
        tenant_id="tenant_alpha",
        subject="Spec",
        predicate="diameter",
        object_value="10mm_high_conf",
        confidence=0.95,
        verification_status=MemoryVerificationState.VERIFIED,
    )

    retriever = ContinualMemoryRetriever(EpisodicMemoryStore(), sem_store)
    query = MemoryRetrievalQuery(query_text="Spec diameter", tenant_id="tenant_alpha")
    res = retriever.retrieve(query)

    assert len(res.candidates) == 2
    assert "10mm_high_conf" in res.candidates[0].content
    assert res.candidates[0].score > res.candidates[1].score


# ============================================================================
# 5. Contradiction System Tests
# ============================================================================

def test_12_contradiction_detection():
    """Verify automated detection of conflicting semantic propositions."""
    sem_store = SemanticMemoryStore()
    con_mgr = ContradictionManager(semantic_store=sem_store)

    sem_store.add_memory(
        tenant_id="tenant_alpha",
        subject="CompanyRevenue",
        predicate="annual_amount",
        object_value="20 lakh",
        verification_status=MemoryVerificationState.VERIFIED,
    )

    conflicts = con_mgr.detect_semantic_conflicts(
        tenant_id="tenant_alpha",
        candidate_subject="CompanyRevenue",
        candidate_predicate="annual_amount",
        candidate_value="32 lakh",
    )

    assert len(conflicts) == 1
    assert "20 lakh" in conflicts[0][1]


def test_13_contradiction_preservation():
    """Verify that contradictory records do not delete historical facts."""
    engine = ContinualCognitionEngine()
    mem1 = engine.add_semantic_fact(
        tenant_id="tenant_alpha",
        subject="ProjectBudget",
        predicate="total_usd",
        object_value="50000",
        verification_status=MemoryVerificationState.VERIFIED,
    )
    mem2 = engine.add_semantic_fact(
        tenant_id="tenant_alpha",
        subject="ProjectBudget",
        predicate="total_usd",
        object_value="75000",
        verification_status=MemoryVerificationState.VERIFIED,
    )

    # Both records remain in storage!
    assert engine.semantic_store.get_memory("tenant_alpha", mem1.memory_id) is not None
    assert engine.semantic_store.get_memory("tenant_alpha", mem2.memory_id) is not None
    assert len(engine.contradiction_mgr.list_unresolved("tenant_alpha")) == 1


def test_14_contradiction_resolution_lifecycle():
    """Verify transition from UNRESOLVED to RESOLVED with evidence."""
    engine = ContinualCognitionEngine()
    mem1 = engine.add_semantic_fact(
        tenant_id="tenant_alpha",
        subject="Headquarters",
        predicate="location",
        object_value="Delhi",
    )
    mem2 = engine.add_semantic_fact(
        tenant_id="tenant_alpha",
        subject="Headquarters",
        predicate="location",
        object_value="Bengaluru",
    )

    unresolved = engine.contradiction_mgr.list_unresolved("tenant_alpha")
    assert len(unresolved) == 1
    con = unresolved[0]

    # Resolve with evidence favoring mem2
    success = engine.contradiction_mgr.resolve_contradiction(
        tenant_id="tenant_alpha",
        contradiction_id=con.contradiction_id,
        resolved_memory_id=mem2.memory_id,
        resolution_evidence=["Official board resolution 2026-03"],
        strategy="board_minute_verification",
    )

    assert success is True
    assert con.resolution_state == ContradictionResolutionState.RESOLVED
    assert con.resolved_memory_id == mem2.memory_id
    assert engine.semantic_store.get_memory("tenant_alpha", mem2.memory_id).verification_status == MemoryVerificationState.VERIFIED
    assert engine.semantic_store.get_memory("tenant_alpha", mem1.memory_id).verification_status == MemoryVerificationState.REJECTED


# ============================================================================
# 6. Experience Consolidation Tests
# ============================================================================

def test_15_memory_consolidation():
    """Verify that episodic experiences consolidate into candidate semantic memories."""
    engine = ContinualCognitionEngine()
    engine.record_experience(
        tenant_id="tenant_alpha",
        session_id="session_1",
        situation="User inquired about client fleet",
        action_or_response="Inspected ledger and noted ClientAlpha operates 14 trucks",
        outcome="ClientAlpha operates 14 trucks verified from dispatch",
    )

    candidates = engine.consolidate("tenant_alpha")
    assert len(candidates) >= 1
    cand = candidates[0]
    assert cand.verification_status == MemoryVerificationState.CANDIDATE
    assert "ClientAlpha" in cand.subject or "ClientAlpha" in cand.statement


# ============================================================================
# 7. Step 22 Offline Training Integration & Authority Boundary
# ============================================================================

def test_16_learning_candidate_generation():
    """Verify that verified semantic memories convert to governed LearningRecords."""
    store = SemanticMemoryStore()
    mem = store.add_memory(
        tenant_id="tenant_alpha",
        subject="Oxygen",
        predicate="atomic_number",
        object_value="8",
        confidence=1.0,
        verification_status=MemoryVerificationState.VERIFIED,
    )

    bridge = MemoryGovernanceBridge()
    record = bridge.create_learning_candidate(
        memory=mem,
        target_output="8",
        input_context="What is the atomic number of Oxygen?",
    )

    assert record.owner_id == "tenant_alpha"
    assert record.status == LearningRecordStatus.VERIFIED
    assert "Oxygen" in record.input_context
    assert record.target_output == "8"


def test_17_step22_learning_pipeline_integration():
    """Verify signoff bridge to Step 22 TRAINING_APPROVED status."""
    store = SemanticMemoryStore()
    mem = store.add_memory(
        tenant_id="tenant_alpha",
        subject="Water",
        predicate="boiling_point_celsius",
        object_value="100",
        confidence=1.0,
        verification_status=MemoryVerificationState.VERIFIED,
    )

    bridge = MemoryGovernanceBridge()
    record = bridge.create_learning_candidate(mem, target_output="100")
    assert record.status == LearningRecordStatus.VERIFIED

    # Admin sign-off
    bridge.approve_for_offline_training(record)
    assert record.status == LearningRecordStatus.TRAINING_APPROVED


def test_18_runtime_weights_remain_unchanged(frozen_model):
    """Verify that memory retrieval, insertion, and consolidation never alter model weights."""
    guard = CoreIntegrityGuard()
    initial_fp = guard.compute_weight_fingerprint(frozen_model)

    engine = ContinualCognitionEngine()
    engine.record_experience("t1", "s1", "Situation X", "Action Y", "Outcome Z")
    engine.add_semantic_fact("t1", "Fact A", "equals", "Val B")
    engine.retrieve(MemoryRetrievalQuery(query_text="Fact A", tenant_id="t1"))
    engine.consolidate("t1")

    # Run dummy model forward pass
    input_ids = torch.tensor([[0, 42, 84, 1]], dtype=torch.long)
    with torch.no_grad():
        frozen_model(input_ids)

    post_fp = guard.compute_weight_fingerprint(frozen_model)
    assert initial_fp == post_fp


def test_19_frozen_model_invariants(frozen_model):
    """Verify parameters, vocab, context, and special tokens remain inviolate."""
    guard = CoreIntegrityGuard()
    res = guard.verify_model(frozen_model)
    assert res.passed is True
    assert res.parameter_count == 3_443_136
    assert res.vocab_size == 4_096
    assert res.max_seq_len == 512
    assert res.bos_id == 0
    assert res.eos_id == 1
    assert res.pad_id == 2


# ============================================================================
# 8. Tenant & Session Isolation Tests
# ============================================================================

def test_20_tenant_isolation():
    """Verify that Tenant Alpha memories can NEVER be retrieved by Tenant Beta."""
    engine = ContinualCognitionEngine()
    engine.add_semantic_fact(
        tenant_id="tenant_alpha",
        subject="ProjectChimera",
        predicate="secret_key",
        object_value="alpha_confidential_987",
        confidence=1.0,
        verification_status=MemoryVerificationState.VERIFIED,
    )

    query_beta = MemoryRetrievalQuery(
        query_text="ProjectChimera secret key",
        tenant_id="tenant_beta",
    )
    result_beta = engine.retrieve(query_beta)
    assert len(result_beta.candidates) == 0

    query_alpha = MemoryRetrievalQuery(
        query_text="ProjectChimera secret key",
        tenant_id="tenant_alpha",
    )
    result_alpha = engine.retrieve(query_alpha)
    assert len(result_alpha.candidates) == 1
    assert "alpha_confidential_987" in result_alpha.candidates[0].content


def test_21_session_isolation():
    """Verify session isolation when cross-session recall is disabled."""
    engine = ContinualCognitionEngine()
    engine.record_experience(
        tenant_id="tenant_alpha",
        session_id="session_A",
        situation="Debug memory leak in worker 4",
        action_or_response="Adjusted max buffer size",
        outcome="Leak resolved in worker 4",
        confidence=1.0,
        verification_status=MemoryVerificationState.VERIFIED,
    )

    query = MemoryRetrievalQuery(
        query_text="memory leak worker 4",
        tenant_id="tenant_alpha",
        session_id="session_B",
        include_cross_session=False,
    )
    result = engine.retrieve(query)
    assert len(result.candidates) == 0

    # Cross-session explicitly enabled
    query_cross = MemoryRetrievalQuery(
        query_text="memory leak worker 4",
        tenant_id="tenant_alpha",
        session_id="session_B",
        include_cross_session=True,
    )
    result_cross = engine.retrieve(query_cross)
    assert len(result_cross.candidates) == 1


# ============================================================================
# 9. Diagnostics, Recovery & Serialization Tests
# ============================================================================

def test_22_corrupted_memory_detection():
    """Verify detection of malformed memory payloads and schema mismatches."""
    with pytest.raises(MemoryStorageSchemaError):
        ContinualMemoryStorage.validate_schema({"schema_version": "99.0"})

    with pytest.raises(MemoryStorageSchemaError):
        ContinualMemoryStorage.validate_schema({"schema_version": "24.1", "episodic": "not_a_dict"})


def test_23_safe_memory_recovery():
    """Verify graceful handling of corrupted records during import."""
    malformed_data = {
        "schema_version": "24.1",
        "exported_at": time.time(),
        "episodic": {
            "tenant_alpha": {
                "ep_corrupt": {"invalid_field": 123}
            }
        },
        "semantic": {},
        "contradictions": {},
    }

    with pytest.raises(MemoryStorageSchemaError) as exc_info:
        ContinualMemoryStorage.import_state(malformed_data)
    assert "Corrupted episode" in str(exc_info.value)


def test_24_hardware_low_resource_behavior():
    """Verify bounded budgets under LOW_RESOURCE policy."""
    policy = MemoryExecutionPolicy.low_resource()
    assert policy.profile == ResourceProfile.LOW_RESOURCE
    assert policy.max_retrieval_candidates == 3
    assert policy.working_memory_capacity == 10
    assert policy.storage_scan_limit == 50

    engine = ContinualCognitionEngine(policy=policy)
    for i in range(10):
        engine.add_semantic_fact(
            tenant_id="tenant_alpha",
            subject=f"Item_{i}",
            predicate="code",
            object_value=f"val_{i}",
            verification_status=MemoryVerificationState.VERIFIED,
        )

    res = engine.retrieve(MemoryRetrievalQuery(query_text="Item code", tenant_id="tenant_alpha", top_k=20))
    assert len(res.candidates) <= 3
    assert res.total_scanned <= 50


def test_25_hardware_standard_behavior():
    """Verify balanced budgets under STANDARD policy."""
    policy = MemoryExecutionPolicy.standard()
    assert policy.profile == ResourceProfile.STANDARD
    assert policy.max_retrieval_candidates == 8
    assert policy.working_memory_capacity == 30


def test_26_hardware_high_resource_behavior():
    """Verify expanded budgets under HIGH_RESOURCE policy."""
    policy = MemoryExecutionPolicy.high_resource()
    assert policy.profile == ResourceProfile.HIGH_RESOURCE
    assert policy.max_retrieval_candidates == 15
    assert policy.working_memory_capacity == 60


def test_27_bounded_execution():
    """Verify hard ceilings prevent unbounded allocation even on extreme inputs."""
    policy = MemoryExecutionPolicy(
        profile=ResourceProfile.HIGH_RESOURCE,
        working_memory_capacity=99999,
        max_retrieval_candidates=5000,
        storage_scan_limit=99999,
        contradiction_check_depth=999,
        consolidation_batch_size=999,
    )
    assert policy.working_memory_capacity == 100
    assert policy.max_retrieval_candidates == 30
    assert policy.storage_scan_limit == 1000
    assert policy.contradiction_check_depth == 50
    assert policy.consolidation_batch_size == 20


def test_28_serialization_round_trip():
    """Verify export and import preserve data fidelity."""
    engine = ContinualCognitionEngine()
    engine.record_experience("t1", "s1", "Situation 1", "Action 1", "Outcome 1")
    engine.add_semantic_fact("t1", "Subj1", "is", "Obj1")

    state = ContinualMemoryStorage.export_state(
        episodic_store=engine.episodic_store,
        semantic_store=engine.semantic_store,
        contradiction_mgr=engine.contradiction_mgr,
    )
    assert state["schema_version"] == "24.1"

    imported = ContinualMemoryStorage.import_state(state)
    assert imported["episodic_store"].count("t1") == 1
    assert imported["semantic_store"].count("t1") == 1


def test_29_malformed_record_rejection():
    """Verify that unverified or rejected memories cannot become learning candidates."""
    store = SemanticMemoryStore()
    mem = store.add_memory(
        tenant_id="tenant_alpha",
        subject="Fake",
        predicate="stat",
        object_value="val",
        verification_status=MemoryVerificationState.CANDIDATE,
    )

    with pytest.raises(MemoryGovernanceError):
        MemoryGovernanceBridge.create_learning_candidate(mem, target_output="val")


def test_30_end_to_end_experience_to_memory_to_retrieval():
    """Verify full workflow: Experience -> Episode -> Semantic Fact -> Retrieval."""
    engine = ContinualCognitionEngine()
    engine.record_experience(
        tenant_id="tenant_100",
        session_id="sess_100",
        situation="Auditing security protocol",
        action_or_response="Applied cipher AES-GCM-256",
        outcome="Cipher AES-GCM-256 enabled on port 8443",
        confidence=0.9,
    )

    # Add verified fact
    engine.add_semantic_fact(
        tenant_id="tenant_100",
        subject="ClusterPort8443",
        predicate="cipher_suite",
        object_value="AES-GCM-256",
        confidence=1.0,
        verification_status=MemoryVerificationState.VERIFIED,
    )

    res = engine.retrieve(
        MemoryRetrievalQuery(query_text="ClusterPort8443 cipher suite", tenant_id="tenant_100")
    )
    assert len(res.candidates) >= 1
    assert "AES-GCM-256" in res.candidates[0].content


def test_31_end_to_end_memory_to_learning_candidate():
    """Verify verified semantic memory -> learning candidate -> approval workflow."""
    engine = ContinualCognitionEngine()
    mem = engine.add_semantic_fact(
        tenant_id="tenant_alpha",
        subject="LightSpeed",
        predicate="vacuum_mps",
        object_value="299792458",
        confidence=1.0,
        verification_status=MemoryVerificationState.VERIFIED,
    )

    candidate = engine.governance_bridge.create_learning_candidate(
        memory=mem,
        target_output="299,792,458 m/s",
        input_context="What is the speed of light in vacuum?",
    )
    assert candidate.status == LearningRecordStatus.VERIFIED

    engine.governance_bridge.approve_for_offline_training(candidate)
    assert candidate.status == LearningRecordStatus.TRAINING_APPROVED


def test_32_capability_gate_cannot_be_bypassed_through_memory():
    """Verify that memory provenance can never authorize capability invocation."""
    gate = CapabilityGate(registry=CapabilityRegistry())
    req = CapabilityRequest(
        capability_id="device_read",
        parameters={},
        context={"provenance_source": "continual_memory"},
    )

    with pytest.raises(CapabilityAuthorizationError) as exc_info:
        gate.authorize(req)
    assert "Authority denial" in str(exc_info.value)
