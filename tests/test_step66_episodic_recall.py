"""
Tests for ChakrView Step 66: Episodic Memory Recall Loop, Structural Relevance & Deterministic Validity Verification.

Validates:
1. Exact task family memory recall
2. Same module memory recall
3. Structural pattern and symptom similarity scoring
4. Irrelevant domain memory rejection
5. Stale memory rejection on structural dependency drift
6. Superseded memory rejection
7. Negative boundary recall as explicit boundary constraint
8. Positive vs negative memory conflict detection
9. Multiple candidate deterministic ranking and ordering
10. Recall budget enforcement
11. Repository fingerprint mismatch handling with provenance
12. Repeated identical recall requests produce bit-exact output
13. Cross-domain negative transfer triggers fail-closed abstention
14. Empty memory index abstains cleanly
15. Neural core immutability (weight hash and parameter count)
"""

from __future__ import annotations

import json
import pytest
import hashlib
import torch

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.runtime.pipeline import EXPECTED_WEIGHT_HASH

from chakrview.cognition.repository.semantic_record import RepositorySemanticRecord
from chakrview.cognition.repository.memory_index import RepositoryMemoryIndex
from chakrview.cognition.repository.state import RepositoryState
from chakrview.cognition.repository.episodic_recall import (
    MemoryRecallStatus,
    MemoryRecallRequest,
    RecallBudget,
    RecalledMemoryItem,
    RecalledContextBundle,
    EpisodicMemoryRecallCoordinator,
)


@pytest.fixture
def clean_memory_index():
    idx = RepositoryMemoryIndex()

    # 1. Tax service positive record
    rec_tax = RepositorySemanticRecord(
        memory_id="mem_tax_service_v1",
        task_family="billing_refactor",
        language="python",
        framework="standard_library",
        repository_pattern="Rate multiplier calculation fix",
        symptom_signature="Tax calculation returning flat fee",
        root_cause_signature="Missing rate multiplier",
        dependency_signature="tax_service -> models",
        affected_modules=["tax_service.py"],
        solution_pattern="def compute_tax(amount, rate): return amount * rate",
        verification_requirements=["test_tax_service.py"],
        known_boundaries=[],
        confidence=0.95,
        evidence_count=3,
        successful_episodes=3,
        failed_episodes=0,
        source_episode_ids=["ep_tax_01"],
    )
    idx.insert(rec_tax)

    # 2. Discount engine positive record
    rec_disc = RepositorySemanticRecord(
        memory_id="mem_discount_engine_v1",
        task_family="billing_refactor",
        language="python",
        framework="standard_library",
        repository_pattern="Tiered discount limit calculation",
        symptom_signature="Discount exceeds order total",
        root_cause_signature="Unbounded discount",
        dependency_signature="discount_engine -> models",
        affected_modules=["discount_engine.py"],
        solution_pattern="def compute_discount(order): return min(order.total, discount)",
        verification_requirements=["test_discount_engine.py"],
        known_boundaries=[],
        confidence=0.90,
        evidence_count=2,
        successful_episodes=2,
        failed_episodes=0,
        source_episode_ids=["ep_disc_01"],
    )
    idx.insert(rec_disc)

    return idx


@pytest.fixture
def repo_states():
    files_v1 = {
        "tax_service.py": "def compute_tax(amount): return 10\n",
        "discount_engine.py": "def compute_discount(order): return 5\n",
        "models.py": "class Order: pass\n",
    }
    state_v1 = RepositoryState.from_files("proj", files_v1, version=1)

    files_v2 = dict(files_v1)
    files_v2["tax_service.py"] = "import sys\ndef compute_tax(amount): return 10\n"
    state_v2 = RepositoryState.from_files("proj", files_v2, version=2)

    return state_v1, state_v2


def test_1_exact_task_family_recall(clean_memory_index, repo_states):
    state_v1, _ = repo_states
    coordinator = EpisodicMemoryRecallCoordinator(clean_memory_index)
    req = MemoryRecallRequest(
        task_description="Refactor tax calculations",
        task_family="billing_refactor",
        target_files=["tax_service.py"],
    )
    bundle = coordinator.recall(req, repo_state=state_v1)
    assert bundle.has_positive_guidance is True
    assert any(m.record.memory_id == "mem_tax_service_v1" for m in bundle.positive_memories)
    assert bundle.positive_memories[0].status == MemoryRecallStatus.RECALLABLE


def test_2_same_module_recall(clean_memory_index, repo_states):
    state_v1, _ = repo_states
    coordinator = EpisodicMemoryRecallCoordinator(clean_memory_index)
    req = MemoryRecallRequest(
        task_description="Update discount limits",
        task_family="pricing_logic",  # Different family, same module
        target_files=["discount_engine.py"],
    )
    bundle = coordinator.recall(req, repo_state=state_v1)
    assert any(m.record.memory_id == "mem_discount_engine_v1" for m in bundle.positive_memories)


def test_3_structural_pattern_similarity(clean_memory_index, repo_states):
    state_v1, _ = repo_states
    coordinator = EpisodicMemoryRecallCoordinator(clean_memory_index)
    req = MemoryRecallRequest(
        task_description="Fix tax calculation",
        task_family="billing_refactor",
        repository_pattern="Rate multiplier calculation fix",
        target_files=["tax_service.py"],
    )
    bundle = coordinator.recall(req, repo_state=state_v1)
    recalled = bundle.positive_memories[0]
    assert recalled.record.memory_id == "mem_tax_service_v1"
    assert recalled.signal_breakdown["pattern_similarity"] > 0.15


def test_4_irrelevant_memory_rejection(clean_memory_index, repo_states):
    state_v1, _ = repo_states
    coordinator = EpisodicMemoryRecallCoordinator(clean_memory_index)
    req = MemoryRecallRequest(
        task_description="Database schema migration",
        task_family="db_migration",
        target_files=["schema.py"],
    )
    bundle = coordinator.recall(req, repo_state=state_v1)
    assert len(bundle.positive_memories) == 0


def test_5_stale_memory_rejection(clean_memory_index, repo_states):
    state_v1, state_v2 = repo_states
    coordinator = EpisodicMemoryRecallCoordinator(clean_memory_index)
    req = MemoryRecallRequest(
        task_description="Refactor tax calculations",
        task_family="billing_refactor",
        target_files=["tax_service.py"],
    )
    bundle = coordinator.recall(req, repo_state=state_v2, reference_state=state_v1)
    stale_items = [it for it in bundle.rejected_items if it.status == MemoryRecallStatus.REJECTED_STALE]
    assert len(stale_items) > 0
    assert stale_items[0].record.memory_id == "mem_tax_service_v1"


def test_6_superseded_memory_rejection(clean_memory_index, repo_states):
    state_v1, _ = repo_states
    old_rec = RepositorySemanticRecord(
        memory_id="mem_tax_v0",
        task_family="billing_refactor",
        language="python",
        framework="standard_library",
        repository_pattern="Old pattern",
        symptom_signature="Old symptom",
        root_cause_signature="Old root",
        dependency_signature="tax_service",
        affected_modules=["tax_service.py"],
        solution_pattern="def solve(): pass",
        verification_requirements=[],
        known_boundaries=[],
        confidence=0.5,
        evidence_count=1,
        successful_episodes=1,
        failed_episodes=0,
        active_version=False,
        superseded_by="mem_tax_service_v1",
    )
    clean_memory_index.insert(old_rec)

    coordinator = EpisodicMemoryRecallCoordinator(clean_memory_index)
    req = MemoryRecallRequest(
        task_description="Refactor tax calculations",
        task_family="billing_refactor",
        target_files=["tax_service.py"],
    )
    bundle = coordinator.recall(req, repo_state=state_v1)
    superseded = [it for it in bundle.rejected_items if it.status == MemoryRecallStatus.REJECTED_SUPERSEDED]
    assert any(it.record.memory_id == "mem_tax_v0" for it in superseded)


def test_7_negative_boundary_recall(clean_memory_index, repo_states):
    state_v1, _ = repo_states
    rec_neg = RepositorySemanticRecord(
        memory_id="sem_neg_lock",
        task_family="billing_refactor",
        language="python",
        framework="standard_library",
        repository_pattern="Known failure: deadlock",
        symptom_signature="Deadlock",
        root_cause_signature="Lock contention",
        dependency_signature="tax_service",
        affected_modules=["tax_service.py"],
        solution_pattern="DO_NOT_APPLY",
        verification_requirements=[],
        known_boundaries=["FAILURE_BOUNDARY (DEADLOCK): Contention detected"],
        confidence=0.99,
        evidence_count=1,
        successful_episodes=0,
        failed_episodes=1,
    )
    clean_memory_index.insert(rec_neg)

    coordinator = EpisodicMemoryRecallCoordinator(clean_memory_index)
    req = MemoryRecallRequest(
        task_description="Fix concurrency in tax service",
        task_family="billing_refactor",
        target_files=["tax_service.py"],
    )
    bundle = coordinator.recall(req, repo_state=state_v1)
    assert bundle.has_negative_boundaries is True
    assert bundle.negative_boundaries[0].record.memory_id == "sem_neg_lock"


def test_8_positive_vs_negative_conflict(clean_memory_index, repo_states):
    state_v1, _ = repo_states
    rec_neg = RepositorySemanticRecord(
        memory_id="sem_neg_lock",
        task_family="billing_refactor",
        language="python",
        framework="standard_library",
        repository_pattern="Known failure: deadlock",
        symptom_signature="Deadlock",
        root_cause_signature="Lock contention",
        dependency_signature="tax_service",
        affected_modules=["tax_service.py"],
        solution_pattern="DO_NOT_APPLY",
        verification_requirements=[],
        known_boundaries=["FAILURE_BOUNDARY (DEADLOCK): Contention detected"],
        confidence=0.99,
        evidence_count=1,
        successful_episodes=0,
        failed_episodes=1,
    )
    clean_memory_index.insert(rec_neg)

    coordinator = EpisodicMemoryRecallCoordinator(clean_memory_index)
    req = MemoryRecallRequest(
        task_description="Fix concurrency in tax service",
        task_family="billing_refactor",
        target_files=["tax_service.py"],
    )
    bundle = coordinator.recall(req, repo_state=state_v1)
    conflicted = [it for it in bundle.rejected_items if it.status == MemoryRecallStatus.CONFLICTED]
    assert len(conflicted) > 0
    assert any(it.record.memory_id == "mem_tax_service_v1" for it in conflicted)


def test_9_multiple_candidate_deterministic_ordering(clean_memory_index, repo_states):
    state_v1, _ = repo_states
    coordinator = EpisodicMemoryRecallCoordinator(clean_memory_index)
    req = MemoryRecallRequest(
        task_description="Fix all billing modules",
        task_family="billing_refactor",
        target_files=["tax_service.py", "discount_engine.py"],
    )
    bundle = coordinator.recall(req, repo_state=state_v1)
    assert len(bundle.positive_memories) == 2
    assert bundle.positive_memories[0].relevance_score >= bundle.positive_memories[1].relevance_score


def test_10_recall_budget_enforcement(clean_memory_index, repo_states):
    state_v1, _ = repo_states
    budget = RecallBudget(max_recalled_memories=1)
    coordinator = EpisodicMemoryRecallCoordinator(clean_memory_index, budget=budget)
    req = MemoryRecallRequest(
        task_description="Fix all billing modules",
        task_family="billing_refactor",
        target_files=["tax_service.py", "discount_engine.py"],
    )
    bundle = coordinator.recall(req, repo_state=state_v1)
    assert len(bundle.positive_memories) == 1
    assert len(bundle.rejected_items) >= 1


def test_11_repository_fingerprint_mismatch_with_provenance(clean_memory_index, repo_states):
    state_v1, _ = repo_states
    coordinator = EpisodicMemoryRecallCoordinator(clean_memory_index)
    req = MemoryRecallRequest(
        task_description="Fix tax calculation",
        task_family="billing_refactor",
        target_files=["tax_service.py"],
        repository_fingerprint="outdated_fp_123",
    )
    bundle = coordinator.recall(req, repo_state=state_v1)
    assert bundle.has_positive_guidance is True
    assert bundle.positive_memories[0].provenance_evidence is not None


def test_12_repeated_run_determinism(clean_memory_index, repo_states):
    state_v1, _ = repo_states
    coordinator = EpisodicMemoryRecallCoordinator(clean_memory_index)
    req = MemoryRecallRequest(
        task_description="Fix tax calculation",
        task_family="billing_refactor",
        target_files=["tax_service.py"],
    )
    b1 = coordinator.recall(req, repo_state=state_v1).to_dict()
    b2 = coordinator.recall(req, repo_state=state_v1).to_dict()
    b1.pop("duration_ms", None)
    b2.pop("duration_ms", None)
    assert json.dumps(b1, sort_keys=True) == json.dumps(b2, sort_keys=True)


def test_13_cross_domain_negative_transfer_defense(clean_memory_index, repo_states):
    state_v1, _ = repo_states
    coordinator = EpisodicMemoryRecallCoordinator(clean_memory_index)
    req = MemoryRecallRequest(
        task_description="OAuth2 authentication setup",
        task_family="auth_oauth2",
        domain_tags={"billing_refactor"},  # Mismatch!
        target_files=["auth.py"],
    )
    bundle = coordinator.recall(req, repo_state=state_v1)
    assert bundle.abstained is True
    assert "Cross-domain negative transfer blocked" in str(bundle.abstain_reason)


def test_14_empty_memory_abstains_cleanly(repo_states):
    state_v1, _ = repo_states
    empty_idx = RepositoryMemoryIndex()
    coordinator = EpisodicMemoryRecallCoordinator(empty_idx)
    req = MemoryRecallRequest(
        task_description="Fix tax calculation",
        task_family="billing_refactor",
        target_files=["tax_service.py"],
    )
    bundle = coordinator.recall(req, repo_state=state_v1)
    assert bundle.has_positive_guidance is False
    assert bundle.total_candidates_evaluated == 0


def test_15_neural_baseline_remains_bit_exact():
    torch.manual_seed(42)
    config = ModelConfig()
    model = ChakrMicro(config)
    model.eval()

    param_count = sum(p.numel() for p in model.parameters())
    assert param_count == 3_443_136, f"Expected 3,443,136 params, got {param_count}"

    hasher = hashlib.sha256()
    with torch.no_grad():
        for name, param in sorted(model.named_parameters()):
            hasher.update(name.encode("utf-8"))
            hasher.update(param.detach().cpu().numpy().tobytes())
    computed_hash = hasher.hexdigest()

    assert computed_hash == EXPECTED_WEIGHT_HASH, f"Weight hash mismatch: {computed_hash}"
