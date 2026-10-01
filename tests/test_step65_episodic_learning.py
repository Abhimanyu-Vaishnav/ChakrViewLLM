"""
Tests for ChakrView Step 65: Active Episodic Learning Loop & Verified Semantic Memory Admission.

Covers:
1. Successful verified episode becomes learnable (ADMITTED_POSITIVE).
2. Unverified proposal cannot become positive memory (REJECTED).
3. Hallucinated proposal cannot become positive memory (REJECTED).
4. Failed execution does not become positive learning (REJECTED).
5. Verified negative experience recorded safely as failure boundary constraint (ADMITTED_NEGATIVE).
6. Stale memory is detected and rejected/abstained by deterministic safety gate.
7. Superseding a previous memory is deterministic and preserves history.
8. Full provenance is preserved from source episode to admitted record.
9. Repeated identical learning input produces bit-exact identical memory admissions.
10. Neural baseline remains bit-exact (hash and parameter count).
"""

from __future__ import annotations

import pytest
import hashlib
import torch

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.runtime.pipeline import EXPECTED_WEIGHT_HASH

from chakrview.cognition.repository.semantic_record import RepositorySemanticRecord
from chakrview.cognition.repository.memory_index import RepositoryMemoryIndex
from chakrview.cognition.repository.context_store import EpistemicState
from chakrview.cognition.repository.neural_adapter import (
    NeuralProposalOutput,
    GroundingVerificationDecision,
)
from chakrview.cognition.repository.refactoring import MultiStepRefactoringResult
from chakrview.cognition.repository.branching import BranchStatus
from chakrview.cognition.repository.episodic_learning import (
    EpisodeOutcome,
    FailureClass,
    AdmissionStatus,
    LearnedObservation,
    EpisodicExperience,
    AdmissionDecision,
    DeterministicAdmissionGate,
    EpisodicLearningCoordinator,
)
from chakrview.cognition.repository.synthesis import (
    CandidateOrigin,
    CandidateProvenance,
    SynthesizedCandidate,
    DeterministicSafetyGate,
    SafetyDecision,
)
from chakrview.cognition.repository.state import RepositoryState


def _make_dummy_execution_result(success: bool = True) -> MultiStepRefactoringResult:
    return MultiStepRefactoringResult(
        task_id="test_task",
        overall_success=success,
        steps_executed=1 if success else 0,
        total_steps=1,
        final_state_fingerprint="fp_final_test" if success else "fp_init_test",
    )


def test_1_successful_verified_episode_becomes_learnable():
    """1. Successful verified episode becomes positive semantic memory."""
    mem_idx = RepositoryMemoryIndex()
    coordinator = EpisodicLearningCoordinator(mem_idx)

    proposal = NeuralProposalOutput(
        proposal_id="prop_01",
        objective="Fix order total calculation",
        target_files=["order_service.py"],
        target_symbols=["calc_total"],
        proposed_patches={"order_service.py": "def calc_total(): return 100"},
        confidence=0.96,
    )
    episode = EpisodicExperience(
        episode_id="ep_succ_01",
        task_description="Fix order total calculation",
        task_family="order_processing",
        initial_state_fingerprint="fp_init_01",
        final_state_fingerprint="fp_final_01",
        neural_proposal=proposal,
        grounding_decision=GroundingVerificationDecision(is_grounded=True, epistemic_state=EpistemicState.KNOWN),
        execution_result=_make_dummy_execution_result(success=True),
        outcome=EpisodeOutcome.SUCCESS_VERIFIED,
    )

    decisions = coordinator.process_episode(episode)
    assert len(decisions) == 1
    assert decisions[0].status == AdmissionStatus.ADMITTED_POSITIVE
    assert decisions[0].admitted_memory_id is not None

    admitted_rec = mem_idx.lookup(decisions[0].admitted_memory_id)
    assert admitted_rec is not None
    assert admitted_rec.task_family == "order_processing"
    assert admitted_rec.successful_episodes == 1
    assert admitted_rec.failed_episodes == 0
    assert admitted_rec.source_episode_ids == ["ep_succ_01"]


def test_2_unverified_proposal_cannot_become_memory():
    """2. Unverified proposal cannot become memory."""
    mem_idx = RepositoryMemoryIndex()
    coordinator = EpisodicLearningCoordinator(mem_idx)

    obs_unverified = LearnedObservation(
        observation_id="obs_unverified_01",
        observation_type="SOLUTION_PATTERN",
        task_family="order_processing",
        target_files=["order_service.py"],
        lesson_summary="Unverified claim without test pass",
        is_positive=True,
        evidence_fingerprint="fp_unverified",
        confidence=0.9,
        supporting_episode_id="ep_unverified_01",
        verification_passed=False,  # NOT verified!
        failure_class=FailureClass.UNVERIFIED_CLAIM,
    )
    episode = EpisodicExperience(
        episode_id="ep_unverified_01",
        task_description="Unverified proposal",
        task_family="order_processing",
        initial_state_fingerprint="fp_init",
        final_state_fingerprint="fp_init",
        outcome=EpisodeOutcome.UNVERIFIED_ABORT,
        derived_observations=[obs_unverified],
    )

    decisions = coordinator.process_episode(episode)
    assert len(decisions) == 1
    assert decisions[0].status == AdmissionStatus.REJECTED
    assert "Positive observation lacks verified test execution" in decisions[0].rationale
    assert mem_idx.count() == 0


def test_3_hallucinated_proposal_cannot_become_memory():
    """3. Hallucinated proposal cannot become positive memory."""
    mem_idx = RepositoryMemoryIndex()
    coordinator = EpisodicLearningCoordinator(mem_idx)

    obs_halluc = LearnedObservation(
        observation_id="obs_halluc_01",
        observation_type="SOLUTION_PATTERN",
        task_family="order_processing",
        target_files=["non_existent_service.py"],
        lesson_summary="Hallucinated file patch",
        is_positive=True,  # Attempting positive admission of hallucination!
        evidence_fingerprint="fp_base",
        confidence=0.5,
        supporting_episode_id="ep_halluc_01",
        verification_passed=False,
    )
    episode = EpisodicExperience(
        episode_id="ep_halluc_01",
        task_description="Hallucinated repair",
        task_family="order_processing",
        initial_state_fingerprint="fp_base",
        final_state_fingerprint="fp_base",
        grounding_decision=GroundingVerificationDecision(
            is_grounded=False,
            epistemic_state=EpistemicState.CONTRADICTED,
            unfounded_files=["non_existent_service.py"],
        ),
        outcome=EpisodeOutcome.PROPOSAL_HALLUCINATED,
        derived_observations=[obs_halluc],
    )

    decisions = coordinator.process_episode(episode)
    assert len(decisions) == 1
    assert decisions[0].status == AdmissionStatus.REJECTED
    assert mem_idx.count() == 0


def test_4_failed_execution_does_not_become_positive_learning():
    """4. Failed execution does not become positive learning."""
    mem_idx = RepositoryMemoryIndex()
    coordinator = EpisodicLearningCoordinator(mem_idx)

    obs_failed = LearnedObservation(
        observation_id="obs_fail_01",
        observation_type="SOLUTION_PATTERN",
        task_family="order_processing",
        target_files=["order_service.py"],
        lesson_summary="Buggy patch that failed tests",
        is_positive=True,  # Attempting positive admission
        evidence_fingerprint="fp_failed",
        confidence=0.6,
        supporting_episode_id="ep_fail_01",
        verification_passed=False,  # Tests failed!
    )
    episode = EpisodicExperience(
        episode_id="ep_fail_01",
        task_description="Failed patch",
        task_family="order_processing",
        initial_state_fingerprint="fp_base",
        final_state_fingerprint="fp_base",
        execution_result=_make_dummy_execution_result(success=False),
        outcome=EpisodeOutcome.EXECUTION_FAILED_ROLLED_BACK,
        derived_observations=[obs_failed],
    )

    decisions = coordinator.process_episode(episode)
    assert len(decisions) == 1
    assert decisions[0].status == AdmissionStatus.REJECTED
    assert mem_idx.count() == 0


def test_5_verified_negative_experience_recorded_safely():
    """5. Verified negative experience can be recorded safely as boundary constraint."""
    mem_idx = RepositoryMemoryIndex()
    coordinator = EpisodicLearningCoordinator(mem_idx)

    obs_neg = LearnedObservation(
        observation_id="obs_neg_01",
        observation_type="NEGATIVE_BOUNDARY",
        task_family="order_processing",
        target_files=["order_service.py"],
        lesson_summary="Deadlock under concurrency",
        is_positive=False,
        evidence_fingerprint="fp_fail_trace",
        confidence=0.95,
        supporting_episode_id="ep_neg_01",
        verification_passed=False,
        failure_class=FailureClass.TEST_FAILURE,
    )
    episode = EpisodicExperience(
        episode_id="ep_neg_01",
        task_description="Concurrency fix attempt",
        task_family="order_processing",
        initial_state_fingerprint="fp_base",
        final_state_fingerprint="fp_base",
        execution_result=_make_dummy_execution_result(success=False),
        outcome=EpisodeOutcome.EXECUTION_FAILED_ROLLED_BACK,
        derived_observations=[obs_neg],
    )

    decisions = coordinator.process_episode(episode)
    assert len(decisions) == 1
    assert decisions[0].status == AdmissionStatus.ADMITTED_NEGATIVE
    admitted_rec = mem_idx.lookup(decisions[0].admitted_memory_id)
    assert admitted_rec is not None
    assert admitted_rec.failed_episodes == 1
    assert admitted_rec.successful_episodes == 0
    assert admitted_rec.solution_pattern == "DO_NOT_APPLY"
    assert any("FAILURE_BOUNDARY" in b for b in admitted_rec.known_boundaries)


def test_6_stale_memory_detected_and_abstained():
    """6. Stale memory is detected when repository structure mutates."""
    mem_idx = RepositoryMemoryIndex()
    stale_rec = RepositorySemanticRecord(
        memory_id="rec_stale_01",
        task_family="order_processing",
        language="python",
        framework="standard_library",
        repository_pattern="Old pattern",
        symptom_signature="Signature",
        root_cause_signature="Root cause",
        dependency_signature="order -> inventory",
        affected_modules=["order_service.py"],
        solution_pattern="def solve(): pass",
        verification_requirements=["test_order.py"],
        known_boundaries=[],
        confidence=0.8,
        evidence_count=1,
        successful_episodes=1,
        failed_episodes=0,
    )
    mem_idx.insert(stale_rec)

    repo_files_v1 = {"order_service.py": "def calc(): return 1\n"}
    repo_files_v2 = {"order_service.py": "import sys\ndef calc(): return 1\n"}

    state_v1 = RepositoryState.from_files("proj", repo_files_v1, version=1)
    state_v2 = RepositoryState.from_files("proj", repo_files_v2, version=2)

    cand = SynthesizedCandidate(
        candidate_id="cand_test_stale",
        objective="Execute stale pattern",
        ordered_steps=[],
        allowed_files={"order_service.py"},
        provenance=CandidateProvenance(
            origin=CandidateOrigin.MEMORY_RETRIEVAL,
            source_memory_ids=["rec_stale_01"],
            domain_tags={"order_processing"},
        ),
    )

    safety_gate = DeterministicSafetyGate()
    decision = safety_gate.evaluate(
        candidate=cand,
        repo_state=state_v1,
        allowed_modified_files={"order_service.py"},
        task_domain="order_processing",
        memory_index=mem_idx,
        reference_state=state_v2,
    )

    assert decision.decision == SafetyDecision.ABSTAIN
    assert any("STALE" in r for r in decision.reasons)


def test_7_superseding_previous_memory_is_deterministic():
    """7. Superseding a previous memory is deterministic and preserves history."""
    mem_idx = RepositoryMemoryIndex()
    coordinator = EpisodicLearningCoordinator(mem_idx)

    # Episode 1: Initial solution
    prop_v1 = NeuralProposalOutput("p1", "v1", ["order_service.py"], ["calc"], {"order_service.py": "v1"}, 0.9)
    ep1 = EpisodicExperience(
        episode_id="ep_01",
        task_description="v1 fix",
        task_family="order_processing",
        initial_state_fingerprint="fp1",
        final_state_fingerprint="fp2",
        neural_proposal=prop_v1,
        grounding_decision=GroundingVerificationDecision(True, EpistemicState.KNOWN),
        execution_result=_make_dummy_execution_result(True),
        outcome=EpisodeOutcome.SUCCESS_VERIFIED,
    )
    dec1 = coordinator.process_episode(ep1)
    assert dec1[0].status == AdmissionStatus.ADMITTED_POSITIVE
    rec1_id = dec1[0].admitted_memory_id

    rec1 = mem_idx.lookup(rec1_id)
    assert rec1.active_version is True
    assert rec1.supersedes is None

    # Episode 2: Newer solution covering identical target_files
    prop_v2 = NeuralProposalOutput("p2", "v2", ["order_service.py"], ["calc"], {"order_service.py": "v2"}, 0.95)
    ep2 = EpisodicExperience(
        episode_id="ep_02",
        task_description="v2 optimized fix",
        task_family="order_processing",
        initial_state_fingerprint="fp1",
        final_state_fingerprint="fp3",
        neural_proposal=prop_v2,
        grounding_decision=GroundingVerificationDecision(True, EpistemicState.KNOWN),
        execution_result=_make_dummy_execution_result(True),
        outcome=EpisodeOutcome.SUCCESS_VERIFIED,
    )
    dec2 = coordinator.process_episode(ep2)
    assert dec2[0].status == AdmissionStatus.ADMITTED_POSITIVE
    assert dec2[0].superseded_memory_id == rec1_id
    rec2_id = dec2[0].admitted_memory_id

    # Verify history is preserved without deletion
    old_rec = mem_idx.lookup(rec1_id)
    assert old_rec is not None
    assert old_rec.active_version is False
    assert old_rec.superseded_by == rec2_id

    new_rec = mem_idx.lookup(rec2_id)
    assert new_rec is not None
    assert new_rec.active_version is True
    assert new_rec.supersedes == rec1_id


def test_8_provenance_is_preserved():
    """8. Provenance is preserved from episode to admitted record."""
    mem_idx = RepositoryMemoryIndex()
    coordinator = EpisodicLearningCoordinator(mem_idx)

    proposal = NeuralProposalOutput("prop_prov", "objective", ["mod.py"], ["sym"], {"mod.py": "code"}, 0.92)
    episode = EpisodicExperience(
        episode_id="ep_provenance_test",
        task_description="Provenance preservation test",
        task_family="prov_family",
        initial_state_fingerprint="fp_prov_init",
        final_state_fingerprint="fp_prov_final",
        neural_proposal=proposal,
        grounding_decision=GroundingVerificationDecision(True, EpistemicState.KNOWN),
        execution_result=_make_dummy_execution_result(True),
        outcome=EpisodeOutcome.SUCCESS_VERIFIED,
    )

    decisions = coordinator.process_episode(episode)
    rec = mem_idx.lookup(decisions[0].admitted_memory_id)
    assert rec is not None
    assert "ep_provenance_test" in rec.source_episode_ids
    assert "Fingerprint: fp_prov_final" in rec.dependency_signature


def test_9_repeated_identical_learning_input_produces_identical_state():
    """9. Repeated identical learning inputs produce bit-exact identical memory admissions."""
    mem_idx_1 = RepositoryMemoryIndex()
    coord_1 = EpisodicLearningCoordinator(mem_idx_1)

    mem_idx_2 = RepositoryMemoryIndex()
    coord_2 = EpisodicLearningCoordinator(mem_idx_2)

    proposal = NeuralProposalOutput("prop_det", "det", ["m.py"], ["s"], {"m.py": "c"}, 0.95)
    episode_1 = EpisodicExperience(
        episode_id="ep_det",
        task_description="Deterministic test",
        task_family="det_family",
        initial_state_fingerprint="fp1",
        final_state_fingerprint="fp2",
        neural_proposal=proposal,
        grounding_decision=GroundingVerificationDecision(True, EpistemicState.KNOWN),
        execution_result=_make_dummy_execution_result(True),
        outcome=EpisodeOutcome.SUCCESS_VERIFIED,
    )
    episode_2 = EpisodicExperience(
        episode_id="ep_det",
        task_description="Deterministic test",
        task_family="det_family",
        initial_state_fingerprint="fp1",
        final_state_fingerprint="fp2",
        neural_proposal=proposal,
        grounding_decision=GroundingVerificationDecision(True, EpistemicState.KNOWN),
        execution_result=_make_dummy_execution_result(True),
        outcome=EpisodeOutcome.SUCCESS_VERIFIED,
    )

    dec1 = coord_1.process_episode(episode_1)
    dec2 = coord_2.process_episode(episode_2)

    assert dec1[0].status == dec2[0].status
    assert dec1[0].admitted_memory_id == dec2[0].admitted_memory_id
    assert mem_idx_1.to_json() == mem_idx_2.to_json()


def test_10_neural_baseline_remains_bit_exact():
    """10. Neural baseline parameters and weight hash remain strictly frozen."""
    torch.manual_seed(42)
    config = ModelConfig()
    model = ChakrMicro(config)
    model.eval()

    # 1. Parameter count check
    param_count = sum(p.numel() for p in model.parameters())
    assert param_count == 3_443_136, f"Expected 3,443,136 params, got {param_count}"

    # 2. Weight hash check
    hasher = hashlib.sha256()
    with torch.no_grad():
        for name, param in sorted(model.named_parameters()):
            hasher.update(name.encode("utf-8"))
            hasher.update(param.detach().cpu().numpy().tobytes())
    computed_hash = hasher.hexdigest()

    assert computed_hash == EXPECTED_WEIGHT_HASH, f"Weight hash mismatch: {computed_hash}"
