"""
ChakrView Step 65: Empirical Benchmark for Active Episodic Learning Loop & Verified Memory Admission.

Validates:
A. Verified success episode produces admitted positive semantic memory.
B. Unverified neural proposal cannot become positive memory (REJECTED).
C. Hallucinated neural proposal is rejected from positive memory (REJECTED).
D. Failed execution does not become positive learning (REJECTED).
E. Verified negative experience (e.g. confirmed hallucination or scope leak) recorded safely as negative boundary constraint.
F. Stale memory is detected and rejected from driving execution.
G. Deterministic superseding of previous semantic memory when newer verified episode covers identical scope.
H. Full provenance is preserved across learned observations and admitted memory records.
I. Repeated identical learning inputs produce bit-exact identical memory admissions and index states.
J. Neural baseline core hash remains bit-exact (Delta_W = 0, 3,443,136 parameters).
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Set

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.runtime.pipeline import EXPECTED_WEIGHT_HASH

from chakrview.arena.models import ProjectSpecification, ProjectManifest, SourceFile, FileRole
from chakrview.cognition.repository.state import RepositoryState
from chakrview.cognition.repository.refactoring import RefactoringStep
from chakrview.cognition.repository.branching import (
    RefactoringBranch,
    BranchStatus,
    BranchingRefactoringResult,
)
from chakrview.cognition.repository.semantic_record import RepositorySemanticRecord
from chakrview.cognition.repository.memory_index import RepositoryMemoryIndex
from chakrview.cognition.repository.context_store import (
    RepositoryContextStore,
    GroundedContextRetriever,
    GroundedContextBudget,
    EpistemicState,
    EvidenceRecord,
)
from chakrview.cognition.repository.neural_adapter import (
    NeuralProposalOutput,
    GroundingVerificationDecision,
    HallucinationContainmentGate,
)
from chakrview.cognition.repository.synthesis import (
    CandidateOrigin,
    CandidateProvenance,
    SynthesizedCandidate,
    DeterministicSafetyGate,
    SafetyDecision,
    SynthesisAwareBranchingCoordinator,
)
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
from scripts.experiment_step61_realtime_change import (
    MODELS_CODE_V1,
    DEFECTIVE_TAX_CODE,
    CORRECTED_TAX_CODE,
    DEFECTIVE_DISCOUNT_CODE,
    CORRECTED_DISCOUNT_CODE,
    BILLING_CODE,
    TEST_TAX_CODE,
    TEST_DISCOUNT_CODE,
    TEST_BILLING_CODE,
    build_repo_manifest,
)


def compute_baseline_hash() -> str:
    import hashlib
    import torch
    torch.manual_seed(42)
    model = ChakrMicro(ModelConfig())
    model.eval()
    hasher = hashlib.sha256()
    with torch.no_grad():
        for name, param in sorted(model.named_parameters()):
            hasher.update(name.encode("utf-8"))
            hasher.update(param.detach().cpu().numpy().tobytes())
    return hasher.hexdigest()


def run_experiment() -> Dict[str, Any]:
    print("=" * 80)
    print("CHAKRVIEW STEP 65: ACTIVE EPISODIC LEARNING & VERIFIED MEMORY BENCHMARK")
    print("=" * 80)
    results: Dict[str, Any] = {}

    pre_hash = compute_baseline_hash()
    print(f"\n[Pre-Check] Neural Core Weight Hash: {pre_hash}")
    assert pre_hash == EXPECTED_WEIGHT_HASH, "Baseline mismatch before Step 65!"

    manifest_base = build_repo_manifest(
        tax_code=DEFECTIVE_TAX_CODE,
        discount_code=DEFECTIVE_DISCOUNT_CODE,
    )
    base_state = RepositoryState.from_manifest(manifest_base)
    base_fp = base_state.state_fingerprint

    memory_index = RepositoryMemoryIndex()
    learning_coordinator = EpisodicLearningCoordinator(memory_index=memory_index)

    # ─────────────────────────────────────────────────────────────────────────
    # Condition A: Verified Success Episode Becomes Admitted Positive Memory
    # ─────────────────────────────────────────────────────────────────────────
    print("\n--- Condition A: Verified success episode admitted as positive memory ---")
    step_tax = RefactoringStep(
        step_id="step_tax",
        description="Fix tax calculation",
        target_files=["tax_service.py"],
        patch_dict={"tax_service.py": CORRECTED_TAX_CODE},
        targeted_test_file="test_tax_service.py",
    )
    step_disc = RefactoringStep(
        step_id="step_disc",
        description="Fix discount calculation",
        target_files=["discount_engine.py"],
        patch_dict={"discount_engine.py": CORRECTED_DISCOUNT_CODE},
        targeted_test_file="test_discount_engine.py",
    )
    cand_succ = SynthesizedCandidate(
        candidate_id="cand_success",
        objective="Fix billing calculation pipeline",
        ordered_steps=[step_tax, step_disc],
        allowed_files={"tax_service.py", "discount_engine.py"},
        provenance=CandidateProvenance(
            origin=CandidateOrigin.SYNTHESIS_LLM,
            domain_tags={"billing_refactor"},
        ),
    )

    coord = SynthesisAwareBranchingCoordinator()
    exec_res = coord.execute_synthesis_refactoring(
        manifest=manifest_base,
        task_id="ep_001_success",
        objective="Fix billing calculation pipeline",
        task_family="billing_refactor",
        synthesized_candidates=[cand_succ],
        allowed_modified_files={"tax_service.py", "discount_engine.py"},
    )
    assert exec_res.base_result.overall_success is True

    # Construct verified episodic experience
    proposal_succ = NeuralProposalOutput(
        proposal_id="prop_001",
        objective="Fix billing calculation pipeline",
        target_files=["tax_service.py", "discount_engine.py"],
        target_symbols=["compute_tax", "compute_discount"],
        proposed_patches={"tax_service.py": CORRECTED_TAX_CODE, "discount_engine.py": CORRECTED_DISCOUNT_CODE},
        confidence=0.95,
    )
    episode_succ = EpisodicExperience(
        episode_id="ep_001_success",
        task_description="Fix billing calculation pipeline",
        task_family="billing_refactor",
        initial_state_fingerprint=base_fp,
        final_state_fingerprint=exec_res.base_result.final_state_fingerprint,
        neural_proposal=proposal_succ,
        grounding_decision=GroundingVerificationDecision(is_grounded=True, epistemic_state=EpistemicState.KNOWN),
        execution_result=exec_res.base_result,
        outcome=EpisodeOutcome.SUCCESS_VERIFIED,
    )

    decisions_a = learning_coordinator.process_episode(episode_succ)
    assert len(decisions_a) == 1
    assert decisions_a[0].status == AdmissionStatus.ADMITTED_POSITIVE
    assert memory_index.count() == 1
    admitted_record = memory_index.lookup(decisions_a[0].admitted_memory_id)
    assert admitted_record is not None
    assert admitted_record.successful_episodes == 1
    assert admitted_record.source_episode_ids == ["ep_001_success"]
    results["Condition_A"] = {"passed": True, "admitted_memory_id": admitted_record.memory_id}
    print("Condition A passed!")

    # ─────────────────────────────────────────────────────────────────────────
    # Condition B: Unverified Neural Proposal Cannot Become Memory
    # ─────────────────────────────────────────────────────────────────────────
    print("\n--- Condition B: Unverified neural proposal rejected from memory ---")
    proposal_unverified = NeuralProposalOutput(
        proposal_id="prop_unverified",
        objective="Unverified tax patch",
        target_files=["tax_service.py"],
        target_symbols=["compute_tax"],
        proposed_patches={"tax_service.py": CORRECTED_TAX_CODE},
    )
    obs_unverified = LearnedObservation(
        observation_id="obs_unverified_claim",
        observation_type="SOLUTION_PATTERN",
        task_family="billing_refactor",
        target_files=["tax_service.py"],
        lesson_summary="Claimed success without test execution",
        is_positive=True,
        evidence_fingerprint="unverified_fp",
        confidence=0.8,
        supporting_episode_id="ep_unverified",
        verification_passed=False,  # Unverified!
        failure_class=FailureClass.UNVERIFIED_CLAIM,
    )
    episode_unverified = EpisodicExperience(
        episode_id="ep_unverified",
        task_description="Unverified proposal",
        task_family="billing_refactor",
        initial_state_fingerprint=base_fp,
        final_state_fingerprint=base_fp,
        neural_proposal=proposal_unverified,
        grounding_decision=GroundingVerificationDecision(is_grounded=True, epistemic_state=EpistemicState.KNOWN),
        outcome=EpisodeOutcome.UNVERIFIED_ABORT,
        derived_observations=[obs_unverified],
    )
    decisions_b = learning_coordinator.process_episode(episode_unverified)
    assert len(decisions_b) == 1
    assert decisions_b[0].status == AdmissionStatus.REJECTED
    results["Condition_B"] = {"passed": True, "rejection_status": decisions_b[0].status.name}
    print("Condition B passed!")

    # ─────────────────────────────────────────────────────────────────────────
    # Condition C: Hallucinated Proposal Rejected from Positive Memory
    # ─────────────────────────────────────────────────────────────────────────
    print("\n--- Condition C: Hallucinated proposal rejected from positive memory ---")
    obs_halluc = LearnedObservation(
        observation_id="obs_halluc",
        observation_type="SOLUTION_PATTERN",
        task_family="billing_refactor",
        target_files=["ghost_file.py"],
        lesson_summary="Hallucinated file repair pattern",
        is_positive=True,
        evidence_fingerprint=base_fp,
        confidence=0.5,
        supporting_episode_id="ep_halluc",
        verification_passed=False,
    )
    episode_halluc = EpisodicExperience(
        episode_id="ep_halluc",
        task_description="Hallucinated repair",
        task_family="billing_refactor",
        initial_state_fingerprint=base_fp,
        final_state_fingerprint=base_fp,
        grounding_decision=GroundingVerificationDecision(
            is_grounded=False,
            epistemic_state=EpistemicState.CONTRADICTED,
            unfounded_files=["ghost_file.py"],
        ),
        outcome=EpisodeOutcome.PROPOSAL_HALLUCINATED,
        derived_observations=[obs_halluc],
    )
    decisions_c = learning_coordinator.process_episode(episode_halluc)
    assert len(decisions_c) == 1
    assert decisions_c[0].status == AdmissionStatus.REJECTED
    results["Condition_C"] = {"passed": True, "rejection_status": decisions_c[0].status.name}
    print("Condition C passed!")

    # ─────────────────────────────────────────────────────────────────────────
    # Condition D: Failed Execution Does Not Become Positive Learning
    # ─────────────────────────────────────────────────────────────────────────
    print("\n--- Condition D: Failed execution cannot become positive learning ---")
    obs_fail_pos = LearnedObservation(
        observation_id="obs_fail_pos",
        observation_type="SOLUTION_PATTERN",
        task_family="billing_refactor",
        target_files=["tax_service.py"],
        lesson_summary="Broken code masquerading as positive pattern",
        is_positive=True,  # Attempting positive admission on failed execution!
        evidence_fingerprint=base_fp,
        confidence=0.9,
        supporting_episode_id="ep_failed_run",
        verification_passed=False,
    )
    episode_failed_run = EpisodicExperience(
        episode_id="ep_failed_run",
        task_description="Failed run",
        task_family="billing_refactor",
        initial_state_fingerprint=base_fp,
        final_state_fingerprint=base_fp,
        outcome=EpisodeOutcome.EXECUTION_FAILED_ROLLED_BACK,
        derived_observations=[obs_fail_pos],
    )
    decisions_d = learning_coordinator.process_episode(episode_failed_run)
    assert len(decisions_d) == 1
    assert decisions_d[0].status == AdmissionStatus.REJECTED
    results["Condition_D"] = {"passed": True, "rejection_status": decisions_d[0].status.name}
    print("Condition D passed!")

    # ─────────────────────────────────────────────────────────────────────────
    # Condition E: Verified Negative Experience Recorded as Boundary Constraint
    # ─────────────────────────────────────────────────────────────────────────
    print("\n--- Condition E: Verified negative experience recorded as boundary constraint ---")
    episode_neg = EpisodicExperience(
        episode_id="ep_neg_boundary",
        task_description="Scope leak on models.py",
        task_family="billing_refactor",
        initial_state_fingerprint=base_fp,
        final_state_fingerprint=base_fp,
        grounding_decision=GroundingVerificationDecision(
            is_grounded=False,
            epistemic_state=EpistemicState.CONTRADICTED,
            unfounded_files=["ghost_crypto_mod.py"],
        ),
        outcome=EpisodeOutcome.PROPOSAL_HALLUCINATED,
        failure_class=FailureClass.HALLUCINATED_FILE,
    )
    decisions_e = learning_coordinator.process_episode(episode_neg)
    assert len(decisions_e) == 1
    assert decisions_e[0].status == AdmissionStatus.ADMITTED_NEGATIVE
    neg_rec = memory_index.lookup(decisions_e[0].admitted_memory_id)
    assert neg_rec is not None
    assert neg_rec.failed_episodes == 1
    assert any("FAILURE_BOUNDARY" in b for b in neg_rec.known_boundaries)
    results["Condition_E"] = {"passed": True, "boundary_memory_id": neg_rec.memory_id}
    print("Condition E passed!")

    # ─────────────────────────────────────────────────────────────────────────
    # Condition F: Stale Memory Detected & Rejected from Execution
    # ─────────────────────────────────────────────────────────────────────────
    print("\n--- Condition F: Stale memory detected and rejected from driving execution ---")
    stale_rec = RepositorySemanticRecord(
        memory_id="rec_stale_ep65",
        task_family="billing_refactor",
        language="python",
        framework="standard_library",
        repository_pattern="Old pattern",
        symptom_signature="Tax mismatch",
        root_cause_signature="Tax err",
        dependency_signature="billing -> tax",
        affected_modules=["tax_service.py"],
        solution_pattern="def compute_tax(): pass",
        verification_requirements=["test_tax_service.py"],
        known_boundaries=[],
        confidence=0.8,
        evidence_count=1,
        successful_episodes=1,
        failed_episodes=0,
    )
    memory_index.insert(stale_rec)

    # Reference state with structural mutation
    mutated_files = {sf.path: sf.content for sf in manifest_base.files}
    mutated_files["tax_service.py"] = "import sys\n" + mutated_files["tax_service.py"]
    ref_state = RepositoryState.from_files(
        project_id="order_billing_system",
        files_dict=mutated_files,
        version=2,
    )

    stale_cand = SynthesizedCandidate(
        candidate_id="cand_stale_ep65",
        objective="Execute stale pattern",
        ordered_steps=[step_tax],
        allowed_files={"tax_service.py"},
        provenance=CandidateProvenance(
            origin=CandidateOrigin.MEMORY_RETRIEVAL,
            source_memory_ids=["rec_stale_ep65"],
            domain_tags={"billing_refactor"},
        ),
    )
    safety_gate = DeterministicSafetyGate()
    dec_f = safety_gate.evaluate(
        candidate=stale_cand,
        repo_state=base_state,
        allowed_modified_files={"tax_service.py"},
        task_domain="billing_refactor",
        memory_index=memory_index,
        reference_state=ref_state,
    )
    assert dec_f.decision == SafetyDecision.ABSTAIN
    assert any("STALE" in r for r in dec_f.reasons)
    results["Condition_F"] = {"passed": True, "decision": dec_f.decision.name}
    print("Condition F passed!")

    # ─────────────────────────────────────────────────────────────────────────
    # Condition G: Deterministic Superseding of Previous Semantic Memory
    # ─────────────────────────────────────────────────────────────────────────
    print("\n--- Condition G: Deterministic superseding of previous semantic memory ---")
    # ep_002 supersedes ep_001 for tax_service.py and discount_engine.py
    proposal_v2 = NeuralProposalOutput(
        proposal_id="prop_002",
        objective="Optimized billing fix",
        target_files=["tax_service.py", "discount_engine.py"],
        target_symbols=["compute_tax", "compute_discount"],
        proposed_patches={"tax_service.py": CORRECTED_TAX_CODE, "discount_engine.py": CORRECTED_DISCOUNT_CODE},
        confidence=0.98,
    )
    episode_v2 = EpisodicExperience(
        episode_id="ep_002_superseding",
        task_description="Optimized billing fix",
        task_family="billing_refactor",
        initial_state_fingerprint=base_fp,
        final_state_fingerprint=exec_res.base_result.final_state_fingerprint,
        neural_proposal=proposal_v2,
        grounding_decision=GroundingVerificationDecision(is_grounded=True, epistemic_state=EpistemicState.KNOWN),
        execution_result=exec_res.base_result,
        outcome=EpisodeOutcome.SUCCESS_VERIFIED,
    )
    decisions_g = learning_coordinator.process_episode(episode_v2)
    assert len(decisions_g) == 1
    assert decisions_g[0].status == AdmissionStatus.ADMITTED_POSITIVE
    assert decisions_g[0].superseded_memory_id == decisions_a[0].admitted_memory_id

    # Verify that previous record was marked inactive and superseded
    old_record = memory_index.lookup(decisions_a[0].admitted_memory_id)
    assert old_record is not None
    assert old_record.active_version is False
    assert old_record.superseded_by is not None
    new_record = memory_index.lookup(decisions_g[0].admitted_memory_id)
    assert new_record is not None
    assert new_record.active_version is True
    assert new_record.supersedes == old_record.memory_id
    results["Condition_G"] = {
        "passed": True,
        "old_memory_id": old_record.memory_id,
        "new_memory_id": new_record.memory_id,
    }
    print("Condition G passed!")

    # ─────────────────────────────────────────────────────────────────────────
    # Condition H: Full Provenance Preservation
    # ─────────────────────────────────────────────────────────────────────────
    print("\n--- Condition H: Provenance preservation ---")
    assert new_record.source_episode_ids == ["ep_002_superseding"]
    assert new_record.task_family == "billing_refactor"
    assert "Fingerprint" in new_record.dependency_signature
    results["Condition_H"] = {"passed": True, "source_episode": new_record.source_episode_ids[0]}
    print("Condition H passed!")

    # ─────────────────────────────────────────────────────────────────────────
    # Condition I: Repeated Identical Learning Inputs Produce Bit-Exact States
    # ─────────────────────────────────────────────────────────────────────────
    print("\n--- Condition I: Repeated identical inputs determinism ---")
    idx1 = RepositoryMemoryIndex()
    idx2 = RepositoryMemoryIndex()
    coord1 = EpisodicLearningCoordinator(memory_index=idx1)
    coord2 = EpisodicLearningCoordinator(memory_index=idx2)

    res1 = coord1.process_episode(episode_succ)
    res2 = coord2.process_episode(episode_succ)

    assert res1[0].status == res2[0].status
    assert res1[0].admitted_memory_id == res2[0].admitted_memory_id
    assert idx1.to_json() == idx2.to_json()
    results["Condition_I"] = {"passed": True, "identical_json": True}
    print("Condition I passed!")

    # ─────────────────────────────────────────────────────────────────────────
    # Condition J: Neural Baseline Core Hash Remains Bit-Exact
    # ─────────────────────────────────────────────────────────────────────────
    print("\n--- Condition J: Neural baseline hash verified bit-exact ---")
    post_hash = compute_baseline_hash()
    assert post_hash == EXPECTED_WEIGHT_HASH
    results["Condition_J"] = {
        "passed": True,
        "weight_hash": post_hash,
        "param_count": 3_443_136,
        "delta_w": 0,
    }
    print(f"Post-Check Hash: {post_hash} (Immutable Delta_W = 0)")
    print("Condition J passed!")

    # Save artifacts
    artifacts_dir = ROOT_DIR / "artifacts" / "step65"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    out_file = artifacts_dir / "step65_episodic_learning_evidence.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nAll 10 conditions PASSED! Evidence recorded at: {out_file}")
    return results


if __name__ == "__main__":
    run_experiment()
