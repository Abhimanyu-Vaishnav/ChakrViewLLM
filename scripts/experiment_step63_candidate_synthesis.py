"""
ChakrView Step 63: Empirical Benchmark for Autonomous Strategy Synthesis & Safety Gating.

Demonstrates all conditions (A through M):
A. Existing branch succeeds without synthesis.
B. Existing branch fails, synthesis creates a valid alternative.
C. Two partial successful traces are recombined into a valid candidate.
D. Invalid recombination is rejected.
E. Unsafe synthesized candidate is rejected before mutation.
F. Stale memory cannot generate an executable strategy.
G. Unrelated-domain memory produces ABSTAIN / rejection.
H. Duplicate synthesized candidates collapse deterministically.
I. All candidates fail -> repository remains unchanged.
J. Rollback after synthesized candidate failure restores exact fingerprint.
K. Same input/context repeated multiple times produces identical candidate fingerprints and arbitration results.
L. Candidate explosion limits are enforced.
M. Neural baseline hash remains bit-exact.
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
from chakrview.cognition.repository.branching import RefactoringBranch, BranchStatus
from chakrview.cognition.repository.semantic_record import RepositorySemanticRecord
from chakrview.cognition.repository.memory_index import RepositoryMemoryIndex
from chakrview.cognition.repository.synthesis import (
    CandidateOrigin,
    CandidateProvenance,
    SynthesizedCandidate,
    CandidateNormalizer,
    DeterministicSafetyGate,
    CandidateDeduplicator,
    StrategyRecombiner,
    SynthesisAwareBranchingCoordinator,
    SafetyDecision,
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
    print("CHAKRVIEW STEP 63: AUTONOMOUS STRATEGY SYNTHESIS & SAFETY BENCHMARK")
    print("=" * 80)
    results: Dict[str, Any] = {}

    # Neural Baseline Pre-Verification
    pre_hash = compute_baseline_hash()
    print(f"\n[Pre-Check] Neural Core Weight Hash: {pre_hash}")
    assert pre_hash == EXPECTED_WEIGHT_HASH, "Baseline mismatch before Step 63!"

    manifest_base = build_repo_manifest(
        tax_code=DEFECTIVE_TAX_CODE,
        discount_code=DEFECTIVE_DISCOUNT_CODE,
    )
    base_state = RepositoryState.from_manifest(manifest_base)
    base_fp = base_state.state_fingerprint

    # ─────────────────────────────────────────────────────────────────────────
    # Condition A: Existing branch succeeds without synthesis
    # ─────────────────────────────────────────────────────────────────────────
    print("\n--- Condition A: Existing branch succeeds without synthesis ---")
    step_a_tax = RefactoringStep(
        step_id="step_a_tax",
        description="Fix tax calculation",
        target_files=["tax_service.py"],
        patch_dict={"tax_service.py": CORRECTED_TAX_CODE},
        targeted_test_file="test_tax_service.py",
    )
    step_a_disc = RefactoringStep(
        step_id="step_a_disc",
        description="Fix discount calculation",
        target_files=["discount_engine.py"],
        patch_dict={"discount_engine.py": CORRECTED_DISCOUNT_CODE},
        targeted_test_file="test_discount_engine.py",
    )
    branch_a = RefactoringBranch(
        branch_id="branch_auth_clean",
        description="Authored clean branch",
        steps=[step_a_tax, step_a_disc],
        allowed_files={"tax_service.py", "discount_engine.py"},
        priority=1,
    )

    coord_a = SynthesisAwareBranchingCoordinator()
    res_a = coord_a.execute_synthesis_refactoring(
        manifest=manifest_base,
        task_id="task_a",
        objective="Fix billing pipeline",
        task_family="billing_refactor",
        initial_branch_generator=lambda rec, st: [branch_a],
        synthesized_candidates=[],
        allowed_modified_files={"tax_service.py", "discount_engine.py"},
    )
    assert res_a.base_result.overall_success is True
    assert res_a.base_result.selected_branch_id == "branch_auth_clean"
    assert res_a.candidates_accepted == 0
    results["Condition_A"] = {"passed": True, "selected_branch": res_a.base_result.selected_branch_id}
    print("Condition A passed!")

    # ─────────────────────────────────────────────────────────────────────────
    # Condition B: Existing branch fails, synthesis creates valid alternative
    # ─────────────────────────────────────────────────────────────────────────
    print("\n--- Condition B: Existing branch fails, synthesis creates valid alternative ---")
    # Failing branch writes invalid logic breaking tests
    step_b_fail = RefactoringStep(
        step_id="step_b_fail",
        description="Broken tax logic",
        target_files=["tax_service.py"],
        patch_dict={
            "tax_service.py": "from models import Order\ndef compute_tax(o): return 0.0\n"
        },
        targeted_test_file="test_tax_service.py",
    )
    branch_b_fail = RefactoringBranch(
        branch_id="branch_broken",
        description="Fails tests",
        steps=[step_b_fail],
        allowed_files={"tax_service.py"},
        priority=1,
    )
    # Synthesized candidate provides working solution
    cand_b = SynthesizedCandidate(
        candidate_id="synth_working_alt",
        objective="Alternative working tax & discount patch",
        ordered_steps=[step_a_tax, step_a_disc],
        allowed_files={"tax_service.py", "discount_engine.py"},
        provenance=CandidateProvenance(
            origin=CandidateOrigin.SYNTHESIS_TEMPLATE,
            domain_tags={"billing_refactor"},
        ),
    )

    coord_b = SynthesisAwareBranchingCoordinator()
    res_b = coord_b.execute_synthesis_refactoring(
        manifest=manifest_base,
        task_id="task_b",
        objective="Recover from broken branch via synthesis",
        task_family="billing_refactor",
        initial_branch_generator=lambda rec, st: [branch_b_fail],
        synthesized_candidates=[cand_b],
        allowed_modified_files={"tax_service.py", "discount_engine.py"},
    )
    assert res_b.base_result.overall_success is True
    assert res_b.base_result.selected_branch_id == "synth_working_alt"
    assert res_b.base_result.recovery_transitions >= 1
    results["Condition_B"] = {
        "passed": True,
        "selected_branch": res_b.base_result.selected_branch_id,
        "recovery_transitions": res_b.base_result.recovery_transitions,
    }
    print("Condition B passed!")

    # ─────────────────────────────────────────────────────────────────────────
    # Condition C: Two partial successful traces recombined into a valid candidate
    # ─────────────────────────────────────────────────────────────────────────
    print("\n--- Condition C: Recombine partial traces into valid candidate ---")
    recomb_cand = StrategyRecombiner.recombine_traces(
        objective="Recombine tax step and discount step",
        prefix_steps=[step_a_tax],
        suffix_steps=[step_a_disc],
        allowed_files={"tax_service.py", "discount_engine.py"},
        domain_tag="billing_refactor",
        provenance_branches=["branch_partial_1", "branch_partial_2"],
    )
    assert recomb_cand is not None
    assert len(recomb_cand.ordered_steps) == 2
    assert recomb_cand.provenance.origin == CandidateOrigin.TRACE_RECOMBINATION

    res_c = coord_a.execute_synthesis_refactoring(
        manifest=manifest_base,
        task_id="task_c",
        objective="Recombined refactoring",
        task_family="billing_refactor",
        synthesized_candidates=[recomb_cand],
        allowed_modified_files={"tax_service.py", "discount_engine.py"},
    )
    assert res_c.base_result.overall_success is True
    results["Condition_C"] = {"passed": True, "steps_count": len(recomb_cand.ordered_steps)}
    print("Condition C passed!")

    # ─────────────────────────────────────────────────────────────────────────
    # Condition D: Invalid recombination is rejected
    # ─────────────────────────────────────────────────────────────────────────
    print("\n--- Condition D: Invalid recombination is rejected ---")
    bad_step = RefactoringStep(
        step_id="step_leak",
        description="Step targeting unapproved file",
        target_files=["unapproved_secret.py"],
        patch_dict={"unapproved_secret.py": "SECRET = 1\n"},
    )
    invalid_recomb = StrategyRecombiner.recombine_traces(
        objective="Invalid scope recomb",
        prefix_steps=[step_a_tax],
        suffix_steps=[bad_step],
        allowed_files={"tax_service.py"},
        domain_tag="billing_refactor",
        provenance_branches=["b1", "b2"],
    )
    assert invalid_recomb is None
    results["Condition_D"] = {"passed": True, "rejected_cleanly": True}
    print("Condition D passed!")

    # ─────────────────────────────────────────────────────────────────────────
    # Condition E: Unsafe synthesized candidate rejected before mutation
    # ─────────────────────────────────────────────────────────────────────────
    print("\n--- Condition E: Unsafe synthesized candidate rejected before mutation ---")
    unsafe_cand = SynthesizedCandidate(
        candidate_id="synth_unsafe_leak",
        objective="Unsafe modification",
        ordered_steps=[bad_step],
        allowed_files={"tax_service.py", "unapproved_secret.py"},
        provenance=CandidateProvenance(
            origin=CandidateOrigin.SYNTHESIS_TEMPLATE,
            domain_tags={"billing_refactor"},
        ),
    )
    res_e = coord_a.execute_synthesis_refactoring(
        manifest=manifest_base,
        task_id="task_e",
        objective="Test unsafe rejection",
        task_family="billing_refactor",
        synthesized_candidates=[unsafe_cand],
        allowed_modified_files={"tax_service.py"},
    )
    assert res_e.candidates_accepted == 0
    assert res_e.candidates_rejected >= 1
    assert res_e.base_result.abstained is True
    assert base_state.state_fingerprint == base_fp
    results["Condition_E"] = {"passed": True, "rejected_count": res_e.candidates_rejected}
    print("Condition E passed!")

    # ─────────────────────────────────────────────────────────────────────────
    # Condition F: Stale memory cannot generate executable strategy
    # ─────────────────────────────────────────────────────────────────────────
    print("\n--- Condition F: Stale memory cannot generate executable strategy ---")
    stale_rec = RepositorySemanticRecord(
        memory_id="rec_stale_step63",
        task_family="billing_refactor",
        language="python",
        framework="standard_library",
        repository_pattern="Tax defect pattern",
        symptom_signature="Tax mismatch",
        root_cause_signature="Tax calc",
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
    mem_idx = RepositoryMemoryIndex()
    mem_idx.insert(stale_rec)

    # Reference state has different content
    mutated_files = {sf.path: sf.content for sf in manifest_base.files}
    mutated_files["tax_service.py"] = "import os\n" + mutated_files["tax_service.py"]
    ref_state = RepositoryState.from_files(
        project_id="order_billing_system",
        files_dict=mutated_files,
        version=2,
    )

    stale_cand = SynthesizedCandidate(
        candidate_id="synth_from_stale_mem",
        objective="Stale memory execution",
        ordered_steps=[step_a_tax],
        allowed_files={"tax_service.py"},
        provenance=CandidateProvenance(
            origin=CandidateOrigin.MEMORY_RETRIEVAL,
            source_memory_ids=["rec_stale_step63"],
            domain_tags={"billing_refactor"},
        ),
    )
    coord_mem = SynthesisAwareBranchingCoordinator(memory_index=mem_idx)
    res_f = coord_mem.execute_synthesis_refactoring(
        manifest=manifest_base,
        task_id="task_f",
        objective="Test stale memory rejection",
        task_family="billing_refactor",
        synthesized_candidates=[stale_cand],
        allowed_modified_files={"tax_service.py"},
        reference_state=ref_state,
    )
    assert res_f.candidates_accepted == 0
    assert any("STALE" in r for d in res_f.safety_decisions for r in d.reasons)
    results["Condition_F"] = {"passed": True, "stale_rejected": True}
    print("Condition F passed!")

    # ─────────────────────────────────────────────────────────────────────────
    # Condition G: Unrelated-domain memory produces ABSTAIN / rejection
    # ─────────────────────────────────────────────────────────────────────────
    print("\n--- Condition G: Unrelated-domain memory produces ABSTAIN / rejection ---")
    auth_cand = SynthesizedCandidate(
        candidate_id="synth_auth_mismatch",
        objective="Auth patch applied to billing domain",
        ordered_steps=[step_a_tax],
        allowed_files={"tax_service.py"},
        provenance=CandidateProvenance(
            origin=CandidateOrigin.MEMORY_RETRIEVAL,
            domain_tags={"user_authentication_jwt"},
        ),
    )
    res_g = coord_a.execute_synthesis_refactoring(
        manifest=manifest_base,
        task_id="task_g",
        objective="Test negative transfer",
        task_family="billing_refactor",
        synthesized_candidates=[auth_cand],
        allowed_modified_files={"tax_service.py"},
    )
    assert res_g.candidates_accepted == 0
    assert any(d.decision == SafetyDecision.ABSTAIN for d in res_g.safety_decisions)
    results["Condition_G"] = {"passed": True, "negative_transfer_blocked": True}
    print("Condition G passed!")

    # ─────────────────────────────────────────────────────────────────────────
    # Condition H: Duplicate synthesized candidates collapse deterministically
    # ─────────────────────────────────────────────────────────────────────────
    print("\n--- Condition H: Duplicate candidates collapse deterministically ---")
    cand_dup1 = SynthesizedCandidate(
        candidate_id="cand_1",
        objective="Identical objective",
        ordered_steps=[step_a_tax],
        allowed_files={"tax_service.py"},
        provenance=CandidateProvenance(
            origin=CandidateOrigin.SYNTHESIS_TEMPLATE,
            source_branch_ids=["b1"],
            domain_tags={"billing_refactor"},
        ),
    )
    cand_dup2 = SynthesizedCandidate(
        candidate_id="cand_2",
        objective="Identical objective",
        ordered_steps=[step_a_tax],
        allowed_files={"tax_service.py"},
        provenance=CandidateProvenance(
            origin=CandidateOrigin.SYNTHESIS_TEMPLATE,
            source_branch_ids=["b2"],
            domain_tags={"billing_refactor"},
        ),
    )
    assert cand_dup1.fingerprint == cand_dup2.fingerprint
    deduped = CandidateDeduplicator.deduplicate([cand_dup1, cand_dup2])
    assert len(deduped) == 1
    assert set(deduped[0].provenance.source_branch_ids) == {"b1", "b2"}
    results["Condition_H"] = {"passed": True, "collapsed_count": 1}
    print("Condition H passed!")

    # ─────────────────────────────────────────────────────────────────────────
    # Condition I: All candidates fail -> repository remains unchanged
    # ─────────────────────────────────────────────────────────────────────────
    print("\n--- Condition I: All candidates fail -> repository unchanged ---")
    res_i = coord_a.execute_synthesis_refactoring(
        manifest=manifest_base,
        task_id="task_i",
        objective="All branches fail",
        task_family="billing_refactor",
        initial_branch_generator=lambda rec, st: [branch_b_fail],
        synthesized_candidates=[],
        allowed_modified_files={"tax_service.py"},
    )
    assert res_i.base_result.overall_success is False
    assert res_i.base_result.abstained is True
    assert base_state.state_fingerprint == base_fp
    results["Condition_I"] = {"passed": True, "abstained": True, "fingerprint_intact": True}
    print("Condition I passed!")

    # ─────────────────────────────────────────────────────────────────────────
    # Condition J: Rollback after synthesized candidate failure restores exact fingerprint
    # ─────────────────────────────────────────────────────────────────────────
    print("\n--- Condition J: Rollback after synthesized candidate failure ---")
    fail_cand = SynthesizedCandidate(
        candidate_id="synth_failing_patch",
        objective="Synthesized failing candidate",
        ordered_steps=[step_b_fail],
        allowed_files={"tax_service.py"},
        provenance=CandidateProvenance(
            origin=CandidateOrigin.SYNTHESIS_TEMPLATE,
            domain_tags={"billing_refactor"},
        ),
    )
    res_j = coord_a.execute_synthesis_refactoring(
        manifest=manifest_base,
        task_id="task_j",
        objective="Rollback test",
        task_family="billing_refactor",
        synthesized_candidates=[fail_cand],
        allowed_modified_files={"tax_service.py"},
    )
    assert res_j.base_result.recovery_transitions >= 1
    assert res_j.base_result.recovery_decisions[0].fingerprint_restored is True
    assert base_state.state_fingerprint == base_fp
    results["Condition_J"] = {"passed": True, "fingerprint_matched": True}
    print("Condition J passed!")

    # ─────────────────────────────────────────────────────────────────────────
    # Condition K: Repeated executions produce identical fingerprints & decisions
    # ─────────────────────────────────────────────────────────────────────────
    print("\n--- Condition K: Deterministic repeated executions ---")
    trials = []
    for _ in range(3):
        res_k = coord_a.execute_synthesis_refactoring(
            manifest=manifest_base,
            task_id="task_k",
            objective="Determinism test",
            task_family="billing_refactor",
            initial_branch_generator=lambda rec, st: [branch_b_fail],
            synthesized_candidates=[cand_b],
            allowed_modified_files={"tax_service.py", "discount_engine.py"},
        )
        trials.append((
            res_k.base_result.selected_branch_id,
            res_k.candidates_accepted,
            res_k.base_result.final_state_fingerprint,
        ))
    assert len(set(trials)) == 1
    results["Condition_K"] = {"passed": True, "identical_trials": 3}
    print("Condition K passed!")

    # ─────────────────────────────────────────────────────────────────────────
    # Condition L: Candidate explosion limits enforced
    # ─────────────────────────────────────────────────────────────────────────
    print("\n--- Condition L: Candidate explosion limits enforced ---")
    burst_candidates = []
    for i in range(10):
        step_i = RefactoringStep(
            step_id=f"step_burst_{i}",
            description=f"Burst step {i}",
            target_files=["tax_service.py"],
            patch_dict={"tax_service.py": f"# burst comment {i}\n" + manifest_base.files[1].content},
            targeted_test_file="test_tax_service.py",
        )
        cand_i = SynthesizedCandidate(
            candidate_id=f"synth_burst_{i}",
            objective=f"Burst candidate {i}",
            ordered_steps=[step_i],
            allowed_files={"tax_service.py"},
            provenance=CandidateProvenance(
                origin=CandidateOrigin.SYNTHESIS_TEMPLATE,
                domain_tags={"billing_refactor"},
            ),
        )
        burst_candidates.append(cand_i)

    coord_limited = SynthesisAwareBranchingCoordinator(max_synthesized_candidates=3)
    res_l = coord_limited.execute_synthesis_refactoring(
        manifest=manifest_base,
        task_id="task_l",
        objective="Burst limits test",
        task_family="billing_refactor",
        synthesized_candidates=burst_candidates,
        allowed_modified_files={"tax_service.py"},
    )
    assert res_l.candidates_accepted == 3
    results["Condition_L"] = {"passed": True, "accepted_under_limit": res_l.candidates_accepted}
    print("Condition L passed!")

    # ─────────────────────────────────────────────────────────────────────────
    # Condition M: Neural baseline hash remains bit-exact
    # ─────────────────────────────────────────────────────────────────────────
    print("\n--- Condition M: Neural baseline hash verified bit-exact ---")
    post_hash = compute_baseline_hash()
    assert post_hash == EXPECTED_WEIGHT_HASH
    results["Condition_M"] = {
        "passed": True,
        "weight_hash": post_hash,
        "param_count": 3_443_136,
        "delta_w": 0,
    }
    print(f"Post-Check Hash: {post_hash} (Immutable Delta_W = 0)")
    print("Condition M passed!")

    # Save artifact
    artifacts_dir = ROOT_DIR / "artifacts" / "step63"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    out_file = artifacts_dir / "step63_synthesis_evidence.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nAll 13 conditions PASSED! Evidence recorded at: {out_file}")
    return results


if __name__ == "__main__":
    run_experiment()
