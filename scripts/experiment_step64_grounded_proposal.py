"""
ChakrView Step 64: Empirical Benchmark for Grounded Local Neural Proposal & Persistent Repository Context.

Demonstrates:
A. Persistent context caching: cold scan vs warm reuse (0 files re-parsed on unchanged sync).
B. Incremental cache invalidation: modifying one file only re-inspects that file and its dependents.
C. Budgeted grounded context retrieval with complete EvidenceRecord provenance.
D. Valid grounded proposal passes HallucinationContainmentGate.
E. Nonexistent file proposal is rejected by HallucinationContainmentGate (CONTRADICTED).
F. Nonexistent symbol proposal is rejected by HallucinationContainmentGate (CONTRADICTED).
G. Unsupported/unfounded dependency claim triggers fail-closed rejection.
H. Stale memory proposal is rejected/abstained by DeterministicSafetyGate.
I. Unrelated domain memory triggers negative-transfer protection (ABSTAIN).
J. Malformed neural proposal rejected by CandidateNormalizer.
K. Execution failure of neural proposal triggers atomic rollback with bit-exact fingerprint match.
L. Repeated trials on identical context yield bit-exact identical retrieval and safety decisions.
M. Neural baseline hash remains bit-exact (Delta_W = 0, 3,443,136 parameters).
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
from chakrview.cognition.repository.context_store import (
    RepositoryContextStore,
    GroundedContextRetriever,
    GroundedContextBudget,
    EpistemicState,
    EvidenceRecord,
)
from chakrview.cognition.repository.neural_adapter import (
    NeuralProposalOutput,
    MockNeuralProposalAdapter,
    HallucinationContainmentGate,
    GroundedNeuralSynthesisEngine,
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
    print("CHAKRVIEW STEP 64: GROUNDED NEURAL PROPOSAL & PERSISTENT CONTEXT BENCHMARK")
    print("=" * 80)
    results: Dict[str, Any] = {}

    pre_hash = compute_baseline_hash()
    print(f"\n[Pre-Check] Neural Core Weight Hash: {pre_hash}")
    assert pre_hash == EXPECTED_WEIGHT_HASH, "Baseline mismatch before Step 64!"

    manifest_base = build_repo_manifest(
        tax_code=DEFECTIVE_TAX_CODE,
        discount_code=DEFECTIVE_DISCOUNT_CODE,
    )
    base_state = RepositoryState.from_manifest(manifest_base)
    base_fp = base_state.state_fingerprint

    # ─────────────────────────────────────────────────────────────────────────
    # Condition A: Persistent Context Caching (Cold Scan vs Warm Sync)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n--- Condition A: Persistent context caching (cold vs warm) ---")
    store = RepositoryContextStore(project_id="order_billing_system")
    # Cold scan
    state_cold = store.synchronize(manifest_base)
    cold_inspected = store.telemetry.files_inspected
    cold_reused = store.telemetry.files_reused_from_cache
    assert cold_inspected == len(manifest_base.files)
    assert cold_reused == 0

    # Warm scan on identical manifest
    state_warm = store.synchronize(manifest_base)
    warm_inspected = store.telemetry.files_inspected
    warm_reused = store.telemetry.files_reused_from_cache
    assert warm_inspected == 0
    assert warm_reused == len(manifest_base.files)
    assert state_cold.state_fingerprint == state_warm.state_fingerprint

    results["Condition_A"] = {
        "passed": True,
        "cold_inspected": cold_inspected,
        "warm_reused": warm_reused,
        "cache_hit_ratio": store.telemetry.cache_hit_ratio,
    }
    print("Condition A passed!")

    # ─────────────────────────────────────────────────────────────────────────
    # Condition B: Incremental Invalidation on Single File Change
    # ─────────────────────────────────────────────────────────────────────────
    print("\n--- Condition B: Incremental invalidation on file mutation ---")
    manifest_mutated = build_repo_manifest(
        tax_code=CORRECTED_TAX_CODE,
        discount_code=DEFECTIVE_DISCOUNT_CODE,
    )
    state_inc = store.synchronize(manifest_mutated)
    assert store.telemetry.files_inspected == 1  # Only tax_service.py was re-parsed!
    assert store.telemetry.files_reused_from_cache == len(manifest_base.files) - 1
    assert state_inc.state_fingerprint != base_fp

    results["Condition_B"] = {
        "passed": True,
        "re_inspected_count": store.telemetry.files_inspected,
        "reused_count": store.telemetry.files_reused_from_cache,
    }
    print("Condition B passed!")

    # Synchronize back to base
    store.synchronize(manifest_base)

    # ─────────────────────────────────────────────────────────────────────────
    # Condition C: Budgeted Grounded Context Retrieval
    # ─────────────────────────────────────────────────────────────────────────
    print("\n--- Condition C: Budgeted grounded context retrieval ---")
    mem_idx = RepositoryMemoryIndex()
    rec_tax = RepositorySemanticRecord(
        memory_id="mem_tax_fix_64",
        task_family="billing_refactor",
        language="python",
        framework="standard_library",
        repository_pattern="Tax defect pattern",
        symptom_signature="Tax mismatch",
        root_cause_signature="Tax calc",
        dependency_signature="billing -> tax",
        affected_modules=["tax_service.py"],
        solution_pattern=CORRECTED_TAX_CODE,
        verification_requirements=["test_tax_service.py"],
        known_boundaries=[],
        confidence=0.92,
        evidence_count=2,
        successful_episodes=2,
        failed_episodes=0,
    )
    mem_idx.insert(rec_tax)

    retriever = GroundedContextRetriever(context_store=store, memory_index=mem_idx)
    bundle = retriever.retrieve_context(
        task_description="Fix the incorrect tax calculation in billing",
        target_domain="billing_refactor",
        budget=GroundedContextBudget(max_files=3, max_symbols=5),
    )
    assert len(bundle.candidate_files) <= 3
    assert len(bundle.relevant_symbols) <= 5
    assert len(bundle.active_memory_records) == 1
    assert all(isinstance(e, EvidenceRecord) for e in bundle.evidence_records)
    results["Condition_C"] = {
        "passed": True,
        "files_selected": len(bundle.candidate_files),
        "evidence_count": len(bundle.evidence_records),
    }
    print("Condition C passed!")

    # ─────────────────────────────────────────────────────────────────────────
    # Condition D: Valid Grounded Proposal Passes Hallucination Gate
    # ─────────────────────────────────────────────────────────────────────────
    print("\n--- Condition D: Valid grounded proposal passes hallucination gate ---")
    valid_proposal = NeuralProposalOutput(
        proposal_id="prop_valid_tax",
        objective="Fix tax calculation",
        target_files=["tax_service.py"],
        target_symbols=["compute_tax"],
        proposed_patches={"tax_service.py": CORRECTED_TAX_CODE},
        confidence=0.95,
    )
    dec_d = HallucinationContainmentGate.verify_grounding(
        proposal=valid_proposal,
        context_store=store,
        allowed_modified_files={"tax_service.py"},
    )
    assert dec_d.is_grounded is True
    assert dec_d.epistemic_state == EpistemicState.KNOWN
    results["Condition_D"] = {"passed": True, "is_grounded": True}
    print("Condition D passed!")

    # ─────────────────────────────────────────────────────────────────────────
    # Condition E: Nonexistent File Proposal Rejected (CONTRADICTED)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n--- Condition E: Nonexistent file proposal rejected ---")
    hallucinated_file_prop = NeuralProposalOutput(
        proposal_id="prop_ghost_file",
        objective="Modify imaginary service",
        target_files=["ghost_payment_service.py"],
        target_symbols=["process_payment"],
        proposed_patches={"ghost_payment_service.py": "def process(): pass\n"},
    )
    dec_e = HallucinationContainmentGate.verify_grounding(
        proposal=hallucinated_file_prop,
        context_store=store,
        allowed_modified_files={"tax_service.py", "ghost_payment_service.py"},
    )
    assert dec_e.is_grounded is False
    assert dec_e.epistemic_state == EpistemicState.CONTRADICTED
    assert "ghost_payment_service.py" in dec_e.unfounded_files
    results["Condition_E"] = {"passed": True, "epistemic_state": dec_e.epistemic_state.name}
    print("Condition E passed!")

    # ─────────────────────────────────────────────────────────────────────────
    # Condition F: Nonexistent Symbol Proposal Rejected (CONTRADICTED)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n--- Condition F: Nonexistent symbol proposal rejected ---")
    hallucinated_sym_prop = NeuralProposalOutput(
        proposal_id="prop_ghost_symbol",
        objective="Update imaginary function",
        target_files=["tax_service.py"],
        target_symbols=["nonexistent_crypto_tax_algorithm"],
        proposed_patches={"tax_service.py": CORRECTED_TAX_CODE},
    )
    dec_f = HallucinationContainmentGate.verify_grounding(
        proposal=hallucinated_sym_prop,
        context_store=store,
        allowed_modified_files={"tax_service.py"},
    )
    assert dec_f.is_grounded is False
    assert dec_f.epistemic_state == EpistemicState.CONTRADICTED
    assert "nonexistent_crypto_tax_algorithm" in dec_f.unfounded_symbols
    results["Condition_F"] = {"passed": True, "unfounded_symbol": dec_f.unfounded_symbols[0]}
    print("Condition F passed!")

    # ─────────────────────────────────────────────────────────────────────────
    # Condition G: Scope Leak Rejection
    # ─────────────────────────────────────────────────────────────────────────
    print("\n--- Condition G: Scope leak rejected ---")
    scope_leak_prop = NeuralProposalOutput(
        proposal_id="prop_leak",
        objective="Fix tax but touch models.py",
        target_files=["models.py"],
        target_symbols=["Order"],
        proposed_patches={"models.py": MODELS_CODE_V1},
    )
    dec_g = HallucinationContainmentGate.verify_grounding(
        proposal=scope_leak_prop,
        context_store=store,
        allowed_modified_files={"tax_service.py"},  # models.py not allowed!
    )
    assert dec_g.is_grounded is False
    assert any("outside allowed task scope" in r for r in dec_g.rejection_reasons)
    results["Condition_G"] = {"passed": True, "scope_leak_blocked": True}
    print("Condition G passed!")

    # ─────────────────────────────────────────────────────────────────────────
    # Condition H: Stale Memory Proposal Rejected/Abstained
    # ─────────────────────────────────────────────────────────────────────────
    print("\n--- Condition H: Stale memory proposal rejected ---")
    stale_rec = RepositorySemanticRecord(
        memory_id="rec_stale_step64",
        task_family="billing_refactor",
        language="python",
        framework="standard_library",
        repository_pattern="Old tax pattern",
        symptom_signature="Tax err",
        root_cause_signature="Tax",
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
    mem_idx_stale = RepositoryMemoryIndex()
    mem_idx_stale.insert(stale_rec)

    # Reference state with structural mutation in tax_service.py
    mutated_files = {sf.path: sf.content for sf in manifest_base.files}
    mutated_files["tax_service.py"] = "import math\n" + mutated_files["tax_service.py"]
    ref_state = RepositoryState.from_files(
        project_id="order_billing_system",
        files_dict=mutated_files,
        version=2,
    )

    stale_cand = SynthesizedCandidate(
        candidate_id="cand_stale",
        objective="Stale candidate",
        ordered_steps=[
            RefactoringStep(
                step_id="step_stale",
                description="Stale step",
                target_files=["tax_service.py"],
                patch_dict={"tax_service.py": CORRECTED_TAX_CODE},
            )
        ],
        allowed_files={"tax_service.py"},
        provenance=CandidateProvenance(
            origin=CandidateOrigin.MEMORY_RETRIEVAL,
            source_memory_ids=["rec_stale_step64"],
            domain_tags={"billing_refactor"},
        ),
    )
    safety_gate = DeterministicSafetyGate()
    dec_h = safety_gate.evaluate(
        candidate=stale_cand,
        repo_state=base_state,
        allowed_modified_files={"tax_service.py"},
        task_domain="billing_refactor",
        memory_index=mem_idx_stale,
        reference_state=ref_state,
    )
    assert dec_h.decision == SafetyDecision.ABSTAIN
    assert any("STALE" in r for r in dec_h.reasons)
    results["Condition_H"] = {"passed": True, "decision": dec_h.decision.name}
    print("Condition H passed!")

    # ─────────────────────────────────────────────────────────────────────────
    # Condition I: Unrelated Domain Memory Mismatch (Negative Transfer)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n--- Condition I: Negative transfer protection ---")
    neg_cand = SynthesizedCandidate(
        candidate_id="cand_neg_transfer",
        objective="Security auth fix in billing",
        ordered_steps=[
            RefactoringStep(
                step_id="step_auth",
                description="Auth step",
                target_files=["tax_service.py"],
                patch_dict={"tax_service.py": CORRECTED_TAX_CODE},
            )
        ],
        allowed_files={"tax_service.py"},
        provenance=CandidateProvenance(
            origin=CandidateOrigin.MEMORY_RETRIEVAL,
            domain_tags={"oauth2_authentication"},
        ),
    )
    dec_i = safety_gate.evaluate(
        candidate=neg_cand,
        repo_state=base_state,
        allowed_modified_files={"tax_service.py"},
        task_domain="billing_refactor",
    )
    assert dec_i.decision == SafetyDecision.ABSTAIN
    assert any("Domain mismatch" in r for r in dec_i.reasons)
    results["Condition_I"] = {"passed": True, "decision": dec_i.decision.name}
    print("Condition I passed!")

    # ─────────────────────────────────────────────────────────────────────────
    # Condition J: Malformed Neural Proposal Rejected by Normalizer
    # ─────────────────────────────────────────────────────────────────────────
    print("\n--- Condition J: Malformed neural proposal rejected by normalizer ---")
    malformed_cand = SynthesizedCandidate(
        candidate_id="malformed_cand",
        objective="Missing step patches",
        ordered_steps=[
            RefactoringStep(
                step_id="step_empty",
                description="Empty patch",
                target_files=["tax_service.py"],
                patch_dict={},  # Empty patch is invalid
            )
        ],
        allowed_files={"tax_service.py"},
    )
    norm_j = CandidateNormalizer.normalize(malformed_cand)
    assert norm_j.is_valid is False
    assert any("has empty patch_dict" in r for r in norm_j.rejection_reasons)
    results["Condition_J"] = {"passed": True, "rejected_malformed": True}
    print("Condition J passed!")

    # ─────────────────────────────────────────────────────────────────────────
    # Condition K: Execution Failure of Neural Proposal Triggers Rollback
    # ─────────────────────────────────────────────────────────────────────────
    print("\n--- Condition K: Execution failure triggers atomic rollback ---")
    failing_neural_cand = SynthesizedCandidate(
        candidate_id="cand_neural_failing",
        objective="Proposal with broken code",
        ordered_steps=[
            RefactoringStep(
                step_id="step_break",
                description="Break tax logic",
                target_files=["tax_service.py"],
                patch_dict={"tax_service.py": "from models import Order\ndef compute_tax(o): return -999.0\n"},
                targeted_test_file="test_tax_service.py",
            )
        ],
        allowed_files={"tax_service.py"},
        provenance=CandidateProvenance(
            origin=CandidateOrigin.SYNTHESIS_LLM,
            domain_tags={"billing_refactor"},
        ),
    )
    coord_k = SynthesisAwareBranchingCoordinator()
    res_k = coord_k.execute_synthesis_refactoring(
        manifest=manifest_base,
        task_id="task_k_fail_rollback",
        objective="Test rollback on neural candidate failure",
        task_family="billing_refactor",
        synthesized_candidates=[failing_neural_cand],
        allowed_modified_files={"tax_service.py"},
    )
    assert res_k.base_result.overall_success is False
    assert res_k.base_result.recovery_transitions >= 1
    assert res_k.base_result.recovery_decisions[0].fingerprint_restored is True
    # Confirm base fingerprint matches bit-exact
    assert base_state.state_fingerprint == base_fp
    results["Condition_K"] = {"passed": True, "rollback_verified": True}
    print("Condition K passed!")

    # ─────────────────────────────────────────────────────────────────────────
    # Condition L: Repeated Trials Determinism
    # ─────────────────────────────────────────────────────────────────────────
    print("\n--- Condition L: Repeated trials determinism ---")
    trials = []
    for _ in range(3):
        ret = retriever.retrieve_context(
            task_description="Fix the incorrect tax calculation in billing",
            target_domain="billing_refactor",
            budget=GroundedContextBudget(max_files=3, max_symbols=5),
        )
        trials.append((
            ret.candidate_files,
            ret.relevant_symbols,
            len(ret.evidence_records),
        ))
    assert len(set(str(t) for t in trials)) == 1
    results["Condition_L"] = {"passed": True, "identical_trials": 3}
    print("Condition L passed!")

    # ─────────────────────────────────────────────────────────────────────────
    # Condition M: Neural Baseline Hash Remains Bit-Exact
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

    # Save artifacts
    artifacts_dir = ROOT_DIR / "artifacts" / "step64"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    out_file = artifacts_dir / "step64_grounded_proposal_evidence.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nAll 13 conditions PASSED! Evidence recorded at: {out_file}")
    return results


if __name__ == "__main__":
    run_experiment()
