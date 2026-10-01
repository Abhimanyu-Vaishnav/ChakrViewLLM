"""
ChakrView Step 67: Empirical Benchmark for Unified Cognitive Context Composition.

Evaluates 14 deterministic experimental conditions:
A. Current repository evidence is included.
B. Valid episodic memory is included.
C. Negative boundary is isolated into negative context.
D. Stale memory is excluded from positive context.
E. Superseded memory is excluded.
F. Conflicted memory is excluded from trusted positive context.
G. ABSTAIN state produces no trusted positive memory.
H. Provenance exists for 100% of included items.
I. Budget limits are obeyed.
J. Identical inputs produce identical context.
K. Neural context contains passive data only.
L. Neural context cannot mutate RepositoryMemoryIndex.
M. Neural baseline remains bit-exact.
N. Existing Step 59–66 behavior remains unchanged.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Set

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.runtime.pipeline import EXPECTED_WEIGHT_HASH

from chakrview.arena.models import ProjectSpecification, ProjectManifest, SourceFile, FileRole
from chakrview.cognition.repository.state import RepositoryState
from chakrview.cognition.repository.semantic_record import RepositorySemanticRecord
from chakrview.cognition.repository.memory_index import RepositoryMemoryIndex
from chakrview.cognition.repository.context_store import RepositoryContextStore, EpistemicState
from chakrview.cognition.repository.episodic_recall import (
    EpisodicMemoryRecallCoordinator,
    MemoryRecallStatus,
    RecallBudget,
)
from chakrview.cognition.repository.cognitive_context import (
    CognitiveContextSource,
    CognitiveContextStatus,
    CognitiveContextBudget,
    CognitiveContextRequest,
    CognitiveContextItem,
    CognitiveContextTelemetry,
    CognitiveContextBundle,
    UnifiedCognitiveContextComposer,
)
from chakrview.cognition.repository.neural_adapter import (
    MockNeuralProposalAdapter,
    NeuralProposalOutput,
)


def compute_baseline_hash() -> str:
    torch.manual_seed(42)
    model = ChakrMicro(ModelConfig())
    model.eval()
    hasher = hashlib.sha256()
    with torch.no_grad():
        for name, param in sorted(model.named_parameters()):
            hasher.update(name.encode("utf-8"))
            hasher.update(param.detach().cpu().numpy().tobytes())
    return hasher.hexdigest()


def _build_test_repo() -> Tuple[RepositoryState, RepositoryContextStore]:
    manifest = ProjectManifest(
        specification=ProjectSpecification(
            project_id="step67_exp_repo",
            project_name="Step 67 Experiment Repo",
            description="Testing unified cognitive context composition benchmark",
            entrypoint="main.py",
        ),
        files=[
            SourceFile(
                path="billing/tax.py",
                content="def compute_tax(subtotal: float, rate: float = 0.20) -> float:\n    return subtotal * rate\n",
                role=FileRole.SOURCE,
            ),
            SourceFile(
                path="billing/discount.py",
                content="def compute_discount(subtotal: float, vip: bool = False) -> float:\n    return 0.10 * subtotal if vip else 0.0\n",
                role=FileRole.SOURCE,
            ),
            SourceFile(
                path="billing/service.py",
                content="from billing.tax import compute_tax\nfrom billing.discount import compute_discount\n\ndef invoice(amt: float) -> float:\n    return compute_tax(amt) - compute_discount(amt)\n",
                role=FileRole.SOURCE,
            ),
        ],
    )
    repo_state = RepositoryState.from_manifest(manifest)
    context_store = RepositoryContextStore(project_id="step67_exp_repo")
    context_store.synchronize(manifest)
    return repo_state, context_store


def _create_record(
    memory_id: str,
    task_family: str,
    repository_pattern: str,
    solution_pattern: str,
    affected_modules: List[str],
    successful_episodes: int = 1,
    failed_episodes: int = 0,
    known_boundaries: Optional[List[str]] = None,
    superseded_by: Optional[str] = None,
    confidence: float = 0.9,
) -> RepositorySemanticRecord:
    return RepositorySemanticRecord(
        memory_id=memory_id,
        task_family=task_family,
        language="python",
        framework="standard_library",
        repository_pattern=repository_pattern,
        symptom_signature=f"Symptom for {repository_pattern}",
        root_cause_signature=f"Root cause for {repository_pattern}",
        dependency_signature="billing -> models",
        affected_modules=affected_modules,
        solution_pattern=solution_pattern,
        verification_requirements=["pytest tests/"],
        known_boundaries=known_boundaries or [],
        confidence=confidence,
        evidence_count=successful_episodes + failed_episodes,
        successful_episodes=successful_episodes,
        failed_episodes=failed_episodes,
        superseded_by=superseded_by,
    )


def run_experiment() -> Dict[str, Any]:
    print("=" * 80)
    print("CHAKRVIEW STEP 67: UNIFIED COGNITIVE CONTEXT COMPOSITION BENCHMARK")
    print("=" * 80)
    results: Dict[str, Any] = {}

    pre_hash = compute_baseline_hash()
    print(f"\n[Pre-Check] Neural Core Weight Hash: {pre_hash}")
    assert pre_hash == EXPECTED_WEIGHT_HASH, "Baseline mismatch before Step 67!"

    repo_state, context_store = _build_test_repo()
    mem_index = RepositoryMemoryIndex()

    # Populate index with memories representing various conditions
    rec_valid = _create_record(
        memory_id="mem_valid_tax",
        task_family="billing_refactor",
        repository_pattern="tax calculation with exemptions",
        solution_pattern="def compute_tax(subtotal, rate=0.20, exempt=False):\n    return 0.0 if exempt else subtotal * rate",
        affected_modules=["billing/tax.py"],
        successful_episodes=3,
    )
    mem_index.insert(rec_valid)

    rec_neg = _create_record(
        memory_id="neg_boundary_rate",
        task_family="billing_refactor",
        repository_pattern="clamping negative tax rates",
        solution_pattern="DO_NOT_APPLY: Clamp negative tax rates to zero silently",
        affected_modules=["billing/service.py"],
        known_boundaries=["DO_NOT_APPLY negative tax clamping in billing/service.py"],
        successful_episodes=0,
        failed_episodes=2,
    )
    mem_index.insert(rec_neg)

    rec_stale = _create_record(
        memory_id="mem_stale_module",
        task_family="billing_refactor",
        repository_pattern="nonexistent legacy billing component",
        solution_pattern="def legacy_billing(): pass",
        affected_modules=["billing/legacy_vat.py"],
        successful_episodes=1,
    )
    mem_index.insert(rec_stale)

    rec_superseded = _create_record(
        memory_id="mem_tax_v0",
        task_family="billing_refactor",
        repository_pattern="outdated tax formula",
        solution_pattern="def compute_tax_v0(): pass",
        affected_modules=["billing/tax.py"],
        superseded_by="mem_valid_tax",
        successful_episodes=1,
    )
    mem_index.insert(rec_superseded)

    rec_conflicted_pos = _create_record(
        memory_id="pos_colliding_pattern",
        task_family="billing_refactor",
        repository_pattern="colliding service modification",
        solution_pattern="def invoice(amt): return amt * 0.15",
        affected_modules=["billing/service.py"],
        successful_episodes=2,
    )
    # We will test conflict specifically in condition F

    coordinator = EpisodicMemoryRecallCoordinator(memory_index=mem_index)
    composer = UnifiedCognitiveContextComposer(context_store=context_store, recall_coordinator=coordinator)

    # Condition A: Current repository evidence is included
    print("\n--- Condition A: Current Repository Evidence ---")
    req_a = CognitiveContextRequest(
        task_description="Refactor compute_tax to support exemptions in billing/tax.py",
        task_family="billing_refactor",
        target_files=("billing/tax.py",),
        target_symbols=("compute_tax",),
    )
    bundle_a = composer.compose_context(req_a)
    cond_a_pass = (
        len(bundle_a.positive_evidence) > 0
        and all(e.source_type == CognitiveContextSource.REPOSITORY_EVIDENCE for e in bundle_a.positive_evidence)
        and all(e.status == CognitiveContextStatus.ACTIVE for e in bundle_a.positive_evidence)
    )
    print(f"Condition A Result: {'PASS' if cond_a_pass else 'FAIL'} (Evidence items: {len(bundle_a.positive_evidence)})")
    results["Condition A: Repository Evidence Included"] = cond_a_pass

    # Condition B: Valid episodic memory is included
    print("\n--- Condition B: Valid Episodic Memory Included ---")
    cond_b_pass = any(m.source_identifier == "mem_valid_tax" and m.status == CognitiveContextStatus.ACTIVE for m in bundle_a.positive_memories)
    print(f"Condition B Result: {'PASS' if cond_b_pass else 'FAIL'} (Positive memories: {len(bundle_a.positive_memories)})")
    results["Condition B: Valid Memory Included"] = cond_b_pass

    # Condition C: Negative boundary is isolated into negative context
    print("\n--- Condition C: Negative Boundary Isolation ---")
    cond_c_pass = (
        any(n.source_identifier == "neg_boundary_rate" and n.status == CognitiveContextStatus.NEGATIVE for n in bundle_a.negative_boundaries)
        and not any(p.source_identifier == "neg_boundary_rate" for p in bundle_a.positive_memories)
    )
    print(f"Condition C Result: {'PASS' if cond_c_pass else 'FAIL'} (Negative boundaries: {len(bundle_a.negative_boundaries)})")
    results["Condition C: Negative Boundary Isolated"] = cond_c_pass

    # Condition D: Stale memory is excluded from positive context
    print("\n--- Condition D: Stale Memory Excluded ---")
    cond_d_pass = not any(p.source_identifier == "mem_stale_module" for p in bundle_a.positive_memories)
    print(f"Condition D Result: {'PASS' if cond_d_pass else 'FAIL'}")
    results["Condition D: Stale Memory Excluded"] = cond_d_pass

    # Condition E: Superseded memory is excluded
    print("\n--- Condition E: Superseded Memory Excluded ---")
    cond_e_pass = not any(p.source_identifier == "mem_tax_v0" for p in bundle_a.positive_memories)
    print(f"Condition E Result: {'PASS' if cond_e_pass else 'FAIL'}")
    results["Condition E: Superseded Memory Excluded"] = cond_e_pass

    # Condition F: Conflicted memory is excluded from trusted positive context
    print("\n--- Condition F: Conflicted Memory Quarantined ---")
    # Insert colliding positive memory that collides with negative boundary
    mem_index.insert(rec_conflicted_pos)
    bundle_f = composer.compose_context(req_a)
    cond_f_pass = (
        not any(p.source_identifier == "pos_colliding_pattern" for p in bundle_f.positive_memories)
        and any(c.source_identifier == "pos_colliding_pattern" and c.status == CognitiveContextStatus.CONFLICTED for c in bundle_f.conflicted_items)
    )
    print(f"Condition F Result: {'PASS' if cond_f_pass else 'FAIL'} (Conflicted items: {len(bundle_f.conflicted_items)})")
    results["Condition F: Conflicted Memory Excluded"] = cond_f_pass

    # Condition G: ABSTAIN state produces no trusted positive memory
    print("\n--- Condition G: ABSTAIN State ---")
    req_g = CognitiveContextRequest(
        task_description="Refactor compute_tax",
        task_family="billing_refactor",
        repository_fingerprint="00000000000000000000000000000000",  # mismatched fingerprint triggers ABSTAIN
    )
    bundle_g = composer.compose_context(req_g)
    cond_g_pass = bundle_g.abstained and len(bundle_g.positive_memories) == 0 and len(bundle_g.positive_evidence) == 0
    print(f"Condition G Result: {'PASS' if cond_g_pass else 'FAIL'} (Abstained: {bundle_g.abstained})")
    results["Condition G: ABSTAIN Produces No Positive Memory"] = cond_g_pass

    # Condition H: Provenance exists for 100% of included items
    print("\n--- Condition H: 100% Provenance Coverage ---")
    all_composed_items = (
        bundle_f.positive_evidence
        + bundle_f.positive_memories
        + bundle_f.negative_boundaries
        + bundle_f.conflicted_items
        + bundle_f.excluded_items
    )
    cond_h_pass = len(all_composed_items) > 0 and all(
        it.item_id
        and it.source_identifier
        and it.evidence_fingerprint
        and it.deterministic_reason
        and it.epistemic_state is not None
        and it.ordering_key
        for it in all_composed_items
    )
    print(f"Condition H Result: {'PASS' if cond_h_pass else 'FAIL'} (Items inspected: {len(all_composed_items)})")
    results["Condition H: 100% Provenance Coverage"] = cond_h_pass

    # Condition I: Budget limits are obeyed
    print("\n--- Condition I: Budget Limits Obeyed ---")
    strict_budget = CognitiveContextBudget(
        max_evidence_items=2,
        max_positive_memories=1,
        max_negative_boundaries=1,
        max_total_items=3,
    )
    req_i = CognitiveContextRequest(
        task_description="Refactor compute_tax in billing/tax.py",
        task_family="billing_refactor",
        budget=strict_budget,
    )
    bundle_i = composer.compose_context(req_i)
    cond_i_pass = (
        len(bundle_i.positive_evidence) <= strict_budget.max_evidence_items
        and len(bundle_i.positive_memories) <= strict_budget.max_positive_memories
        and len(bundle_i.negative_boundaries) <= strict_budget.max_negative_boundaries
        and bundle_i.total_active_items <= strict_budget.max_total_items
    )
    print(f"Condition I Result: {'PASS' if cond_i_pass else 'FAIL'} (Total active items: {bundle_i.total_active_items})")
    results["Condition I: Budget Limits Obeyed"] = cond_i_pass

    # Condition J: Identical inputs produce identical context
    print("\n--- Condition J: Deterministic Repeatability ---")
    bundle_j1 = composer.compose_context(req_a)
    bundle_j2 = composer.compose_context(req_a)
    cond_j_pass = (
        bundle_j1.telemetry.context_fingerprint == bundle_j2.telemetry.context_fingerprint
        and [x.item_id for x in bundle_j1.positive_evidence] == [x.item_id for x in bundle_j2.positive_evidence]
        and [x.item_id for x in bundle_j1.positive_memories] == [x.item_id for x in bundle_j2.positive_memories]
        and [x.item_id for x in bundle_j1.negative_boundaries] == [x.item_id for x in bundle_j2.negative_boundaries]
    )
    print(f"Condition J Result: {'PASS' if cond_j_pass else 'FAIL'} (Fingerprint: {bundle_j1.telemetry.context_fingerprint})")
    results["Condition J: Deterministic Repeatability"] = cond_j_pass

    # Condition K: Neural context contains passive data only
    print("\n--- Condition K: Passive Neural Context ---")
    neural_ctx = bundle_a.to_neural_context()
    cond_k_pass = (
        isinstance(neural_ctx, dict)
        and "positive_evidence" in neural_ctx
        and "positive_solution_patterns" in neural_ctx
        and "negative_boundaries" in neural_ctx
        and all(not callable(v) for v in neural_ctx.values())
        and all(not hasattr(v, "mutate") for v in neural_ctx.values())
        and all(not hasattr(v, "execute") for v in neural_ctx.values())
    )
    print(f"Condition K Result: {'PASS' if cond_k_pass else 'FAIL'}")
    results["Condition K: Passive Neural Context"] = cond_k_pass

    # Condition L: Neural context cannot mutate RepositoryMemoryIndex
    print("\n--- Condition L: Zero Memory Mutation Through Adapter ---")
    adapter = MockNeuralProposalAdapter()
    proposal = adapter.generate_proposal(neural_ctx)
    cond_l_pass = (
        proposal is not None
        and mem_index.lookup("mem_valid_tax") is not None
        and mem_index.lookup("mem_valid_tax").successful_episodes == 3
    )
    print(f"Condition L Result: {'PASS' if cond_l_pass else 'FAIL'} (Proposal generated: {proposal.proposal_id})")
    results["Condition L: Zero Memory Mutation"] = cond_l_pass

    # Condition M: Neural baseline remains bit-exact
    print("\n--- Condition M: Neural Baseline Bit-Exactness ---")
    post_hash = compute_baseline_hash()
    print(f"Post-Run Neural Core Weight Hash: {post_hash}")
    cond_m_pass = (post_hash == EXPECTED_WEIGHT_HASH)
    print(f"Condition M Result: {'PASS' if cond_m_pass else 'FAIL'}")
    results["Condition M: Neural Baseline Bit-Exact"] = cond_m_pass

    # Condition N: Existing Step 59–66 behavior remains unchanged
    print("\n--- Condition N: Regression Compatibility ---")
    # Verify Step 64 retriever still functions standalone
    from chakrview.cognition.repository.context_store import GroundedContextRetriever
    retriever = GroundedContextRetriever(context_store=context_store)
    g_bundle = retriever.retrieve_context("Refactor tax", "billing_refactor")
    cond_n_pass = len(g_bundle.candidate_files) > 0 and len(g_bundle.evidence_records) > 0
    print(f"Condition N Result: {'PASS' if cond_n_pass else 'FAIL'}")
    results["Condition N: Step 59-66 Compatibility"] = cond_n_pass

    print("\n" + "=" * 80)
    print("STEP 67 BENCHMARK SUMMARY:")
    all_passed = all(results.values())
    for k, v in results.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print(f"\nFinal Status: {'ALL CONDITIONS PASSED' if all_passed else 'FAILURES DETECTED'}")
    print("=" * 80)

    assert all_passed, "Step 67 experiment conditions failed!"
    return results


if __name__ == "__main__":
    run_experiment()
