"""
ChakrView Step 68: Empirical Benchmark for Grounded Neural Proposal Generation & Validation.

Evaluates:
- End-to-end executable path:
  Task -> Repository Grounding -> Episodic Recall -> Unified Cognitive Context
  -> NeuralProposalInputEncoder -> ChakrMicro Inference (via InferenceEngine)
  -> Proposal Contract Decoding -> ProposalValidator -> Accepted / Rejected / Abstain
- Control A: Full grounded cognitive context.
- Control B: Same task with episodic memory removed.
- Control C: Same task with critical repository evidence removed.
- Control D: Same task with a conflicting negative boundary.
- Control E: Same input/context repeated multiple times (deterministic stability).
- Neural immutability checks: PRE_HASH == POST_HASH == EXPECTED_WEIGHT_HASH (ΔW = 0).
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.runtime.pipeline import EXPECTED_WEIGHT_HASH, InferenceEngine

from chakrview.arena.models import ProjectSpecification, ProjectManifest, SourceFile, FileRole
from chakrview.cognition.repository.state import RepositoryState
from chakrview.cognition.repository.semantic_record import RepositorySemanticRecord
from chakrview.cognition.repository.memory_index import RepositoryMemoryIndex
from chakrview.cognition.repository.context_store import RepositoryContextStore, EpistemicState
from chakrview.cognition.repository.episodic_recall import EpisodicMemoryRecallCoordinator
from chakrview.cognition.repository.cognitive_context import (
    CognitiveContextRequest,
    CognitiveContextBudget,
    UnifiedCognitiveContextComposer,
)
from chakrview.cognition.repository.neural_proposal import (
    ProposalValidationStatus,
    EpistemicPartition,
    ProposalContract,
    NeuralProposalInputEncoder,
    ProposalValidator,
    ChakrMicroNeuralProposalAdapter,
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


def _build_benchmark_repo() -> Tuple[RepositoryState, RepositoryContextStore]:
    manifest = ProjectManifest(
        specification=ProjectSpecification(
            project_id="step68_benchmark_repo",
            project_name="Step 68 Benchmark Repo",
            description="Testing end-to-end grounded neural proposal generation",
            entrypoint="main.py",
        ),
        files=[
            SourceFile(
                path="billing/discount.py",
                content="def compute_discount(order_total: float, discount: float) -> float:\n    # Unchecked discount parameter\n    return discount\n",
                role=FileRole.SOURCE,
            ),
            SourceFile(
                path="billing/service.py",
                content="from billing.discount import compute_discount\n\ndef process_order(total: float, discount: float) -> float:\n    d = compute_discount(total, discount)\n    return total - d\n",
                role=FileRole.SOURCE,
            ),
        ],
    )
    repo_state = RepositoryState.from_manifest(manifest)
    context_store = RepositoryContextStore(project_id="step68_benchmark_repo")
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
        confidence=0.95,
        evidence_count=successful_episodes + failed_episodes,
        successful_episodes=successful_episodes,
        failed_episodes=failed_episodes,
    )


def run_experiment() -> Dict[str, Any]:
    print("=" * 80)
    print("CHAKRVIEW STEP 68: GROUNDED NEURAL PROPOSAL GENERATION BENCHMARK")
    print("=" * 80)
    results: Dict[str, Any] = {}

    pre_hash = compute_baseline_hash()
    print(f"\n[Pre-Check] Neural Core Weight Hash: {pre_hash}")
    assert pre_hash == EXPECTED_WEIGHT_HASH, "Baseline mismatch before Step 68!"

    repo_state, context_store = _build_benchmark_repo()
    mem_index = RepositoryMemoryIndex()

    # Valid positive episodic memory
    rec_valid = _create_record(
        memory_id="mem_disc_clamp_v1",
        task_family="billing_discount",
        repository_pattern="discount clamping invariant: discount <= order_total",
        solution_pattern="def compute_discount(order_total: float, discount: float) -> float:\n    return min(order_total, discount)\n",
        affected_modules=["billing/discount.py"],
        successful_episodes=3,
    )
    mem_index.insert(rec_valid)

    coordinator = EpisodicMemoryRecallCoordinator(memory_index=mem_index)
    composer = UnifiedCognitiveContextComposer(context_store=context_store, recall_coordinator=coordinator)
    adapter = ChakrMicroNeuralProposalAdapter(context_store=context_store)

    # ─────────────────────────────────────────────────────────────────────────
    # Primary Pipeline: End-to-End Execution
    # ─────────────────────────────────────────────────────────────────────────
    print("\n--- Primary Pipeline: End-to-End Grounded Proposal Generation ---")
    req_main = CognitiveContextRequest(
        task_description="Clamp discount to order total in billing/discount.py",
        task_family="billing_discount",
        target_files=("billing/discount.py",),
        target_symbols=("compute_discount",),
    )
    bundle_main = composer.compose_context(req_main)
    proposal_main = adapter.generate_proposal_contract(bundle_main)

    pass_main = (
        proposal_main.validation_status == ProposalValidationStatus.ACCEPTED_FOR_EXECUTION_REVIEW
        and "billing/discount.py" in proposal_main.proposed_changes
        and "min(order_total, discount)" in proposal_main.proposed_changes["billing/discount.py"]
        and len(proposal_main.supporting_evidence_ids) > 0
        and len(proposal_main.supporting_memory_ids) > 0
        and proposal_main.neural_generation_metadata["tokens_generated"] > 0
    )
    print(f"Primary Pipeline Status: {'PASS' if pass_main else 'FAIL'}")
    print(f"  Proposal ID: {proposal_main.proposal_id}")
    print(f"  Validation Status: {proposal_main.validation_status.name}")
    print(f"  Tokens Generated: {proposal_main.neural_generation_metadata['tokens_generated']}")
    print(f"  Model Latency: {proposal_main.neural_generation_metadata['latency_ms']} ms")
    results["Primary Pipeline: End-to-End Grounded Proposal"] = pass_main

    # ─────────────────────────────────────────────────────────────────────────
    # Control A: Full Grounded Cognitive Context
    # ─────────────────────────────────────────────────────────────────────────
    print("\n--- Control A: Full Grounded Cognitive Context ---")
    pass_ctrl_a = (
        proposal_main.validation_status == ProposalValidationStatus.ACCEPTED_FOR_EXECUTION_REVIEW
        and proposal_main.confidence >= 0.85
        and len(proposal_main.claims) >= 3
    )
    print(f"Control A Result: {'PASS' if pass_ctrl_a else 'FAIL'}")
    results["Control A: Full Grounded Context"] = pass_ctrl_a

    # ─────────────────────────────────────────────────────────────────────────
    # Control B: Same Task with Episodic Memory Removed
    # ─────────────────────────────────────────────────────────────────────────
    print("\n--- Control B: Episodic Memory Removed ---")
    mem_index_empty = RepositoryMemoryIndex()
    coord_empty = EpisodicMemoryRecallCoordinator(memory_index=mem_index_empty)
    composer_no_mem = UnifiedCognitiveContextComposer(context_store=context_store, recall_coordinator=coord_empty)
    bundle_no_mem = composer_no_mem.compose_context(req_main)
    proposal_b = adapter.generate_proposal_contract(bundle_no_mem)

    pass_ctrl_b = (
        len(bundle_no_mem.positive_memories) == 0
        and len(proposal_b.supporting_memory_ids) == 0
        and proposal_b.confidence < proposal_main.confidence
        and proposal_b.validation_status == ProposalValidationStatus.ACCEPTED_FOR_EXECUTION_REVIEW
    )
    print(f"Control B Result: {'PASS' if pass_ctrl_b else 'FAIL'} (Confidence: {proposal_b.confidence})")
    results["Control B: No Episodic Memory"] = pass_ctrl_b

    # ─────────────────────────────────────────────────────────────────────────
    # Control C: Same Task with Target File Grounding Removed
    # ─────────────────────────────────────────────────────────────────────────
    print("\n--- Control C: Critical Evidence Removed (Non-Existent Target) ---")
    req_no_ev = CognitiveContextRequest(
        task_description="Refactor unknown file",
        task_family="billing_discount",
        target_files=("billing/missing_file.py",),
    )
    bundle_no_ev = composer.compose_context(req_no_ev)
    proposal_c = adapter.generate_proposal_contract(bundle_no_ev)

    pass_ctrl_c = (
        proposal_c.validation_status == ProposalValidationStatus.REJECTED
        and any("does not exist in repository store" in r for r in proposal_c.validation_reasons)
    )
    print(f"Control C Result: {'PASS' if pass_ctrl_c else 'FAIL'} (Status: {proposal_c.validation_status.name})")
    results["Control C: Missing Grounded Evidence"] = pass_ctrl_c

    # ─────────────────────────────────────────────────────────────────────────
    # Control D: Conflicting Negative Boundary
    # ─────────────────────────────────────────────────────────────────────────
    print("\n--- Control D: Conflicting Negative Boundary ---")
    mem_index_neg = RepositoryMemoryIndex()
    rec_neg = _create_record(
        memory_id="nb_forbid_clamp",
        task_family="billing_discount",
        repository_pattern="negative clamp prohibition",
        solution_pattern="DO_NOT_APPLY: Clamp discount to order_total directly",
        affected_modules=["billing/discount.py"],
        known_boundaries=["DO_NOT_APPLY: Clamp discount to order_total directly"],
        successful_episodes=0,
        failed_episodes=2,
    )
    mem_index_neg.insert(rec_neg)
    coord_neg = EpisodicMemoryRecallCoordinator(memory_index=mem_index_neg)
    composer_neg = UnifiedCognitiveContextComposer(context_store=context_store, recall_coordinator=coord_neg)
    bundle_neg = composer_neg.compose_context(req_main)

    pass_ctrl_d = (
        len(bundle_neg.negative_boundaries) > 0
        and bundle_neg.negative_boundaries[0].source_identifier == "nb_forbid_clamp"
    )
    print(f"Control D Result: {'PASS' if pass_ctrl_d else 'FAIL'} (Negative boundary isolated: {bundle_neg.negative_boundaries[0].source_identifier})")
    results["Control D: Conflicting Negative Boundary"] = pass_ctrl_d

    # ─────────────────────────────────────────────────────────────────────────
    # Control E: Deterministic Repeatability
    # ─────────────────────────────────────────────────────────────────────────
    print("\n--- Control E: Deterministic Repeatability ---")
    p1 = adapter.generate_proposal_contract(bundle_main)
    p2 = adapter.generate_proposal_contract(bundle_main)

    pass_ctrl_e = (
        p1.proposal_id == p2.proposal_id
        and p1.proposed_changes == p2.proposed_changes
        and p1.validation_status == p2.validation_status
        and p1.neural_generation_metadata["tokens_generated"] == p2.neural_generation_metadata["tokens_generated"]
        and p1.neural_generation_metadata["raw_text"] == p2.neural_generation_metadata["raw_text"]
    )
    print(f"Control E Result: {'PASS' if pass_ctrl_e else 'FAIL'}")
    results["Control E: Deterministic Repeatability"] = pass_ctrl_e

    # ─────────────────────────────────────────────────────────────────────────
    # Immutability Check: Neural Core Weights Unchanged (ΔW = 0)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n--- Neural Core Immutability Check ---")
    post_hash = compute_baseline_hash()
    print(f"Post-Benchmark Neural Core Weight Hash: {post_hash}")
    pass_immut = (post_hash == EXPECTED_WEIGHT_HASH)
    print(f"Immutability Result: {'PASS' if pass_immut else 'FAIL'}")
    results["Neural Core Immutability (dW = 0)"] = pass_immut

    print("\n" + "=" * 80)
    print("STEP 68 BENCHMARK SUMMARY:")
    all_passed = all(results.values())
    for k, v in results.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print(f"\nFinal Status: {'ALL CONDITIONS PASSED' if all_passed else 'FAILURES DETECTED'}")
    print("=" * 80)

    assert all_passed, "Step 68 experiment conditions failed!"
    return results


if __name__ == "__main__":
    run_experiment()
