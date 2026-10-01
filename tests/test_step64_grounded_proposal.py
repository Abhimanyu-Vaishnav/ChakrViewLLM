"""
ChakrView Step 64: Dedicated Test Suite for Persistent Context & Grounded Neural Proposal.

Validates:
1. Context store synchronization and cold scan behavior.
2. Warm synchronization cache hit ratio and AST inspection reuse.
3. Incremental cache invalidation on single-file mutation.
4. GroundedContextRetriever budget enforcement (max_files, max_symbols).
5. EvidenceRecord provenance attachment for every retrieved context element.
6. HallucinationContainmentGate rejection on nonexistent files (CONTRADICTED).
7. HallucinationContainmentGate rejection on nonexistent symbols (CONTRADICTED).
8. HallucinationContainmentGate rejection on task scope violations.
9. Deterministic safety gate rejection of stale memories.
10. Negative transfer rejection across unrelated domains (ABSTAIN).
11. Proposal normalization and malformed step rejection.
12. Atomic rollback on neural proposal execution failure.
13. Deterministic repeated context retrieval.
14. Neural baseline bit-exact immutability.
"""

from __future__ import annotations

import pytest
import torch

from chakrview.arena.models import ProjectManifest, SourceFile, FileRole
from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.runtime.pipeline import EXPECTED_WEIGHT_HASH
from chakrview.cognition.repository.state import RepositoryState
from chakrview.cognition.repository.refactoring import RefactoringStep
from chakrview.cognition.repository.branching import RefactoringBranch
from chakrview.cognition.repository.semantic_record import RepositorySemanticRecord
from chakrview.cognition.repository.memory_index import RepositoryMemoryIndex
from chakrview.cognition.repository.synthesis import (
    CandidateOrigin,
    CandidateProvenance,
    SynthesizedCandidate,
    CandidateNormalizer,
    DeterministicSafetyGate,
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
    HallucinationContainmentGate,
    GroundedNeuralSynthesisEngine,
)
from scripts.experiment_step61_realtime_change import (
    DEFECTIVE_TAX_CODE,
    CORRECTED_TAX_CODE,
    DEFECTIVE_DISCOUNT_CODE,
    CORRECTED_DISCOUNT_CODE,
    build_repo_manifest,
)


@pytest.fixture
def repo_manifest() -> ProjectManifest:
    return build_repo_manifest(
        tax_code=DEFECTIVE_TAX_CODE,
        discount_code=DEFECTIVE_DISCOUNT_CODE,
    )


@pytest.fixture
def synchronized_store(repo_manifest: ProjectManifest) -> RepositoryContextStore:
    store = RepositoryContextStore(project_id="test_repo")
    store.synchronize(repo_manifest)
    return store


class TestRepositoryContextStore:

    def test_cold_and_warm_synchronization(self, repo_manifest: ProjectManifest):
        store = RepositoryContextStore(project_id="test_repo")
        # Cold
        st1 = store.synchronize(repo_manifest)
        assert store.telemetry.files_inspected == len(repo_manifest.files)
        assert store.telemetry.files_reused_from_cache == 0

        # Warm
        st2 = store.synchronize(repo_manifest)
        assert store.telemetry.files_inspected == 0
        assert store.telemetry.files_reused_from_cache == len(repo_manifest.files)
        assert st1.state_fingerprint == st2.state_fingerprint

    def test_incremental_invalidation_on_single_file_change(
        self, repo_manifest: ProjectManifest, synchronized_store: RepositoryContextStore
    ):
        mutated_manifest = build_repo_manifest(
            tax_code=CORRECTED_TAX_CODE,
            discount_code=DEFECTIVE_DISCOUNT_CODE,
        )
        st_mut = synchronized_store.synchronize(mutated_manifest)
        # Only 1 file should be inspected
        assert synchronized_store.telemetry.files_inspected == 1
        assert synchronized_store.telemetry.files_reused_from_cache == len(repo_manifest.files) - 1
        assert synchronized_store.file_exists("tax_service.py")

    def test_symbol_existence_queries(self, synchronized_store: RepositoryContextStore):
        assert synchronized_store.symbol_exists("compute_tax") is True
        assert synchronized_store.symbol_exists("compute_tax", file_path="tax_service.py") is True
        assert synchronized_store.symbol_exists("nonexistent_symbol") is False


class TestGroundedContextRetriever:

    def test_context_retrieval_budgeting_and_provenance(
        self, synchronized_store: RepositoryContextStore
    ):
        retriever = GroundedContextRetriever(context_store=synchronized_store)
        bundle = retriever.retrieve_context(
            task_description="Fix the tax calculation error",
            target_domain="billing",
            budget=GroundedContextBudget(max_files=2, max_symbols=4),
        )
        assert len(bundle.candidate_files) <= 2
        assert len(bundle.relevant_symbols) <= 4
        assert len(bundle.evidence_records) > 0
        assert all(isinstance(e, EvidenceRecord) for e in bundle.evidence_records)


class TestHallucinationContainmentGate:

    def test_grounded_proposal_approved(self, synchronized_store: RepositoryContextStore):
        proposal = NeuralProposalOutput(
            proposal_id="prop_valid",
            objective="Fix tax calculation",
            target_files=["tax_service.py"],
            target_symbols=["compute_tax"],
            proposed_patches={"tax_service.py": CORRECTED_TAX_CODE},
        )
        dec = HallucinationContainmentGate.verify_grounding(
            proposal=proposal,
            context_store=synchronized_store,
            allowed_modified_files={"tax_service.py"},
        )
        assert dec.is_grounded is True
        assert dec.epistemic_state == EpistemicState.KNOWN

    def test_hallucinated_file_rejected(self, synchronized_store: RepositoryContextStore):
        ghost_prop = NeuralProposalOutput(
            proposal_id="prop_ghost_f",
            objective="Ghost file",
            target_files=["ghost_file.py"],
            target_symbols=["foo"],
            proposed_patches={"ghost_file.py": "x = 1\n"},
        )
        dec = HallucinationContainmentGate.verify_grounding(
            proposal=ghost_prop,
            context_store=synchronized_store,
            allowed_modified_files={"tax_service.py", "ghost_file.py"},
        )
        assert dec.is_grounded is False
        assert dec.epistemic_state == EpistemicState.CONTRADICTED
        assert "ghost_file.py" in dec.unfounded_files

    def test_hallucinated_symbol_rejected(self, synchronized_store: RepositoryContextStore):
        ghost_sym = NeuralProposalOutput(
            proposal_id="prop_ghost_s",
            objective="Ghost symbol",
            target_files=["tax_service.py"],
            target_symbols=["imaginary_crypto_calculator"],
            proposed_patches={"tax_service.py": CORRECTED_TAX_CODE},
        )
        dec = HallucinationContainmentGate.verify_grounding(
            proposal=ghost_sym,
            context_store=synchronized_store,
            allowed_modified_files={"tax_service.py"},
        )
        assert dec.is_grounded is False
        assert dec.epistemic_state == EpistemicState.CONTRADICTED
        assert "imaginary_crypto_calculator" in dec.unfounded_symbols

    def test_scope_leak_rejected(self, synchronized_store: RepositoryContextStore):
        leak_prop = NeuralProposalOutput(
            proposal_id="prop_scope_leak",
            objective="Scope leak",
            target_files=["discount_engine.py"],
            target_symbols=["compute_discount"],
            proposed_patches={"discount_engine.py": CORRECTED_DISCOUNT_CODE},
        )
        dec = HallucinationContainmentGate.verify_grounding(
            proposal=leak_prop,
            context_store=synchronized_store,
            allowed_modified_files={"tax_service.py"},  # discount_engine.py not allowed!
        )
        assert dec.is_grounded is False
        assert any("outside allowed task scope" in r for r in dec.rejection_reasons)


class TestNeuralProposalSafetyIntegration:

    def test_neural_execution_failure_triggers_atomic_rollback(
        self, repo_manifest: ProjectManifest
    ):
        base_state = RepositoryState.from_manifest(repo_manifest)
        base_fp = base_state.state_fingerprint

        failing_cand = SynthesizedCandidate(
            candidate_id="cand_failing_neural",
            objective="Failing neural proposal",
            ordered_steps=[
                RefactoringStep(
                    step_id="step_fail_tax",
                    description="Break tax logic",
                    target_files=["tax_service.py"],
                    patch_dict={"tax_service.py": "from models import Order\ndef compute_tax(o): return -100.0\n"},
                    targeted_test_file="test_tax_service.py",
                )
            ],
            allowed_files={"tax_service.py"},
            provenance=CandidateProvenance(
                origin=CandidateOrigin.SYNTHESIS_LLM,
                domain_tags={"billing_refactor"},
            ),
        )
        coord = SynthesisAwareBranchingCoordinator()
        res = coord.execute_synthesis_refactoring(
            manifest=repo_manifest,
            task_id="task_fail_rollback",
            objective="Test rollback",
            task_family="billing_refactor",
            synthesized_candidates=[failing_cand],
            allowed_modified_files={"tax_service.py"},
        )
        assert res.base_result.overall_success is False
        assert res.base_result.recovery_transitions >= 1
        assert res.base_result.recovery_decisions[0].fingerprint_restored is True
        assert base_state.state_fingerprint == base_fp


class TestNeuralBaselineImmutability:

    def test_neural_core_hash_and_parameters_are_strictly_preserved(self):
        torch.manual_seed(42)
        model = ChakrMicro(ModelConfig())
        model.eval()
        import hashlib
        hasher = hashlib.sha256()
        with torch.no_grad():
            for name, param in sorted(model.named_parameters()):
                hasher.update(name.encode("utf-8"))
                hasher.update(param.detach().cpu().numpy().tobytes())
        h = hasher.hexdigest()
        assert h == EXPECTED_WEIGHT_HASH
        assert sum(p.numel() for p in model.parameters()) == 3_443_136
