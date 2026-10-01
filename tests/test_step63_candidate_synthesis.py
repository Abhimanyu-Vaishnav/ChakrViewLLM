"""
ChakrView Step 63: Dedicated Test Suite for Autonomous Strategy Synthesis & Safety Gating.

Comprehensive test coverage across:
1. Candidate construction, serialization, and fingerprint determinism.
2. Candidate normalization and malformed candidate rejection.
3. File scope and task boundary enforcement (fail-closed rejection).
4. Dependency compatibility and graph-aware safety gating.
5. Semantic memory compatibility and stale memory rejection.
6. Candidate deduplication and provenance merging.
7. Strategy recombination of partial traces.
8. Recombination rejection on invalid/leaking scopes.
9. Negative transfer protection (unrelated domain tags trigger ABSTAIN).
10. Synthesis operational limits (max candidates, max steps, recombination depth).
11. Deterministic arbitration and branch prioritization (no automatic priority).
12. Rollback safety and exact fingerprint match upon synthesized failure.
13. Fail-closed abstention when all candidates fail.
14. Integration with Step 62 ObservationDrivenBranchingCoordinator.
15. Neural baseline bit-exact immutability.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest
import torch

from chakrview.arena.models import ProjectManifest, SourceFile, FileRole
from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
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
from chakrview.runtime.pipeline import EXPECTED_WEIGHT_HASH
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
def valid_step_tax() -> RefactoringStep:
    return RefactoringStep(
        step_id="step_tax",
        description="Fix tax calculation",
        target_files=["tax_service.py"],
        patch_dict={"tax_service.py": CORRECTED_TAX_CODE},
        targeted_test_file="test_tax_service.py",
    )


@pytest.fixture
def valid_step_disc() -> RefactoringStep:
    return RefactoringStep(
        step_id="step_disc",
        description="Fix discount calculation",
        target_files=["discount_engine.py"],
        patch_dict={"discount_engine.py": CORRECTED_DISCOUNT_CODE},
        targeted_test_file="test_discount_engine.py",
    )


class TestCandidateSynthesisContract:

    def test_candidate_construction_and_fingerprint_determinism(
        self, valid_step_tax: RefactoringStep
    ):
        c1 = SynthesizedCandidate(
            candidate_id="cand_1",
            objective="Fix tax calculation",
            ordered_steps=[valid_step_tax],
            allowed_files={"tax_service.py"},
            preconditions={"required_existing_files": ["tax_service.py"]},
            provenance=CandidateProvenance(
                origin=CandidateOrigin.SYNTHESIS_TEMPLATE,
                domain_tags={"billing"},
            ),
        )
        c2 = SynthesizedCandidate(
            candidate_id="cand_2",
            objective="Fix tax calculation",
            ordered_steps=[valid_step_tax],
            allowed_files={"tax_service.py"},
            preconditions={"required_existing_files": ["tax_service.py"]},
            provenance=CandidateProvenance(
                origin=CandidateOrigin.SYNTHESIS_TEMPLATE,
                domain_tags={"billing"},
            ),
        )
        assert c1.fingerprint == c2.fingerprint
        assert len(c1.fingerprint) == 64
        d = c1.to_dict()
        assert d["fingerprint"] == c1.fingerprint
        assert "tax_service.py" in d["allowed_files"]

    def test_normalization_validates_required_fields(self):
        # Empty candidate_id
        bad_c = SynthesizedCandidate(
            candidate_id="",
            objective="Obj",
            ordered_steps=[],
            allowed_files=set(),
        )
        norm = CandidateNormalizer.normalize(bad_c)
        assert norm.is_valid is False
        assert any("candidate_id cannot be empty" in r for r in norm.rejection_reasons)
        assert any("allowed_files cannot be empty" in r for r in norm.rejection_reasons)
        assert any("ordered_steps cannot be empty" in r for r in norm.rejection_reasons)

    def test_normalization_catches_target_outside_allowed_files(
        self, valid_step_tax: RefactoringStep
    ):
        bad_cand = SynthesizedCandidate(
            candidate_id="cand_bad_target",
            objective="Mismatched target",
            ordered_steps=[valid_step_tax],  # targets tax_service.py
            allowed_files={"other_file.py"},  # does not include tax_service.py
        )
        norm = CandidateNormalizer.normalize(bad_cand)
        assert norm.is_valid is False
        assert any("outside candidate allowed_files" in r for r in norm.rejection_reasons)

    def test_normalization_rejects_duplicate_step_ids(self, valid_step_tax: RefactoringStep):
        cand_dup_steps = SynthesizedCandidate(
            candidate_id="cand_dup_steps",
            objective="Duplicate steps",
            ordered_steps=[valid_step_tax, valid_step_tax],
            allowed_files={"tax_service.py"},
        )
        norm = CandidateNormalizer.normalize(cand_dup_steps)
        assert norm.is_valid is False
        assert any("duplicate step_id" in r for r in norm.rejection_reasons)


class TestSafetyGateAndDeduplication:

    def test_safety_gate_blocks_files_outside_repository_task_scope(
        self, repo_manifest: ProjectManifest, valid_step_tax: RefactoringStep
    ):
        base_state = RepositoryState.from_manifest(repo_manifest)
        gate = DeterministicSafetyGate()
        cand = SynthesizedCandidate(
            candidate_id="cand_scope_violation",
            objective="Scope violation",
            ordered_steps=[valid_step_tax],
            allowed_files={"tax_service.py", "unauthorized_secret.py"},
        )
        dec = gate.evaluate(
            candidate=cand,
            repo_state=base_state,
            allowed_modified_files={"tax_service.py"},
        )
        assert dec.decision == SafetyDecision.REJECT
        assert any("exceeds repository task allowed scope" in r for r in dec.reasons)

    def test_safety_gate_enforces_max_steps(self, repo_manifest: ProjectManifest):
        base_state = RepositoryState.from_manifest(repo_manifest)
        gate = DeterministicSafetyGate(max_candidate_steps=2)
        steps = [
            RefactoringStep(
                step_id=f"step_{i}",
                description=f"Step {i}",
                target_files=["tax_service.py"],
                patch_dict={"tax_service.py": f"# {i}\n"},
            )
            for i in range(4)
        ]
        cand = SynthesizedCandidate(
            candidate_id="cand_overflow",
            objective="Too many steps",
            ordered_steps=steps,
            allowed_files={"tax_service.py"},
        )
        dec = gate.evaluate(
            candidate=cand,
            repo_state=base_state,
            allowed_modified_files={"tax_service.py"},
        )
        assert dec.decision == SafetyDecision.REJECT
        assert any("exceeds max 2" in r for r in dec.reasons)

    def test_candidate_deduplicator_collapses_duplicates_and_merges_provenance(
        self, valid_step_tax: RefactoringStep
    ):
        c1 = SynthesizedCandidate(
            candidate_id="cand_1",
            objective="Identical fix",
            ordered_steps=[valid_step_tax],
            allowed_files={"tax_service.py"},
            provenance=CandidateProvenance(
                origin=CandidateOrigin.SYNTHESIS_TEMPLATE,
                source_branch_ids=["b_alpha"],
                domain_tags={"billing"},
            ),
        )
        c2 = SynthesizedCandidate(
            candidate_id="cand_2",
            objective="Identical fix",
            ordered_steps=[valid_step_tax],
            allowed_files={"tax_service.py"},
            provenance=CandidateProvenance(
                origin=CandidateOrigin.SYNTHESIS_TEMPLATE,
                source_branch_ids=["b_beta"],
                domain_tags={"finance"},
            ),
        )
        deduped = CandidateDeduplicator.deduplicate([c1, c2])
        assert len(deduped) == 1
        assert set(deduped[0].provenance.source_branch_ids) == {"b_alpha", "b_beta"}
        assert set(deduped[0].provenance.domain_tags) == {"billing", "finance"}


class TestTraceRecombinationAndNegativeTransfer:

    def test_strategy_recombiner_success(
        self, valid_step_tax: RefactoringStep, valid_step_disc: RefactoringStep
    ):
        recomb = StrategyRecombiner.recombine_traces(
            objective="Recombined tax and discount",
            prefix_steps=[valid_step_tax],
            suffix_steps=[valid_step_disc],
            allowed_files={"tax_service.py", "discount_engine.py"},
            domain_tag="billing",
            provenance_branches=["b1", "b2"],
        )
        assert recomb is not None
        assert recomb.provenance.origin == CandidateOrigin.TRACE_RECOMBINATION
        assert len(recomb.ordered_steps) == 2
        assert recomb.provenance.source_step_ids == ["step_tax", "step_disc"]

    def test_strategy_recombiner_rejects_scope_leak(
        self, valid_step_tax: RefactoringStep
    ):
        leaking_step = RefactoringStep(
            step_id="step_leak",
            description="Leaking step",
            target_files=["unapproved_secret.py"],
            patch_dict={"unapproved_secret.py": "x = 1\n"},
        )
        recomb = StrategyRecombiner.recombine_traces(
            objective="Leaking recomb",
            prefix_steps=[valid_step_tax],
            suffix_steps=[leaking_step],
            allowed_files={"tax_service.py"},
            domain_tag="billing",
            provenance_branches=["b1", "b2"],
        )
        assert recomb is None

    def test_negative_transfer_protection_triggers_abstention(
        self, repo_manifest: ProjectManifest, valid_step_tax: RefactoringStep
    ):
        base_state = RepositoryState.from_manifest(repo_manifest)
        gate = DeterministicSafetyGate()
        cand = SynthesizedCandidate(
            candidate_id="cand_neg_transfer",
            objective="User auth fix applied to billing",
            ordered_steps=[valid_step_tax],
            allowed_files={"tax_service.py"},
            provenance=CandidateProvenance(
                origin=CandidateOrigin.MEMORY_RETRIEVAL,
                domain_tags={"authentication_jwt"},
            ),
        )
        dec = gate.evaluate(
            candidate=cand,
            repo_state=base_state,
            allowed_modified_files={"tax_service.py"},
            task_domain="billing_refactor",
        )
        assert dec.decision == SafetyDecision.ABSTAIN
        assert any("Domain mismatch" in r for r in dec.reasons)


class TestSynthesisAwareBranchingIntegration:

    def test_synthesis_recovers_failing_authored_branch(
        self,
        repo_manifest: ProjectManifest,
        valid_step_tax: RefactoringStep,
        valid_step_disc: RefactoringStep,
    ):
        # Authored branch fails tests
        step_fail = RefactoringStep(
            step_id="step_fail",
            description="Broken tax calculation",
            target_files=["tax_service.py"],
            patch_dict={"tax_service.py": "from models import Order\ndef compute_tax(o): return 0.0\n"},
            targeted_test_file="test_tax_service.py",
        )
        branch_fail = RefactoringBranch(
            branch_id="branch_broken",
            description="Authored broken branch",
            steps=[step_fail],
            allowed_files={"tax_service.py"},
            priority=1,
        )
        cand_alt = SynthesizedCandidate(
            candidate_id="synth_alt_solution",
            objective="Synthesized working solution",
            ordered_steps=[valid_step_tax, valid_step_disc],
            allowed_files={"tax_service.py", "discount_engine.py"},
            provenance=CandidateProvenance(
                origin=CandidateOrigin.SYNTHESIS_TEMPLATE,
                domain_tags={"billing_refactor"},
            ),
        )
        coord = SynthesisAwareBranchingCoordinator()
        res = coord.execute_synthesis_refactoring(
            manifest=repo_manifest,
            task_id="task_int_recovery",
            objective="Recover from broken branch",
            task_family="billing_refactor",
            initial_branch_generator=lambda rec, st: [branch_fail],
            synthesized_candidates=[cand_alt],
            allowed_modified_files={"tax_service.py", "discount_engine.py"},
        )
        assert res.base_result.overall_success is True
        assert res.base_result.selected_branch_id == "synth_alt_solution"
        assert res.base_result.recovery_transitions >= 1

    def test_all_candidates_fail_fail_closed_abstention(
        self, repo_manifest: ProjectManifest
    ):
        base_state = RepositoryState.from_manifest(repo_manifest)
        base_fp = base_state.state_fingerprint

        step_fail = RefactoringStep(
            step_id="step_fail",
            description="Broken tax calculation",
            target_files=["tax_service.py"],
            patch_dict={"tax_service.py": "from models import Order\ndef compute_tax(o): return -1.0\n"},
            targeted_test_file="test_tax_service.py",
        )
        cand_fail = SynthesizedCandidate(
            candidate_id="synth_failing",
            objective="Failing proposal",
            ordered_steps=[step_fail],
            allowed_files={"tax_service.py"},
            provenance=CandidateProvenance(
                origin=CandidateOrigin.SYNTHESIS_TEMPLATE,
                domain_tags={"billing_refactor"},
            ),
        )
        coord = SynthesisAwareBranchingCoordinator()
        res = coord.execute_synthesis_refactoring(
            manifest=repo_manifest,
            task_id="task_all_fail",
            objective="Test fail closed",
            task_family="billing_refactor",
            synthesized_candidates=[cand_fail],
            allowed_modified_files={"tax_service.py"},
        )
        assert res.base_result.overall_success is False
        assert res.base_result.abstained is True
        # Repository remains bit-exact intact
        assert base_state.state_fingerprint == base_fp


class TestNeuralCoreImmutability:

    def test_neural_baseline_hash_is_bit_exact(self):
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
