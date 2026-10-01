from __future__ import annotations

import pytest
from pathlib import Path
import tempfile
import shutil

from chakrview.arena.models import ProjectSpecification, ProjectManifest, SourceFile, FileRole
from chakrview.cognition.repository.state import RepositoryState
from chakrview.cognition.repository.context_store import RepositoryContextStore, EpistemicState
from chakrview.cognition.repository.patch_planning import (
    PatchOperationType,
    PatchOperation,
    PatchPlan,
    PatchPlanStatus,
    PatchPlanner,
    PatchPlanValidator,
    sanitize_relative_path,
    generate_deterministic_diff,
)
from chakrview.cognition.repository.neural_proposal import (
    ProposalContract,
    ProposalValidationStatus,
)


@pytest.fixture
def sample_repo_state_and_store():
    manifest = ProjectManifest(
        specification=ProjectSpecification(
            project_id="test_plan_repo",
            project_name="Test Plan Repo",
            description="Testing deterministic patch planning",
            entrypoint="calculator.py",
        ),
        files=[
            SourceFile(
                path="calculator.py",
                content="def add(a, b):\n    return a + b\n\ndef multiply(a, b):\n    return a * b\n",
                role=FileRole.SOURCE,
            ),
            SourceFile(
                path="utils.py",
                content="DEFAULT_TIMEOUT = 30\n",
                role=FileRole.SOURCE,
            ),
        ],
    )
    repo_state = RepositoryState.from_manifest(manifest)
    store = RepositoryContextStore(project_id="test_plan_repo")
    store.synchronize(manifest)
    
    yield repo_state, store


@pytest.fixture
def valid_accepted_proposal(sample_repo_state_and_store):
    repo_state, _ = sample_repo_state_and_store
    return ProposalContract(
        proposal_id="prop_calc_01",
        task_id="task_calc_fix",
        proposal_type="BUG_FIX",
        summary="Fix precision in multiply",
        proposed_changes={
            "calculator.py": "def add(a, b):\n    return a + b\n\ndef multiply(a, b):\n    return round(a * b, 4)\n",
        },
        target_files=("calculator.py",),
        target_symbols=("multiply",),
        reasoning_trace_summary="Multiply needed rounding.",
        supporting_evidence_ids=("ev_01", "ev_02"),
        supporting_memory_ids=("mem_01",),
        negative_boundary_ids=("neg_01",),
        confidence=0.95,
        epistemic_state=EpistemicState.KNOWN,
        context_fingerprint="ctx_fp_0123456789abcdef",
        validation_status=ProposalValidationStatus.ACCEPTED_FOR_EXECUTION_REVIEW,
    )


class TestPatchPlanning:
    def test_path_sanitization(self):
        # Valid relative paths
        assert sanitize_relative_path("src/calc.py") == "src/calc.py"
        assert sanitize_relative_path("a/b/c.txt") == "a/b/c.txt"
        
        # Traversal and escapes
        with pytest.raises(ValueError, match="Path traversal"):
            sanitize_relative_path("../secret.py")
        with pytest.raises(ValueError, match="Path traversal"):
            sanitize_relative_path("src/../../secret.py")
        with pytest.raises(ValueError, match="Absolute paths"):
            sanitize_relative_path("/etc/passwd")
        with pytest.raises(ValueError, match="Absolute paths"):
            sanitize_relative_path("C:/Windows/System32")
        with pytest.raises(ValueError, match="UNC paths"):
            sanitize_relative_path("//server/share/file.py")
        with pytest.raises(ValueError, match="Empty"):
            sanitize_relative_path("")

    def test_deterministic_diff_generation(self):
        old_content = "def hello():\n    return 'hi'\n"
        new_content = "def hello():\n    return 'hello world'\n"
        diff1 = generate_deterministic_diff("src/hello.py", old_content, new_content)
        diff2 = generate_deterministic_diff("src/hello.py", old_content, new_content)
        
        assert diff1 == diff2
        assert "--- a/src/hello.py" in diff1
        assert "+++ b/src/hello.py" in diff1
        assert "-    return 'hi'" in diff1
        assert "+    return 'hello world'" in diff1

    def test_plan_creation_from_valid_proposal(self, sample_repo_state_and_store, valid_accepted_proposal):
        repo_state, _ = sample_repo_state_and_store
        plan = PatchPlanner.create_plan(
            proposal=valid_accepted_proposal,
            repo_state=repo_state,
            allowed_files={"calculator.py", "utils.py"},
        )
        
        assert plan.status == PatchPlanStatus.PLANNED
        assert plan.target_files == ("calculator.py",)
        assert len(plan.operations) == 1
        assert plan.operations[0].operation_type == PatchOperationType.MODIFY_FILE
        assert len(plan.plan_fingerprint) == 16
        assert plan.operations[0].provenance_evidence_ids == ("ev_01", "ev_02")
        assert plan.operations[0].provenance_memory_ids == ("mem_01",)
        assert plan.operations[0].provenance_negative_boundary_ids == ("neg_01",)

    def test_identical_inputs_identical_fingerprints(self, sample_repo_state_and_store, valid_accepted_proposal):
        repo_state, _ = sample_repo_state_and_store
        plan1 = PatchPlanner.create_plan(valid_accepted_proposal, repo_state)
        plan2 = PatchPlanner.create_plan(valid_accepted_proposal, repo_state)
        
        assert plan1.plan_fingerprint == plan2.plan_fingerprint
        assert plan1.operations[0].operation_fingerprint == plan2.operations[0].operation_fingerprint
        assert plan1.unified_diff == plan2.unified_diff

    def test_changed_content_changes_fingerprint(self, sample_repo_state_and_store, valid_accepted_proposal):
        repo_state, _ = sample_repo_state_and_store
        plan1 = PatchPlanner.create_plan(valid_accepted_proposal, repo_state)
        
        # Modify proposed edits in a new proposal
        proposal2 = ProposalContract(
            proposal_id=valid_accepted_proposal.proposal_id,
            task_id=valid_accepted_proposal.task_id,
            proposal_type=valid_accepted_proposal.proposal_type,
            summary=valid_accepted_proposal.summary,
            proposed_changes={
                "calculator.py": "def add(a, b): return a + b\ndef multiply(a, b): return int(a * b)\n",
            },
            target_files=("calculator.py",),
            target_symbols=("multiply",),
            reasoning_trace_summary="Different logic",
            supporting_evidence_ids=valid_accepted_proposal.supporting_evidence_ids,
            supporting_memory_ids=valid_accepted_proposal.supporting_memory_ids,
            negative_boundary_ids=valid_accepted_proposal.negative_boundary_ids,
            confidence=0.90,
            epistemic_state=EpistemicState.KNOWN,
            context_fingerprint=valid_accepted_proposal.context_fingerprint,
            validation_status=ProposalValidationStatus.ACCEPTED_FOR_EXECUTION_REVIEW,
        )
        plan2 = PatchPlanner.create_plan(proposal2, repo_state)
        
        assert plan1.plan_fingerprint != plan2.plan_fingerprint
        assert plan1.operations[0].operation_fingerprint != plan2.operations[0].operation_fingerprint

    def test_fail_closed_on_rejected_or_conflicted_proposal(self, sample_repo_state_and_store, valid_accepted_proposal):
        repo_state, _ = sample_repo_state_and_store
        
        # Rejected proposal
        prop_rej = ProposalContract(
            proposal_id="prop_rej",
            task_id=valid_accepted_proposal.task_id,
            proposal_type="BUG_FIX",
            summary="Rejected fix",
            proposed_changes={"calculator.py": "pass\n"},
            target_files=("calculator.py",),
            target_symbols=("multiply",),
            reasoning_trace_summary="Rejected reason",
            supporting_evidence_ids=(),
            supporting_memory_ids=(),
            negative_boundary_ids=(),
            confidence=0.1,
            epistemic_state=EpistemicState.UNKNOWN,
            context_fingerprint="ctx_fp",
            validation_status=ProposalValidationStatus.REJECTED,
        )
        plan_rej = PatchPlanner.create_plan(prop_rej, repo_state)
        assert plan_rej.status == PatchPlanStatus.REJECTED
        assert len(plan_rej.operations) == 0

        # Conflicted proposal
        prop_conf = ProposalContract(
            proposal_id="prop_conf",
            task_id=valid_accepted_proposal.task_id,
            proposal_type="BUG_FIX",
            summary="Conflicted fix",
            proposed_changes={"calculator.py": "pass\n"},
            target_files=("calculator.py",),
            target_symbols=("multiply",),
            reasoning_trace_summary="Conflicted reason",
            supporting_evidence_ids=(),
            supporting_memory_ids=(),
            negative_boundary_ids=(),
            confidence=0.5,
            epistemic_state=EpistemicState.UNKNOWN,
            context_fingerprint="ctx_fp",
            validation_status=ProposalValidationStatus.CONFLICTED,
        )
        plan_conf = PatchPlanner.create_plan(prop_conf, repo_state)
        assert plan_conf.status == PatchPlanStatus.CONFLICTED
        assert len(plan_conf.operations) == 0

    def test_fail_closed_on_unallowed_file(self, sample_repo_state_and_store, valid_accepted_proposal):
        repo_state, _ = sample_repo_state_and_store
        plan = PatchPlanner.create_plan(
            valid_accepted_proposal,
            repo_state,
            allowed_files={"utils.py"},  # calculator.py is excluded
        )
        assert plan.status == PatchPlanStatus.REJECTED
        assert any("outside allowed_files" in r for r in plan.validation_reasons)

    def test_independent_validator_approves_valid_plan(self, sample_repo_state_and_store, valid_accepted_proposal):
        repo_state, store = sample_repo_state_and_store
        plan = PatchPlanner.create_plan(
            valid_accepted_proposal,
            repo_state,
            allowed_files={"calculator.py", "utils.py"},
        )
        
        validated = PatchPlanValidator.validate_plan(
            plan=plan,
            context_store=store,
            current_repo_state=repo_state,
            allowed_files={"calculator.py", "utils.py"},
        )
        
        assert validated.status == PatchPlanStatus.READY_FOR_EXECUTION_REVIEW

    def test_independent_validator_rejects_dangerous_keywords(self, sample_repo_state_and_store, valid_accepted_proposal):
        repo_state, store = sample_repo_state_and_store
        dangerous_prop = ProposalContract(
            proposal_id="prop_danger",
            task_id="task_calc_fix",
            proposal_type="BUG_FIX",
            summary="Dangerous command injection",
            proposed_changes={
                "calculator.py": "import os\ndef multiply(a, b):\n    os.system('calc.exe')\n    return a * b\n",
            },
            target_files=("calculator.py",),
            target_symbols=("multiply",),
            reasoning_trace_summary="Dangerous injection",
            supporting_evidence_ids=("ev_01",),
            supporting_memory_ids=("mem_01",),
            negative_boundary_ids=(),
            confidence=0.8,
            epistemic_state=EpistemicState.KNOWN,
            context_fingerprint="ctx_fp",
            validation_status=ProposalValidationStatus.ACCEPTED_FOR_EXECUTION_REVIEW,
        )
        plan = PatchPlanner.create_plan(dangerous_prop, repo_state)
        validated = PatchPlanValidator.validate_plan(
            plan=plan,
            context_store=store,
            current_repo_state=repo_state,
        )
        assert validated.status == PatchPlanStatus.REJECTED
        assert any("Prohibited execution handle" in r for r in validated.validation_reasons)
