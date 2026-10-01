from __future__ import annotations

import pytest
from pathlib import Path
import tempfile
import shutil

from chakrview.arena.models import ProjectSpecification, ProjectManifest, SourceFile, FileRole
from chakrview.arena.workspace import IsolatedWorkspace
from chakrview.cognition.repository.state import RepositoryState
from chakrview.cognition.repository.context_store import RepositoryContextStore, EpistemicState
from chakrview.cognition.repository.patch_planning import (
    PatchPlanner,
    PatchPlanValidator,
    PatchPlanStatus,
)
from chakrview.cognition.repository.patch_execution import (
    SafePatchExecutor,
    PatchExecutionRequest,
    PatchExecutionStatus,
    PatchExecutionResult,
)
from chakrview.cognition.repository.neural_proposal import (
    ProposalContract,
    ProposalValidationStatus,
)


@pytest.fixture
def workspace_and_plan():
    manifest = ProjectManifest(
        specification=ProjectSpecification(
            project_id="test_exec_repo",
            project_name="Test Exec Repo",
            description="Testing patch execution",
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
    workspace = IsolatedWorkspace(project_manifest=manifest)
    repo_state = RepositoryState.from_manifest(manifest)
    
    store = RepositoryContextStore(project_id="test_exec_repo")
    store.synchronize(manifest)
    
    proposal = ProposalContract(
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
        negative_boundary_ids=(),
        confidence=0.95,
        epistemic_state=EpistemicState.KNOWN,
        context_fingerprint="ctx_fp_0123456789abcdef",
        validation_status=ProposalValidationStatus.ACCEPTED_FOR_EXECUTION_REVIEW,
    )
    
    plan = PatchPlanner.create_plan(proposal=proposal, repo_state=repo_state)
    validated_plan = PatchPlanValidator.validate_plan(
        plan=plan,
        context_store=store,
        current_repo_state=repo_state,
    )
    assert validated_plan.status == PatchPlanStatus.READY_FOR_EXECUTION_REVIEW
    
    yield workspace, store, validated_plan
    
    workspace.cleanup()


class TestPatchExecution:
    def test_execution_requires_explicit_approval(self, workspace_and_plan):
        workspace, store, plan = workspace_and_plan
        executor = SafePatchExecutor(workspace=workspace, context_store=store)
        
        # Request with approved=False
        unapproved_req = PatchExecutionRequest(
            plan=plan,
            approved=False,
            approval_token="",
        )
        res = executor.execute_patch(unapproved_req)
        assert res.status == PatchExecutionStatus.READY_FOR_EXECUTION_REVIEW
        assert res.is_applied is False
        assert any("explicit approval" in m for m in res.messages)

    def test_approved_execution_succeeds_and_verifies(self, workspace_and_plan):
        workspace, store, plan = workspace_and_plan
        executor = SafePatchExecutor(workspace=workspace, context_store=store)
        
        # Original content of unrelated file
        utils_initial = workspace.get_source_file_path("utils.py").read_text(encoding="utf-8")
        
        req = PatchExecutionRequest(
            plan=plan,
            approved=True,
            approval_token="USER_TOKEN_OK",
        )
        res = executor.execute_patch(req)
        
        assert res.status == PatchExecutionStatus.VERIFIED
        assert res.is_applied is True
        assert res.is_verified is True
        assert "calculator.py" in res.applied_files
        assert res.rollback_record is not None
        
        # Verify content modified in calculator.py
        calc_content = workspace.get_source_file_path("calculator.py").read_text(encoding="utf-8")
        assert "round(a * b, 4)" in calc_content
        
        # Verify unrelated file completely unchanged
        utils_after = workspace.get_source_file_path("utils.py").read_text(encoding="utf-8")
        assert utils_after == utils_initial

    def test_rollback_restores_original_state(self, workspace_and_plan):
        workspace, store, plan = workspace_and_plan
        executor = SafePatchExecutor(workspace=workspace, context_store=store)
        
        orig_calc = workspace.get_source_file_path("calculator.py").read_text(encoding="utf-8")
        
        req = PatchExecutionRequest(
            plan=plan,
            approved=True,
            approval_token="USER_TOKEN_OK",
        )
        res = executor.execute_patch(req)
        assert res.is_applied is True
        
        # Rollback
        success = executor.rollback(plan.plan_id)
        assert success is True
        
        restored_calc = workspace.get_source_file_path("calculator.py").read_text(encoding="utf-8")
        assert restored_calc == orig_calc

    def test_stale_repository_detected_before_write(self, workspace_and_plan):
        workspace, store, plan = workspace_and_plan
        executor = SafePatchExecutor(workspace=workspace, context_store=store)
        
        # Mutate the repository out from under the plan
        workspace.write_source_file("other.py", "x = 99\n")
        
        req = PatchExecutionRequest(
            plan=plan,
            approved=True,
            approval_token="USER_TOKEN_OK",
        )
        res = executor.execute_patch(req)
        
        assert res.status == PatchExecutionStatus.STALE
        assert res.is_applied is False
        assert any("Stale repository state detected" in m for m in res.messages)
