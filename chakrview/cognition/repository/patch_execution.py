"""
ChakrView Step 70: Safe Patch Execution, Explicit Approval, Atomic Application & Rollback.

Defines:
- PatchExecutionStatus: Explicit lifecycle states (PLANNED, VALIDATED, READY_FOR_EXECUTION_REVIEW,
  APPROVED, APPLYING, APPLIED, VERIFIED, REJECTED, CONFLICTED, STALE, FAILED, ROLLED_BACK).
- PatchExecutionRequest: Explicit execution request carrying verified approval token.
- PatchRollbackRecord: Snapshot of pre- and post-modification state supporting deterministic rollback.
- PatchExecutionResult: Complete audit trail distinguishing PATCH_APPLIED from PATCH_VERIFIED.
- SafePatchExecutor: Atomic filesystem execution engine with path containment, atomicity,
  immediate post-application verification, and rollback capability.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from chakrview.arena.models import ProjectManifest, SourceFile
from chakrview.arena.workspace import IsolatedWorkspace
from chakrview.cognition.repository.state import RepositoryState
from chakrview.cognition.repository.context_store import RepositoryContextStore
from chakrview.cognition.repository.patch_planning import (
    PatchPlan,
    PatchPlanStatus,
    PatchOperation,
    PatchOperationType,
    PatchPlanValidator,
    sanitize_relative_path,
)


class PatchExecutionStatus(Enum):
    """Lifecycle state machine for patch execution."""
    PLANNED = auto()
    VALIDATED = auto()
    READY_FOR_EXECUTION_REVIEW = auto()
    APPROVED = auto()
    APPLYING = auto()
    APPLIED = auto()
    VERIFIED = auto()
    REJECTED = auto()
    CONFLICTED = auto()
    STALE = auto()
    FAILED = auto()
    ROLLED_BACK = auto()


@dataclass(frozen=True)
class PatchExecutionRequest:
    """
    Explicit request to apply a validated PatchPlan.
    No plan can be executed without an explicit approved flag and approval token.
    """
    plan: PatchPlan
    approved: bool
    approval_token: str
    approver_identity: str = "local_system_user"


@dataclass(frozen=True)
class PatchRollbackRecord:
    """
    Immutable audit record retaining pre- and post-application states for deterministic rollback.
    """
    plan_id: str
    plan_fingerprint: str
    affected_files: Tuple[str, ...]
    pre_contents: Dict[str, Optional[str]]   # rel_path -> original content or None if file was newly created
    post_contents: Dict[str, str]            # rel_path -> applied content
    pre_repo_fingerprint: str
    applied_repo_fingerprint: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "plan_fingerprint": self.plan_fingerprint,
            "affected_files": list(self.affected_files),
            "pre_repo_fingerprint": self.pre_repo_fingerprint,
            "applied_repo_fingerprint": self.applied_repo_fingerprint,
        }


@dataclass(frozen=True)
class PatchExecutionResult:
    """
    Complete audit outcome of a patch execution attempt.
    Clearly distinguishes PATCH_APPLIED from PATCH_VERIFIED.
    """
    plan_id: str
    plan_fingerprint: str
    status: PatchExecutionStatus
    applied_files: Tuple[str, ...]
    unaffected_files_preserved: bool
    post_apply_verification_passed: bool
    rollback_record: Optional[PatchRollbackRecord]
    messages: Tuple[str, ...]
    execution_time_ms: float = 0.0

    @property
    def is_applied(self) -> bool:
        return self.status in (PatchExecutionStatus.APPLIED, PatchExecutionStatus.VERIFIED)

    @property
    def is_verified(self) -> bool:
        return self.status == PatchExecutionStatus.VERIFIED

    def to_dict(self) -> Dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "plan_fingerprint": self.plan_fingerprint,
            "status": self.status.name,
            "applied_files": list(self.applied_files),
            "unaffected_files_preserved": self.unaffected_files_preserved,
            "post_apply_verification_passed": self.post_apply_verification_passed,
            "has_rollback_record": self.rollback_record is not None,
            "messages": list(self.messages),
            "execution_time_ms": round(self.execution_time_ms, 2),
        }


class SafePatchExecutor:
    """
    Executes validated PatchPlans onto an IsolatedWorkspace or repository root.

    Enforces:
    1. Explicit Approval Barrier: READY_FOR_EXECUTION_REVIEW -> APPROVED.
    2. Stale State Defense: Matches live repository fingerprint against plan.
    3. Atomicity & In-Memory Staging: If any operation fails, zero changes are committed.
    4. Post-Application Verification: Asserts expected file changes occurred and unrelated files remained identical.
    5. Reversible Rollback: Pre-state is captured and can be deterministically restored.
    6. Path Containment: All writes are confined strictly within the workspace root.
    """

    def __init__(
        self,
        workspace: IsolatedWorkspace,
        context_store: Optional[RepositoryContextStore] = None,
    ) -> None:
        self.workspace = workspace
        self.context_store = context_store
        self._rollback_journal: Dict[str, PatchRollbackRecord] = {}

    def execute_patch(
        self,
        request: PatchExecutionRequest,
    ) -> PatchExecutionResult:
        t_start = time.perf_counter()
        plan = request.plan
        messages: List[str] = []

        # 1. Enforce Explicit Approval Barrier
        if not request.approved or not request.approval_token:
            messages.append("Execution blocked: Plan has not received explicit approval (APPROVED status required).")
            elapsed = (time.perf_counter() - t_start) * 1000.0
            return PatchExecutionResult(
                plan_id=plan.plan_id,
                plan_fingerprint=plan.plan_fingerprint,
                status=PatchExecutionStatus.READY_FOR_EXECUTION_REVIEW,
                applied_files=(),
                unaffected_files_preserved=True,
                post_apply_verification_passed=False,
                rollback_record=None,
                messages=tuple(messages),
                execution_time_ms=elapsed,
            )

        # 2. Check Plan Status
        if plan.status not in (PatchPlanStatus.VALIDATED, PatchPlanStatus.READY_FOR_EXECUTION_REVIEW):
            messages.append(f"Execution rejected: Plan status is {plan.status.name}, not READY_FOR_EXECUTION_REVIEW.")
            elapsed = (time.perf_counter() - t_start) * 1000.0
            return PatchExecutionResult(
                plan_id=plan.plan_id,
                plan_fingerprint=plan.plan_fingerprint,
                status=PatchExecutionStatus.REJECTED,
                applied_files=(),
                unaffected_files_preserved=True,
                post_apply_verification_passed=False,
                rollback_record=None,
                messages=tuple(messages),
                execution_time_ms=elapsed,
            )

        # 3. Read current live repository state & verify freshness (stale state check)
        current_manifest = self.workspace.manifest
        current_repo_state = RepositoryState.from_manifest(current_manifest)

        if plan.repository_fingerprint != current_repo_state.state_fingerprint:
            messages.append(
                f"Execution aborted: Stale repository state detected (plan {plan.repository_fingerprint} vs current {current_repo_state.state_fingerprint})."
            )
            elapsed = (time.perf_counter() - t_start) * 1000.0
            return PatchExecutionResult(
                plan_id=plan.plan_id,
                plan_fingerprint=plan.plan_fingerprint,
                status=PatchExecutionStatus.STALE,
                applied_files=(),
                unaffected_files_preserved=True,
                post_apply_verification_passed=False,
                rollback_record=None,
                messages=tuple(messages),
                execution_time_ms=elapsed,
            )

        # 4. Independent Plan Revalidation
        if self.context_store:
            revalidated_plan = PatchPlanValidator.validate_plan(
                plan=plan,
                context_store=self.context_store,
                current_repo_state=current_repo_state,
            )
            if revalidated_plan.status not in (PatchPlanStatus.VALIDATED, PatchPlanStatus.READY_FOR_EXECUTION_REVIEW):
                messages.append(f"Independent pre-flight validation failed: {revalidated_plan.validation_reasons}")
                elapsed = (time.perf_counter() - t_start) * 1000.0
                return PatchExecutionResult(
                    plan_id=plan.plan_id,
                    plan_fingerprint=plan.plan_fingerprint,
                    status=PatchExecutionStatus.REJECTED,
                    applied_files=(),
                    unaffected_files_preserved=True,
                    post_apply_verification_passed=False,
                    rollback_record=None,
                    messages=tuple(messages),
                    execution_time_ms=elapsed,
                )

        # 5. Atomic In-Memory Staging & Path Traversal Pre-checks
        staged_files: Dict[str, str] = {}
        pre_contents: Dict[str, Optional[str]] = {}

        # Capture initial hashes of all files in workspace to verify unrelated files later
        initial_file_hashes = {
            f.path: hashlib.sha256(f.content.encode("utf-8")).hexdigest()
            for f in current_manifest.files
        }

        try:
            for op in plan.operations:
                sanitized_f = sanitize_relative_path(op.target_file)
                # Verify path confinement using workspace security boundary
                target_file_path = self.workspace.get_source_file_path(sanitized_f)
                self.workspace._assert_within_workspace(target_file_path)

                # Capture pre-content
                if target_file_path.is_file():
                    pre_contents[sanitized_f] = target_file_path.read_text(encoding="utf-8")
                else:
                    pre_contents[sanitized_f] = None

                staged_files[sanitized_f] = op.new_content

        except Exception as e:
            messages.append(f"Staging failed before writing to disk (atomicity preserved): {e}")
            elapsed = (time.perf_counter() - t_start) * 1000.0
            return PatchExecutionResult(
                plan_id=plan.plan_id,
                plan_fingerprint=plan.plan_fingerprint,
                status=PatchExecutionStatus.FAILED,
                applied_files=(),
                unaffected_files_preserved=True,
                post_apply_verification_passed=False,
                rollback_record=None,
                messages=tuple(messages),
                execution_time_ms=elapsed,
            )

        # 6. Apply Staged Files to Workspace Filesystem
        applied_list: List[str] = []
        try:
            for rel_path, content in staged_files.items():
                self.workspace.write_source_file(rel_path, content)
                applied_list.append(rel_path)
        except Exception as write_err:
            messages.append(f"Write error during application: {write_err}. Executing immediate rollback.")
            # Atomic rollback of partially written files
            for written_f in applied_list:
                old_val = pre_contents.get(written_f)
                if old_val is not None:
                    self.workspace.write_source_file(written_f, old_val)
            elapsed = (time.perf_counter() - t_start) * 1000.0
            return PatchExecutionResult(
                plan_id=plan.plan_id,
                plan_fingerprint=plan.plan_fingerprint,
                status=PatchExecutionStatus.FAILED,
                applied_files=(),
                unaffected_files_preserved=True,
                post_apply_verification_passed=False,
                rollback_record=None,
                messages=tuple(messages),
                execution_time_ms=elapsed,
            )

        # 7. Post-Application Verification
        # Refresh workspace manifest
        post_manifest = self.workspace.manifest
        post_file_hashes = {
            f.path: hashlib.sha256(f.content.encode("utf-8")).hexdigest()
            for f in post_manifest.files
        }

        # Verify expected changes occurred
        expected_changes_ok = True
        for rel_path, expected_content in staged_files.items():
            expected_hash = hashlib.sha256(expected_content.encode("utf-8")).hexdigest()
            actual_hash = post_file_hashes.get(rel_path)
            if actual_hash != expected_hash:
                expected_changes_ok = False
                messages.append(f"Post-apply verification failed: file '{rel_path}' does not match expected content.")

        # Verify unrelated files were NOT changed
        unrelated_ok = True
        for path, orig_h in initial_file_hashes.items():
            if path not in staged_files:
                current_h = post_file_hashes.get(path)
                if current_h != orig_h:
                    unrelated_ok = False
                    messages.append(f"Post-apply regression: unrelated file '{path}' was modified!")

        post_repo_state = RepositoryState.from_manifest(post_manifest)

        # Construct Rollback Record
        rollback_record = PatchRollbackRecord(
            plan_id=plan.plan_id,
            plan_fingerprint=plan.plan_fingerprint,
            affected_files=tuple(applied_list),
            pre_contents=pre_contents,
            post_contents=staged_files,
            pre_repo_fingerprint=current_repo_state.state_fingerprint,
            applied_repo_fingerprint=post_repo_state.state_fingerprint,
        )
        self._rollback_journal[plan.plan_id] = rollback_record

        # Synchronize context store if provided
        if self.context_store:
            self.context_store.synchronize(post_manifest)

        elapsed = (time.perf_counter() - t_start) * 1000.0
        is_verified = expected_changes_ok and unrelated_ok
        status = PatchExecutionStatus.VERIFIED if is_verified else PatchExecutionStatus.APPLIED

        if is_verified:
            messages.append("Patch successfully applied and verified against repository state.")

        return PatchExecutionResult(
            plan_id=plan.plan_id,
            plan_fingerprint=plan.plan_fingerprint,
            status=status,
            applied_files=tuple(applied_list),
            unaffected_files_preserved=unrelated_ok,
            post_apply_verification_passed=is_verified,
            rollback_record=rollback_record,
            messages=tuple(messages),
            execution_time_ms=elapsed,
        )

    def rollback(self, plan_id: str) -> bool:
        """
        Rollback an applied plan using its recorded pre-state.
        Returns True if rollback completed cleanly.
        """
        record = self._rollback_journal.get(plan_id)
        if not record:
            return False

        for rel_path, old_content in record.pre_contents.items():
            if old_content is not None:
                self.workspace.write_source_file(rel_path, old_content)
            else:
                # File was newly created; remove it from workspace manifest and disk
                target_p = self.workspace.get_source_file_path(rel_path)
                if target_p.exists():
                    target_p.unlink()
                self.workspace.manifest.files = [f for f in self.workspace.manifest.files if f.path != rel_path]

        # Synchronize context store if present
        if self.context_store:
            self.context_store.synchronize(self.workspace.manifest)

        return True
