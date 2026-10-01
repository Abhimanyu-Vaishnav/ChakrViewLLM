"""
ChakrView Step 69: Deterministic Patch Planning, Provenance Tracking & Independent Validation.

Defines:
- PatchOperationType: Explicit atomic mutation types (CREATE_FILE, MODIFY_FILE, DELETE_FILE, REPLACE).
- PatchOperation: Strongly typed, provenance-linked atomic mutation record with deterministic fingerprint.
- PatchPlanStatus: Explicit plan lifecycle states (PLANNED, VALIDATED, READY_FOR_EXECUTION_REVIEW, REJECTED, CONFLICTED, STALE).
- PatchPlan: Canonical, deterministic plan specifying affected files, operations, unified diffs, and provenance.
- PatchPlanner: Compiles validated ProposalContract into a structured PatchPlan.
- PatchPlanValidator: Independent deterministic gate auditing the PatchPlan against live repository facts.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import re
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from chakrview.cognition.repository.state import RepositoryState
from chakrview.cognition.repository.context_store import RepositoryContextStore, EpistemicState
from chakrview.cognition.repository.neural_proposal import (
    ProposalContract,
    ProposalValidationStatus,
)


class PatchOperationType(Enum):
    """Atomic operation types supported in a deterministic patch plan."""
    CREATE_FILE = auto()
    MODIFY_FILE = auto()
    DELETE_FILE = auto()
    REPLACE = auto()


class PatchPlanStatus(Enum):
    """Lifecycle status for a PatchPlan."""
    PLANNED = auto()
    VALIDATED = auto()
    READY_FOR_EXECUTION_REVIEW = auto()
    REJECTED = auto()
    CONFLICTED = auto()
    STALE = auto()


def _canonical_digest(data: str) -> str:
    """Computes a deterministic 16-hex-character digest of string content."""
    return hashlib.sha256(data.encode("utf-8")).hexdigest()[:16]


def sanitize_relative_path(rel_path: str) -> str:
    """
    Normalizes and validates a relative path against directory traversal and drive escapes.
    Raises ValueError if path attempts to escape repository root.
    """
    cleaned = rel_path.strip().replace("\\", "/")
    if not cleaned:
        raise ValueError("Empty file path.")
    # Disallow UNC prefix or special devices
    if cleaned.startswith("//") or rel_path.startswith("\\\\"):
        raise ValueError(f"UNC paths not permitted: '{rel_path}'")
    if cleaned.startswith("/") or re.match(r"^[a-zA-Z]:", cleaned):
        raise ValueError(f"Absolute paths not permitted: '{rel_path}'")
    parts = cleaned.split("/")
    if any(p in ("..", ".") for p in parts if p):
        raise ValueError(f"Path traversal ('..' or '.') detected: '{rel_path}'")
    return cleaned


@dataclass(frozen=True)
class PatchOperation:
    """
    Atomic code modification operation with complete provenance and deterministic fingerprint.
    Identity is independent of timestamps or process IDs.
    """
    operation_type: PatchOperationType
    target_file: str
    target_symbol: Optional[str]
    original_content: Optional[str]
    new_content: str
    operation_fingerprint: str
    provenance_evidence_ids: Tuple[str, ...]
    provenance_memory_ids: Tuple[str, ...]
    provenance_negative_boundary_ids: Tuple[str, ...]

    @classmethod
    def create(
        cls,
        operation_type: PatchOperationType,
        target_file: str,
        new_content: str,
        original_content: Optional[str] = None,
        target_symbol: Optional[str] = None,
        evidence_ids: Sequence[str] = (),
        memory_ids: Sequence[str] = (),
        negative_boundary_ids: Sequence[str] = (),
    ) -> PatchOperation:
        sanitized_file = sanitize_relative_path(target_file)
        # Compute canonical operation fingerprint
        hasher = hashlib.sha256()
        hasher.update(operation_type.name.encode("utf-8"))
        hasher.update(sanitized_file.encode("utf-8"))
        if target_symbol:
            hasher.update(target_symbol.encode("utf-8"))
        if original_content is not None:
            hasher.update(original_content.encode("utf-8"))
        hasher.update(new_content.encode("utf-8"))
        op_fp = hasher.hexdigest()[:16]

        return cls(
            operation_type=operation_type,
            target_file=sanitized_file,
            target_symbol=target_symbol,
            original_content=original_content,
            new_content=new_content,
            operation_fingerprint=op_fp,
            provenance_evidence_ids=tuple(sorted(evidence_ids)),
            provenance_memory_ids=tuple(sorted(memory_ids)),
            provenance_negative_boundary_ids=tuple(sorted(negative_boundary_ids)),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "operation_type": self.operation_type.name,
            "target_file": self.target_file,
            "target_symbol": self.target_symbol,
            "has_original_content": self.original_content is not None,
            "new_content_bytes": len(self.new_content),
            "operation_fingerprint": self.operation_fingerprint,
            "provenance_evidence_ids": list(self.provenance_evidence_ids),
            "provenance_memory_ids": list(self.provenance_memory_ids),
            "provenance_negative_boundary_ids": list(self.provenance_negative_boundary_ids),
        }


@dataclass(frozen=True)
class PatchPlan:
    """
    Deterministic specification of repository modifications derived from a validated ProposalContract.
    All fingerprints are calculated strictly from canonical content.
    """
    plan_id: str
    proposal_id: str
    context_fingerprint: str
    repository_fingerprint: str
    target_files: Tuple[str, ...]
    operations: Tuple[PatchOperation, ...]
    unified_diff: str
    plan_fingerprint: str
    affected_files_fingerprint: str
    status: PatchPlanStatus = PatchPlanStatus.PLANNED
    validation_reasons: Tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "proposal_id": self.proposal_id,
            "context_fingerprint": self.context_fingerprint,
            "repository_fingerprint": self.repository_fingerprint,
            "target_files": list(self.target_files),
            "operations": [op.to_dict() for op in self.operations],
            "unified_diff": self.unified_diff,
            "plan_fingerprint": self.plan_fingerprint,
            "affected_files_fingerprint": self.affected_files_fingerprint,
            "status": self.status.name,
            "validation_reasons": list(self.validation_reasons),
        }


def generate_deterministic_diff(
    target_file: str,
    original_content: str,
    new_content: str,
) -> str:
    """
    Generates a deterministic unified diff for a single file.
    Does NOT depend on local system timestamps or line ending differences.
    """
    orig_lines = [l + "\n" for l in original_content.replace("\r\n", "\n").split("\n")]
    new_lines = [l + "\n" for l in new_content.replace("\r\n", "\n").split("\n")]
    # Remove trailing artificial newline if source was empty
    if not original_content:
        orig_lines = []
    if not new_content:
        new_lines = []

    diff_lines = list(difflib.unified_diff(
        orig_lines,
        new_lines,
        fromfile=f"a/{target_file}",
        tofile=f"b/{target_file}",
        lineterm="\n",
    ))
    return "".join(diff_lines)


class PatchPlanner:
    """
    Compiles a validated ProposalContract and active RepositoryState into a PatchPlan.
    Fails closed if the proposal is rejected, conflicted, or ungrounded.
    """

    @classmethod
    def create_plan(
        cls,
        proposal: ProposalContract,
        repo_state: RepositoryState,
        allowed_files: Optional[Set[str]] = None,
    ) -> PatchPlan:
        reasons: List[str] = []

        # 1. Fail closed on proposal status
        if proposal.validation_status != ProposalValidationStatus.ACCEPTED_FOR_EXECUTION_REVIEW:
            status = PatchPlanStatus.CONFLICTED if proposal.validation_status == ProposalValidationStatus.CONFLICTED else PatchPlanStatus.REJECTED
            reasons.append(f"Proposal status '{proposal.validation_status.name}' is not ACCEPTED_FOR_EXECUTION_REVIEW.")
            # Return rejected/conflicted plan stub
            return PatchPlan(
                plan_id=f"plan_rejected_{proposal.proposal_id}",
                proposal_id=proposal.proposal_id,
                context_fingerprint=proposal.context_fingerprint,
                repository_fingerprint=repo_state.state_fingerprint,
                target_files=proposal.target_files,
                operations=(),
                unified_diff="",
                plan_fingerprint=_canonical_digest(f"rejected:{proposal.proposal_id}:{repo_state.state_fingerprint}"),
                affected_files_fingerprint="",
                status=status,
                validation_reasons=tuple(reasons + list(proposal.validation_reasons)),
            )

        # 2. Check allowed_files boundary
        effective_allowed = allowed_files or set(proposal.target_files)
        for tf in proposal.target_files:
            try:
                sanitized = sanitize_relative_path(tf)
            except ValueError as e:
                reasons.append(str(e))
                continue
            if sanitized not in effective_allowed:
                reasons.append(f"Target file '{sanitized}' outside allowed_files scope.")

        operations_list: List[PatchOperation] = []
        diff_chunks: List[str] = []

        # 3. Create operations deterministically in sorted file order
        sorted_files = sorted(proposal.proposed_changes.keys())
        for f_path in sorted_files:
            try:
                sanitized_f = sanitize_relative_path(f_path)
            except ValueError as e:
                reasons.append(str(e))
                continue

            new_patch_content = proposal.proposed_changes[f_path]

            # Check if file exists in repo_state
            if sanitized_f in repo_state.files:
                f_state = repo_state.files[sanitized_f]
                # If content is already present in FileState, read it
                orig_content = ""
                # For test repos or full state, f_state carries metadata; content may be retrieved from workspace if applicable
                op = PatchOperation.create(
                    operation_type=PatchOperationType.MODIFY_FILE,
                    target_file=sanitized_f,
                    new_content=new_patch_content,
                    original_content=orig_content,
                    evidence_ids=proposal.supporting_evidence_ids,
                    memory_ids=proposal.supporting_memory_ids,
                    negative_boundary_ids=proposal.negative_boundary_ids,
                )
                operations_list.append(op)
                diff = generate_deterministic_diff(sanitized_f, orig_content, new_patch_content)
                diff_chunks.append(diff)
            else:
                # File creation
                op = PatchOperation.create(
                    operation_type=PatchOperationType.CREATE_FILE,
                    target_file=sanitized_f,
                    new_content=new_patch_content,
                    original_content=None,
                    evidence_ids=proposal.supporting_evidence_ids,
                    memory_ids=proposal.supporting_memory_ids,
                    negative_boundary_ids=proposal.negative_boundary_ids,
                )
                operations_list.append(op)
                diff = generate_deterministic_diff(sanitized_f, "", new_patch_content)
                diff_chunks.append(diff)

        # 4. Compute canonical fingerprints
        hasher_plan = hashlib.sha256()
        hasher_plan.update(proposal.proposal_id.encode("utf-8"))
        hasher_plan.update(proposal.context_fingerprint.encode("utf-8"))
        hasher_plan.update(repo_state.state_fingerprint.encode("utf-8"))
        for op in operations_list:
            hasher_plan.update(op.operation_fingerprint.encode("utf-8"))
        plan_fp = hasher_plan.hexdigest()[:16]

        hasher_aff = hashlib.sha256()
        for f in sorted_files:
            hasher_aff.update(f.encode("utf-8"))
        aff_fp = hasher_aff.hexdigest()[:16]

        full_diff = "\n".join(diff_chunks)
        plan_id = f"plan_{plan_fp}"

        status = PatchPlanStatus.PLANNED if not reasons else PatchPlanStatus.REJECTED

        return PatchPlan(
            plan_id=plan_id,
            proposal_id=proposal.proposal_id,
            context_fingerprint=proposal.context_fingerprint,
            repository_fingerprint=repo_state.state_fingerprint,
            target_files=tuple(sorted_files),
            operations=tuple(operations_list),
            unified_diff=full_diff,
            plan_fingerprint=plan_fp,
            affected_files_fingerprint=aff_fp,
            status=status,
            validation_reasons=tuple(reasons),
        )


class PatchPlanValidator:
    """
    Independent gate auditing the PatchPlan against the repository state and security boundaries.
    Does NOT trust the PatchPlanner.
    """

    @classmethod
    def validate_plan(
        cls,
        plan: PatchPlan,
        context_store: RepositoryContextStore,
        current_repo_state: RepositoryState,
        allowed_files: Optional[Set[str]] = None,
    ) -> PatchPlan:
        reasons: List[str] = list(plan.validation_reasons)
        status = PatchPlanStatus.READY_FOR_EXECUTION_REVIEW

        # 1. State fingerprint match (stale state check)
        if plan.repository_fingerprint != current_repo_state.state_fingerprint:
            reasons.append(
                f"Stale plan repository fingerprint: plan {plan.repository_fingerprint} != current {current_repo_state.state_fingerprint}"
            )
            status = PatchPlanStatus.STALE

        # 2. Re-verify plan fingerprint integrity
        hasher_plan = hashlib.sha256()
        hasher_plan.update(plan.proposal_id.encode("utf-8"))
        hasher_plan.update(plan.context_fingerprint.encode("utf-8"))
        hasher_plan.update(plan.repository_fingerprint.encode("utf-8"))
        for op in plan.operations:
            hasher_plan.update(op.operation_fingerprint.encode("utf-8"))
        expected_plan_fp = hasher_plan.hexdigest()[:16]

        if plan.plan_fingerprint != expected_plan_fp:
            reasons.append(f"Corrupted plan fingerprint: {plan.plan_fingerprint} != computed {expected_plan_fp}")
            status = PatchPlanStatus.REJECTED

        # 3. Check allowed_files
        effective_allowed = allowed_files or set(plan.target_files)
        for tf in plan.target_files:
            try:
                sanitized = sanitize_relative_path(tf)
            except ValueError as e:
                reasons.append(f"Security violation: {e}")
                status = PatchPlanStatus.REJECTED
                continue
            if sanitized not in effective_allowed:
                reasons.append(f"File '{sanitized}' outside allowed_files scope.")
                status = PatchPlanStatus.REJECTED

        # 4. Check each operation against context store and security boundaries
        for op in plan.operations:
            try:
                sanitized_op_file = sanitize_relative_path(op.target_file)
            except ValueError as e:
                reasons.append(f"Security violation: {e}")
                status = PatchPlanStatus.REJECTED
                continue

            if op.operation_type in (PatchOperationType.MODIFY_FILE, PatchOperationType.DELETE_FILE):
                if not context_store.file_exists(sanitized_op_file) and sanitized_op_file not in current_repo_state.files:
                    reasons.append(f"Target file for {op.operation_type.name} does not exist: '{sanitized_op_file}'")
                    status = PatchPlanStatus.REJECTED

            # Security check: dangerous keywords
            dangerous_patterns = ["os.system", "subprocess.", "eval(", "exec(", "shutil.rmtree"]
            for dp in dangerous_patterns:
                if dp in op.new_content:
                    reasons.append(f"Prohibited execution handle '{dp}' in operation for '{op.target_file}'.")
                    status = PatchPlanStatus.REJECTED

            # Provenance check
            if not op.provenance_evidence_ids and not op.provenance_memory_ids:
                reasons.append(f"Operation on '{op.target_file}' lacks supporting provenance.")
                if status != PatchPlanStatus.REJECTED:
                    status = PatchPlanStatus.REJECTED

        if reasons and status not in (PatchPlanStatus.STALE, PatchPlanStatus.CONFLICTED):
            status = PatchPlanStatus.REJECTED

        return PatchPlan(
            plan_id=plan.plan_id,
            proposal_id=plan.proposal_id,
            context_fingerprint=plan.context_fingerprint,
            repository_fingerprint=plan.repository_fingerprint,
            target_files=plan.target_files,
            operations=plan.operations,
            unified_diff=plan.unified_diff,
            plan_fingerprint=plan.plan_fingerprint,
            affected_files_fingerprint=plan.affected_files_fingerprint,
            status=status,
            validation_reasons=tuple(reasons),
        )
