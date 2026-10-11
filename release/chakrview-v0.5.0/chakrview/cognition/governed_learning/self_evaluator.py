"""
ChakrView Step 95: Governed Self-Evaluation & Failure Analysis.

Builds the structured self-audit engine that interrogates task outcomes:
- Analyzes:
  1. What was the goal?
  2. What was actually done?
  3. What evidence supported the decisions?
  4. What went wrong (if failed)?
  5. What assumptions were invalidated?
  6. What information was missing?
  7. Which files/modules were affected?
  8. Was the result verified?
  9. What was the task verdict (ACCEPT / PARTIAL / REVISE / REJECT / ABSTAIN)?
  10. What operational lesson should be retained?
- Produces a formal GovernedFailureAnalysis or TaskAuditReport.
- Converts failures into structured learning signals rather than silently ignoring or discarding them.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from chakrview.cognition.ppb.expansion_models import GoalVerificationVerdict, GoalVerificationResult
from chakrview.cognition.ppb.task_models import PersistentTaskNode, TaskNodeStatus
from chakrview.cognition.governed_learning.experience_models import (
    ExperienceEpistemicCategory,
    ExperienceProvenance,
    GovernedExperienceRecord,
)


class FailureClass(str, Enum):
    """Taxonomy of operational and cognitive failure modes."""
    SYNTAX_OR_EXECUTION = "SYNTAX_OR_EXECUTION"       # Code compilation, syntax, runtime crash
    DEPENDENCY_MISMATCH = "DEPENDENCY_MISMATCH"       # Missing, outdated, or conflicting symbol/import
    RESOURCE_EXHAUSTION = "RESOURCE_EXHAUSTION"       # Context overflow or out-of-memory
    MISSING_KNOWLEDGE = "MISSING_KNOWLEDGE"           # Critical uninspected entity or unresolved UNKNOWN
    CONSTRAINT_VIOLATION = "CONSTRAINT_VIOLATION"     # Boundary violated or dangerous operation blocked
    VERIFICATION_FAILURE = "VERIFICATION_FAILURE"     # Post-execution tests or assertions failed
    TIMEOUT_OR_ABORT = "TIMEOUT_OR_ABORT"             # Exceeded loop bounds or deadlock


@dataclass
class GovernedFailureAnalysis:
    """
    Detailed audit of a failed subtask or unaccomplished objective.
    """
    failure_id: str
    task_id: str
    failure_class: FailureClass
    root_cause_summary: str
    invalidated_assumptions: List[str] = field(default_factory=list)
    missing_information: List[str] = field(default_factory=list)
    affected_files: List[str] = field(default_factory=list)
    recommended_recovery_action: str = ""
    reusable_avoidance_rule: str = ""
    timestamp_utc: str = field(
        default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "failure_id": self.failure_id,
            "task_id": self.task_id,
            "failure_class": self.failure_class.value,
            "root_cause_summary": self.root_cause_summary,
            "invalidated_assumptions": list(self.invalidated_assumptions),
            "missing_information": list(self.missing_information),
            "affected_files": list(self.affected_files),
            "recommended_recovery_action": self.recommended_recovery_action,
            "reusable_avoidance_rule": self.reusable_avoidance_rule,
            "timestamp_utc": self.timestamp_utc,
        }


@dataclass
class TaskAuditReport:
    """
    Step 95: Complete post-task self-audit report.
    Answers the 10 diagnostic questions required by the milestone.
    """
    audit_id: str
    task_id: str
    stated_goal: str
    actions_taken: List[str]
    supporting_evidence_ids: List[str]
    verdict: GoalVerificationVerdict
    is_success: bool
    failure_analysis: Optional[GovernedFailureAnalysis] = None
    extracted_lesson: Optional[str] = None
    affected_modules: List[str] = field(default_factory=list)
    confidence: float = 1.0
    audit_timestamp_utc: str = field(
        default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "audit_id": self.audit_id,
            "task_id": self.task_id,
            "stated_goal": self.stated_goal,
            "actions_taken": list(self.actions_taken),
            "supporting_evidence_ids": list(self.supporting_evidence_ids),
            "verdict": self.verdict.value,
            "is_success": self.is_success,
            "failure_analysis": self.failure_analysis.to_dict() if self.failure_analysis else None,
            "extracted_lesson": self.extracted_lesson,
            "affected_modules": list(self.affected_modules),
            "confidence": self.confidence,
            "audit_timestamp_utc": self.audit_timestamp_utc,
        }


class GovernedSelfEvaluator:
    """
    Step 95: Evaluates task outcomes and diagnoses failures into learning signals.
    """

    def audit_node_execution(
        self,
        node: PersistentTaskNode,
        execution_result: Dict[str, Any],
        error_message: Optional[str] = None,
    ) -> TaskAuditReport:
        """
        Conducts a formal post-execution evaluation on an individual task node.
        """
        audit_id = f"audit_{node.node_id}_{int(time.time())}"
        actions = [f"Executed node {node.node_id} ({node.resource_type.value})"]
        evidence = list(node.completion_evidence_ids)
        affected = list(node.affected_files)

        if node.status == TaskNodeStatus.FAILED or error_message is not None:
            err = error_message or node.failure_reason or "Unknown execution failure"
            f_class = self._classify_error(err)
            f_analysis = GovernedFailureAnalysis(
                failure_id=f"fail_{node.node_id}_{int(time.time())}",
                task_id=node.node_id,
                failure_class=f_class,
                root_cause_summary=err,
                invalidated_assumptions=[f"Assumed {node.node_id} would succeed on first attempt without {f_class.value}"],
                missing_information=[f"Missing prerequisite details for {f_class.value}"],
                affected_files=affected,
                recommended_recovery_action="Apply alternate strategy or acquire missing dependency context",
                reusable_avoidance_rule=f"Check for {f_class.value} before executing similar tasks",
            )
            lesson = f"Failure Lesson: {node.title} failed due to {f_class.value} ('{err}'). Verify prerequisites first."

            return TaskAuditReport(
                audit_id=audit_id,
                task_id=node.node_id,
                stated_goal=node.description or node.title,
                actions_taken=actions,
                supporting_evidence_ids=evidence,
                verdict=GoalVerificationVerdict.REJECT,
                is_success=False,
                failure_analysis=f_analysis,
                extracted_lesson=lesson,
                affected_modules=affected,
                confidence=0.9,
            )

        # Successful execution
        lesson = f"Success Lesson: {node.title} succeeded cleanly using {node.resource_type.value} resource strategy."
        return TaskAuditReport(
            audit_id=audit_id,
            task_id=node.node_id,
            stated_goal=node.description or node.title,
            actions_taken=actions,
            supporting_evidence_ids=evidence,
            verdict=GoalVerificationVerdict.ACCEPT,
            is_success=True,
            failure_analysis=None,
            extracted_lesson=lesson,
            affected_modules=affected,
            confidence=1.0,
        )

    def _classify_error(self, err_text: str) -> FailureClass:
        lower = err_text.lower()
        if "syntax" in lower or "cannot import" in lower or "importerror" in lower or "modulenotfound" in lower:
            return FailureClass.DEPENDENCY_MISMATCH
        elif "context" in lower or "memory" in lower or "overflow" in lower or "budget" in lower:
            return FailureClass.RESOURCE_EXHAUSTION
        elif "unknown" in lower or "not found" in lower or "missing" in lower:
            return FailureClass.MISSING_KNOWLEDGE
        elif "constraint" in lower or "unauthorized" in lower or "forbidden" in lower:
            return FailureClass.CONSTRAINT_VIOLATION
        elif "assert" in lower or "test" in lower or "verification" in lower:
            return FailureClass.VERIFICATION_FAILURE
        elif "timeout" in lower or "cycle" in lower:
            return FailureClass.TIMEOUT_OR_ABORT
        else:
            return FailureClass.SYNTAX_OR_EXECUTION

    def convert_audit_to_experience(
        self,
        audit: TaskAuditReport,
        project_id: str,
    ) -> GovernedExperienceRecord:
        """
        Translates a TaskAuditReport into a permanent GovernedExperienceRecord.
        """
        cat = ExperienceEpistemicCategory.SUCCESS if audit.is_success else ExperienceEpistemicCategory.FAILURE
        obs = audit.failure_analysis.root_cause_summary if audit.failure_analysis else "All task assertions passed"

        return GovernedExperienceRecord(
            record_id=f"exp_{audit.audit_id}",
            project_id=project_id,
            task_goal=audit.stated_goal,
            action_taken="; ".join(audit.actions_taken),
            observation=obs,
            result_summary=audit.verdict.value,
            epistemic_category=cat,
            is_success=audit.is_success,
            evidence_ids=list(audit.supporting_evidence_ids),
            confidence=audit.confidence,
            affected_modules=list(audit.affected_modules),
            lesson_summary=audit.extracted_lesson,
            diagnostics={
                "audit_verdict": audit.verdict.value,
                "failure_class": audit.failure_analysis.failure_class.value if audit.failure_analysis else None,
            },
            provenance=ExperienceProvenance(
                source_type="SELF_EVALUATION",
                task_id=audit.task_id,
                rationale=f"Evaluated task outcome: {audit.verdict.value}",
            ),
        )
