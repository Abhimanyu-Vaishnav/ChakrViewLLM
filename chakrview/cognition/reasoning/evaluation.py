"""
ChakrView Step 73: Self-Evaluation & Bounded Revision Subsystem.

Defines:
- EvaluationVerdict: Strict decision states (ACCEPT, REVISE, REJECT, ABSTAIN).
- EvaluationCriterionResult: Outcome of an individual self-audit check.
- SelfEvaluationReport: Authoritative audit report checking grounding, internal consistency,
  constraint satisfaction, uncertainty representation, and dangerous handles.
- BoundedRevisionCoordinator: Governed revision loop supporting up to MAX_REVISION_CYCLES (2)
  without infinite loops or unbounded self-reflection.
"""

from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from chakrview.cognition.repository.state import RepositoryState
from chakrview.cognition.repository.context_store import RepositoryContextStore
from chakrview.cognition.repository.cognitive_context import (
    CognitiveContextBundle,
    CognitiveContextItem,
)
from chakrview.cognition.repository.neural_proposal import (
    ProposalContract,
    ProposalValidationStatus,
)
from chakrview.cognition.reasoning.structured import (
    EpistemicCategory,
    EpistemicConfidenceState,
    ReasoningClaim,
    StructuredReasoningArtifact,
    StructuredReasoningEngine,
)
from chakrview.cognition.reasoning.critical import (
    CriticalAnalysisReport,
    CriticalThinkingEngine,
    EvidenceStrength,
)


class EvaluationVerdict(Enum):
    """Authoritative self-evaluation decision."""
    ACCEPT = auto()     # Valid, fully grounded, internally consistent, constraints satisfied
    REVISE = auto()     # Bounded, resolvable defect detected (e.g. ungrounded claim or alternative preferred)
    REJECT = auto()     # Irrecoverable defect (e.g. hallucinated file, dangerous execution keyword)
    ABSTAIN = auto()    # Genuine lack of evidence, unresolved contradiction, or safety conflict


@dataclass(frozen=True)
class EvaluationCriterionResult:
    """Outcome of evaluating a single cognitive safety or quality criterion."""
    criterion_name: str
    passed: bool
    details: str
    severity: str = "ERROR"  # "ERROR", "WARNING", "INFO"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "criterion_name": self.criterion_name,
            "passed": self.passed,
            "details": self.details,
            "severity": self.severity,
        }


@dataclass(frozen=True)
class SelfEvaluationReport:
    """
    Authoritative evaluation audit inspecting reasoning, critical analysis, and proposal.
    Does NOT trust the generator.
    """
    report_id: str
    artifact_id: str
    verdict: EvaluationVerdict
    criteria_results: Tuple[EvaluationCriterionResult, ...]
    identified_defects: Tuple[str, ...]
    revision_recommendations: Tuple[str, ...]
    cycle_index: int = 0
    report_fingerprint: str = ""

    def __post_init__(self) -> None:
        if not self.report_fingerprint:
            hasher = hashlib.sha256()
            hasher.update(self.artifact_id.encode("utf-8"))
            hasher.update(self.verdict.name.encode("utf-8"))
            hasher.update(str(self.cycle_index).encode("utf-8"))
            for c in self.criteria_results:
                hasher.update(c.criterion_name.encode("utf-8"))
                hasher.update(str(c.passed).encode("utf-8"))
            for d in self.identified_defects:
                hasher.update(d.encode("utf-8"))
            object.__setattr__(self, "report_fingerprint", hasher.hexdigest()[:16])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "report_id": self.report_id,
            "artifact_id": self.artifact_id,
            "verdict": self.verdict.name,
            "criteria_results": [c.to_dict() for c in self.criteria_results],
            "identified_defects": list(self.identified_defects),
            "revision_recommendations": list(self.revision_recommendations),
            "cycle_index": self.cycle_index,
            "report_fingerprint": self.report_fingerprint,
        }


class SelfEvaluator:
    """
    Independent deterministic evaluator auditing the combination of:
    1. Structured Reasoning Artifact
    2. Critical Analysis Report
    3. ProposalContract (if present)
    against repository context and safety invariants.
    """

    @classmethod
    def evaluate(
        cls,
        reasoning: StructuredReasoningArtifact,
        critical_report: CriticalAnalysisReport,
        proposal: Optional[ProposalContract] = None,
        context_bundle: Optional[CognitiveContextBundle] = None,
        context_store: Optional[RepositoryContextStore] = None,
        cycle_index: int = 0,
    ) -> SelfEvaluationReport:
        criteria: List[EvaluationCriterionResult] = []
        defects: List[str] = []
        revisions: List[str] = []

        # 0. Check if reasoning or critical report already abstained
        if reasoning.is_abstained or critical_report.recommendation == "ABSTAIN":
            criteria.append(EvaluationCriterionResult(
                criterion_name="epistemic_grounding",
                passed=False,
                details=f"Upstream abstention: {reasoning.abstain_reason or critical_report.unresolved_contradictions}",
            ))
            return SelfEvaluationReport(
                report_id=f"eval_abstain_{reasoning.artifact_fingerprint[:8]}",
                artifact_id=reasoning.artifact_id,
                verdict=EvaluationVerdict.ABSTAIN,
                criteria_results=tuple(criteria),
                identified_defects=(f"Abstained reasoning: {reasoning.abstain_reason}",),
                revision_recommendations=("Preserve uncertainty; do not fabricate evidence.",),
                cycle_index=cycle_index,
            )

        # 1. Grounding Invariant: Verify all claimed files & symbols exist
        grounding_ok = True
        if context_store:
            for claim in reasoning.claims:
                if claim.target_file and not context_store.file_exists(claim.target_file):
                    grounding_ok = False
                    defects.append(f"Claim '{claim.claim_id}' targets non-existent file '{claim.target_file}'.")
                if claim.target_symbol and not context_store.symbol_exists(claim.target_symbol):
                    grounding_ok = False
                    defects.append(f"Claim '{claim.claim_id}' targets non-existent symbol '{claim.target_symbol}'.")

        criteria.append(EvaluationCriterionResult(
            criterion_name="factual_grounding",
            passed=grounding_ok,
            details="All targeted files and symbols confirmed in repository context store." if grounding_ok else "Hallucinated target file or symbol detected.",
        ))

        # 2. Contradiction & Conflict Invariant
        has_contradictions = len(critical_report.unresolved_contradictions) > 0
        criteria.append(EvaluationCriterionResult(
            criterion_name="contradiction_containment",
            passed=not has_contradictions,
            details="No unresolved contradictions with negative boundaries or quarantined context." if not has_contradictions else f"Contradictions found: {critical_report.unresolved_contradictions}",
        ))
        if has_contradictions:
            for c in critical_report.unresolved_contradictions:
                defects.append(f"Contradiction: {c}")

        # 3. Provenance Verification
        provenance_ok = True
        for claim in reasoning.claims:
            if claim.category == EpistemicCategory.FACT and not claim.supporting_evidence_ids:
                provenance_ok = False
                defects.append(f"Claim '{claim.claim_id}' classified as FACT lacks supporting evidence ID.")
        criteria.append(EvaluationCriterionResult(
            criterion_name="provenance_integrity",
            passed=provenance_ok,
            details="All FACT claims carry verified evidence IDs." if provenance_ok else "One or more FACT claims lack provenance linkage.",
        ))

        # 4. Prohibited Execution Handles Check
        dangerous_patterns = ["os.system", "subprocess.", "eval(", "exec(", "shutil.rmtree"]
        dangerous_found = False
        if proposal:
            for path, patch_code in proposal.proposed_changes.items():
                for dp in dangerous_patterns:
                    if dp in patch_code:
                        dangerous_found = True
                        defects.append(f"Prohibited execution handle '{dp}' detected in proposal for '{path}'.")

        criteria.append(EvaluationCriterionResult(
            criterion_name="security_handles",
            passed=not dangerous_found,
            details="Zero prohibited execution handles detected." if not dangerous_found else "Dangerous execution handle found in proposal.",
        ))

        # 5. Task Constraints Satisfaction
        constraints_ok = True
        if context_bundle and context_bundle.request.explicit_constraints:
            for c in context_bundle.request.explicit_constraints:
                # Check if proposal violates explicit constraint
                if proposal and proposal.validation_status == ProposalValidationStatus.CONFLICTED:
                    constraints_ok = False
                    defects.append(f"Proposal conflicts with explicit constraint: '{c}'")

        criteria.append(EvaluationCriterionResult(
            criterion_name="task_constraints",
            passed=constraints_ok,
            details="All explicit task constraints satisfied." if constraints_ok else "Explicit constraint violation detected.",
        ))

        # 6. Determine Verdict
        if dangerous_found or not grounding_ok:
            verdict = EvaluationVerdict.REJECT
        elif has_contradictions:
            verdict = EvaluationVerdict.ABSTAIN
        elif not provenance_ok:
            if cycle_index < 2:
                verdict = EvaluationVerdict.REVISE
                revisions.append("Demote ungrounded FACT claims to INFERENCE or provide valid evidence IDs.")
            else:
                verdict = EvaluationVerdict.REJECT
        elif critical_report.recommendation == "EXPLORE_ALTERNATIVES" and cycle_index < 2:
            verdict = EvaluationVerdict.REVISE
            revisions.append("Evaluate top alternative hypothesis before finalizing patch.")
        else:
            verdict = EvaluationVerdict.ACCEPT

        report_id = f"eval_{hashlib.sha256((reasoning.artifact_id + verdict.name + str(cycle_index)).encode('utf-8')).hexdigest()[:10]}"

        return SelfEvaluationReport(
            report_id=report_id,
            artifact_id=reasoning.artifact_id,
            verdict=verdict,
            criteria_results=tuple(criteria),
            identified_defects=tuple(defects),
            revision_recommendations=tuple(revisions),
            cycle_index=cycle_index,
        )


@dataclass(frozen=True)
class BoundedRevisionResult:
    """Outcome of a bounded self-evaluation and revision execution."""
    final_verdict: EvaluationVerdict
    initial_evaluation: SelfEvaluationReport
    final_evaluation: SelfEvaluationReport
    cycles_completed: int
    revised_reasoning: StructuredReasoningArtifact
    revised_proposal: Optional[ProposalContract]
    audit_trail: Tuple[str, ...]


class BoundedRevisionCoordinator:
    """
    Executes a bounded revision cycle (maximum 2 passes) to self-correct
    epistemic defects deterministically without entering infinite self-reflection loops.
    """
    MAX_CYCLES: int = 2

    @classmethod
    def coordinate(
        cls,
        context_bundle: CognitiveContextBundle,
        proposal: Optional[ProposalContract] = None,
        context_store: Optional[RepositoryContextStore] = None,
    ) -> BoundedRevisionResult:
        audit: List[str] = []
        cycle = 0

        # Cycle 0: Initial Reasoning -> Critical Thinking -> Self-Evaluation
        current_reasoning = StructuredReasoningEngine.reason(
            context_bundle=context_bundle,
            proposal=proposal,
            context_store=context_store,
        )
        audit.append(f"Cycle 0: Generated initial reasoning artifact '{current_reasoning.artifact_id}'.")

        current_critical = CriticalThinkingEngine.evaluate(
            artifact=current_reasoning,
            context_bundle=context_bundle,
            context_store=context_store,
        )
        audit.append(f"Cycle 0: Critical analysis recommended '{current_critical.recommendation}'.")

        current_eval = SelfEvaluator.evaluate(
            reasoning=current_reasoning,
            critical_report=current_critical,
            proposal=proposal,
            context_bundle=context_bundle,
            context_store=context_store,
            cycle_index=cycle,
        )
        audit.append(f"Cycle 0: Self-evaluation verdict '{current_eval.verdict.name}'.")
        initial_eval = current_eval
        current_proposal = proposal

        # Bounded Revision Loop
        while current_eval.verdict == EvaluationVerdict.REVISE and cycle < cls.MAX_CYCLES:
            cycle += 1
            audit.append(f"Entering revision cycle {cycle} to resolve defects: {current_eval.identified_defects}")

            # Apply deterministic revision:
            # Demote ungrounded claims or incorporate alternative hypothesis
            revised_claims: List[ReasoningClaim] = []
            for claim in current_reasoning.claims:
                if claim.category == EpistemicCategory.FACT and not claim.supporting_evidence_ids:
                    # Demote to INFERENCE
                    revised_claims.append(ReasoningClaim(
                        claim_id=claim.claim_id,
                        category=EpistemicCategory.INFERENCE,
                        statement=claim.statement,
                        confidence_state=EpistemicConfidenceState.INFERRED,
                        target_file=claim.target_file,
                        target_symbol=claim.target_symbol,
                    ))
                else:
                    revised_claims.append(claim)

            # Re-build revised reasoning artifact
            current_reasoning = StructuredReasoningArtifact(
                artifact_id=f"{current_reasoning.artifact_id}_rev{cycle}",
                task_id=current_reasoning.task_id,
                context_fingerprint=current_reasoning.context_fingerprint,
                repository_fingerprint=current_reasoning.repository_fingerprint,
                claims=tuple(revised_claims),
                assumptions=current_reasoning.assumptions,
                observations=current_reasoning.observations,
                candidate_conclusions=current_reasoning.candidate_conclusions,
                unresolved_questions=current_reasoning.unresolved_questions,
                investigation_requirements=current_reasoning.investigation_requirements,
                overall_confidence=current_reasoning.overall_confidence,
                is_abstained=current_reasoning.is_abstained,
                abstain_reason=current_reasoning.abstain_reason,
            )

            current_critical = CriticalThinkingEngine.evaluate(
                artifact=current_reasoning,
                context_bundle=context_bundle,
                context_store=context_store,
            )

            current_eval = SelfEvaluator.evaluate(
                reasoning=current_reasoning,
                critical_report=current_critical,
                proposal=current_proposal,
                context_bundle=context_bundle,
                context_store=context_store,
                cycle_index=cycle,
            )
            audit.append(f"Cycle {cycle}: Self-evaluation verdict '{current_eval.verdict.name}'.")

        # If still in REVISE state after reaching MAX_CYCLES, fail closed to REJECT
        final_verdict = current_eval.verdict
        if final_verdict == EvaluationVerdict.REVISE:
            final_verdict = EvaluationVerdict.REJECT
            audit.append(f"Exceeded max revision cycles ({cls.MAX_CYCLES}). Fail-closed transition to REJECT.")

        return BoundedRevisionResult(
            final_verdict=final_verdict,
            initial_evaluation=initial_eval,
            final_evaluation=current_eval,
            cycles_completed=cycle,
            revised_reasoning=current_reasoning,
            revised_proposal=current_proposal,
            audit_trail=tuple(audit),
        )
