"""
ChakrView Step 72: Critical Thinking & Alternative Hypotheses Subsystem.

Defines:
- EvidenceBalance: Structured analysis of corroborating vs contradictory evidence.
- AlternativeHypothesis: Rigorous representation of competing explanations or implementations.
- CriticalAnalysisReport: Deep audit assessing evidence strength, assumption vulnerabilities,
  conflicts, and alternative hypotheses without fabricating evidence.
- CriticalThinkingEngine: Evaluates claims against negative boundaries, conflicting evidence,
  and alternative options.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from chakrview.cognition.repository.state import RepositoryState
from chakrview.cognition.repository.context_store import RepositoryContextStore
from chakrview.cognition.repository.cognitive_context import (
    CognitiveContextBundle,
    CognitiveContextItem,
    CognitiveContextStatus,
)
from chakrview.cognition.reasoning.structured import (
    EpistemicCategory,
    EpistemicConfidenceState,
    ReasoningClaim,
    AssumptionRecord,
    StructuredReasoningArtifact,
    InvestigationRequirement,
)


class EvidenceStrength(Enum):
    """Graduated assessment of evidence quality."""
    DEFINITIVE = auto()     # Direct AST check or verified test assertion in active repository
    CORROBORATED = auto()   # Multi-source alignment across episodic memory and AST inspection
    INDIRECT = auto()       # Inferred through dependency graph traversal or similarity matching
    CONTESTED = auto()      # Conflicting evidence or negative boundary collision
    STALE = auto()          # Evidence referencing outdated repository state
    INSUFFICIENT = auto()   # Inadequate evidence to support assertion


@dataclass(frozen=True)
class EvidenceBalance:
    """
    Structured breakdown of corroborating vs contradictory evidence for a specific claim.
    """
    claim_id: str
    supporting_evidence_ids: Tuple[str, ...]
    contradicting_evidence_ids: Tuple[str, ...]
    strength: EvidenceStrength
    has_conflicts: bool
    is_stale: bool
    evaluation_summary: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "supporting_evidence_ids": list(self.supporting_evidence_ids),
            "contradicting_evidence_ids": list(self.contradicting_evidence_ids),
            "strength": self.strength.name,
            "has_conflicts": self.has_conflicts,
            "is_stale": self.is_stale,
            "evaluation_summary": self.evaluation_summary,
        }


@dataclass(frozen=True)
class AlternativeHypothesis:
    """
    Represents an alternative or competing explanation or implementation strategy.
    Prevents fixation on a single unverified hypothesis.
    """
    hypothesis_id: str
    description: str
    rationale: str
    plausibility: float  # 0.0 to 1.0 deterministic relative plausibility
    required_evidence: str
    tradeoffs: Tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "description": self.description,
            "rationale": self.rationale,
            "plausibility": round(self.plausibility, 2),
            "required_evidence": self.required_evidence,
            "tradeoffs": list(self.tradeoffs),
        }


@dataclass(frozen=True)
class CriticalAnalysisReport:
    """
    Comprehensive critical audit of a reasoning artifact.
    Identifies blind spots, unexamined assumptions, and alternative explanations.
    """
    report_id: str
    artifact_id: str
    evidence_balances: Tuple[EvidenceBalance, ...]
    alternative_hypotheses: Tuple[AlternativeHypothesis, ...]
    vulnerable_assumptions: Tuple[AssumptionRecord, ...]
    unresolved_contradictions: Tuple[str, ...]
    epistemic_reliability_score: float  # 0.0 to 1.0
    recommendation: str  # "PROCEED", "EXPLORE_ALTERNATIVES", "ACQUIRE_EVIDENCE", "ABSTAIN"
    report_fingerprint: str = ""

    def __post_init__(self) -> None:
        if not self.report_fingerprint:
            hasher = hashlib.sha256()
            hasher.update(self.artifact_id.encode("utf-8"))
            hasher.update(self.recommendation.encode("utf-8"))
            for b in self.evidence_balances:
                hasher.update(b.claim_id.encode("utf-8"))
                hasher.update(b.strength.name.encode("utf-8"))
            for alt in self.alternative_hypotheses:
                hasher.update(alt.hypothesis_id.encode("utf-8"))
            object.__setattr__(self, "report_fingerprint", hasher.hexdigest()[:16])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "report_id": self.report_id,
            "artifact_id": self.artifact_id,
            "evidence_balances": [b.to_dict() for b in self.evidence_balances],
            "alternative_hypotheses": [a.to_dict() for a in self.alternative_hypotheses],
            "vulnerable_assumptions": [v.to_dict() for v in self.vulnerable_assumptions],
            "unresolved_contradictions": list(self.unresolved_contradictions),
            "epistemic_reliability_score": round(self.epistemic_reliability_score, 2),
            "recommendation": self.recommendation,
            "report_fingerprint": self.report_fingerprint,
        }


class CriticalThinkingEngine:
    """
    Executes disciplined critical evaluation of a StructuredReasoningArtifact.
    Enforces epistemological rigor:
    1. Checks if claims have actual evidence or are ungrounded leaps.
    2. Identifies collisions with negative boundaries or quarantined context items.
    3. Generates competing plausible hypotheses.
    4. Evaluates assumption vulnerabilities.
    """

    @classmethod
    def evaluate(
        cls,
        artifact: StructuredReasoningArtifact,
        context_bundle: Optional[CognitiveContextBundle] = None,
        context_store: Optional[RepositoryContextStore] = None,
    ) -> CriticalAnalysisReport:
        balances: List[EvidenceBalance] = []
        contradictions: List[str] = []
        vulnerabilities: List[AssumptionRecord] = []
        alternatives: List[AlternativeHypothesis] = []

        # If reasoning artifact was already abstained, pass-through critical assessment
        if artifact.is_abstained:
            balances.append(EvidenceBalance(
                claim_id="abstained_artifact",
                supporting_evidence_ids=(),
                contradicting_evidence_ids=(),
                strength=EvidenceStrength.INSUFFICIENT,
                has_conflicts=False,
                is_stale=False,
                evaluation_summary=f"Artifact was abstained: {artifact.abstain_reason}",
            ))
            return CriticalAnalysisReport(
                report_id=f"crit_abstain_{artifact.artifact_fingerprint[:8]}",
                artifact_id=artifact.artifact_id,
                evidence_balances=tuple(balances),
                alternative_hypotheses=(),
                vulnerable_assumptions=(),
                unresolved_contradictions=(artifact.abstain_reason or "Artifact abstained",),
                epistemic_reliability_score=0.0,
                recommendation="ABSTAIN",
            )

        # 1. Audit evidence balance for each claim
        quarantined_ids = set()
        stale_ids = set()
        if context_bundle:
            for conf in context_bundle.conflicted_items:
                quarantined_ids.add(conf.item_id)
                quarantined_ids.add(conf.source_identifier)
            for item in context_bundle.positive_evidence + context_bundle.positive_memories:
                if item.status == CognitiveContextStatus.STALE:
                    stale_ids.add(item.item_id)

        for claim in artifact.claims:
            # Check for collision with quarantined conflict IDs
            has_conflict = False
            for sid in claim.supporting_evidence_ids + claim.supporting_memory_ids:
                if sid in quarantined_ids:
                    has_conflict = True
                    contradictions.append(f"Claim '{claim.claim_id}' references quarantined conflict item '{sid}'.")

            # Check if evidence is stale
            is_stale = any(sid in stale_ids for sid in claim.supporting_evidence_ids)

            # Determine strength
            if has_conflict:
                strength = EvidenceStrength.CONTESTED
                summary = "Claim contradicted by quarantined negative boundary or task constraint."
            elif is_stale:
                strength = EvidenceStrength.STALE
                summary = "Supporting evidence is flagged as stale relative to repository state."
            elif claim.category == EpistemicCategory.FACT and claim.supporting_evidence_ids:
                strength = EvidenceStrength.DEFINITIVE
                summary = "Directly grounded in repository AST or dependency fact."
            elif claim.supporting_memory_ids:
                strength = EvidenceStrength.CORROBORATED
                summary = "Corroborated by verified episodic memory pattern."
            elif claim.supporting_evidence_ids:
                strength = EvidenceStrength.INDIRECT
                summary = "Inferred from active repository context."
            else:
                strength = EvidenceStrength.INSUFFICIENT
                summary = "No supporting evidence or memory IDs provided."

            balance = EvidenceBalance(
                claim_id=claim.claim_id,
                supporting_evidence_ids=claim.supporting_evidence_ids + claim.supporting_memory_ids,
                contradicting_evidence_ids=claim.contradicting_evidence_ids,
                strength=strength,
                has_conflicts=has_conflict,
                is_stale=is_stale,
                evaluation_summary=summary,
            )
            balances.append(balance)

        # 2. Audit assumptions
        for asm in artifact.assumptions:
            if not asm.is_empirically_verified and asm.criticality == "HIGH":
                vulnerabilities.append(asm)

        # 3. Formulate alternative hypotheses
        proposal_claims = [c for c in artifact.claims if c.category == EpistemicCategory.PROPOSAL]
        if proposal_claims:
            for p_claim in proposal_claims:
                # Synthesize alternative approaches deterministically
                alternatives.append(AlternativeHypothesis(
                    hypothesis_id=f"alt_guard_{p_claim.claim_id[:8]}",
                    description=f"Defensive precondition guard instead of inline rewrite for {p_claim.target_file or 'target'}",
                    rationale="Avoids changing existing algorithmic flow by asserting valid preconditions at the function boundary.",
                    plausibility=0.75,
                    required_evidence="Function entrypoint call sites and caller contract expectations.",
                    tradeoffs=("Lower risk of algorithmic regression", "May raise exceptions if invalid inputs passed"),
                ))
                alternatives.append(AlternativeHypothesis(
                    hypothesis_id=f"alt_clamp_{p_claim.claim_id[:8]}",
                    description=f"Mathematical range clamping for {p_claim.target_file or 'target'}",
                    rationale="Guarantees bounded invariant satisfaction deterministically without external dependencies.",
                    plausibility=0.85,
                    required_evidence="Mathematical invariant boundaries and expected return types.",
                    tradeoffs=("Clean and self-contained", "Suppresses out-of-range signals silently if unlogged"),
                ))

        # 4. Calculate reliability score
        total_claims = len(balances)
        if total_claims == 0:
            rel_score = 0.0
            rec = "ABSTAIN"
        else:
            supported_count = sum(1 for b in balances if b.strength in (EvidenceStrength.DEFINITIVE, EvidenceStrength.CORROBORATED))
            contested_count = sum(1 for b in balances if b.strength == EvidenceStrength.CONTESTED)
            stale_count = sum(1 for b in balances if b.strength == EvidenceStrength.STALE)

            if contested_count > 0:
                rel_score = max(0.0, 0.4 - (0.2 * contested_count))
                rec = "ABSTAIN"
            elif stale_count > 0:
                rel_score = 0.5
                rec = "ACQUIRE_EVIDENCE"
            elif supported_count >= (total_claims // 2):
                rel_score = min(1.0, 0.7 + (0.3 * (supported_count / total_claims)))
                rec = "PROCEED"
            else:
                rel_score = 0.6
                rec = "EXPLORE_ALTERNATIVES"

        report_id = f"crit_{hashlib.sha256((artifact.artifact_id + rec).encode('utf-8')).hexdigest()[:10]}"

        return CriticalAnalysisReport(
            report_id=report_id,
            artifact_id=artifact.artifact_id,
            evidence_balances=tuple(balances),
            alternative_hypotheses=tuple(alternatives),
            vulnerable_assumptions=tuple(vulnerabilities),
            unresolved_contradictions=tuple(contradictions),
            epistemic_reliability_score=rel_score,
            recommendation=rec,
        )
