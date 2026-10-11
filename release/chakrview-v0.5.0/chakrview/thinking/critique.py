"""
Critique Engine for ChakrView Deliberation (Step 21).

Evaluates candidate solutions, hypotheses, and intermediate steps against
structured quality and safety criteria.

Architectural Rule:
CRITIQUE != AUTHORITY.
Critique evaluates logical consistency and evidence grounding; it does not
authorize capabilities or override system constraints.
"""

from dataclasses import dataclass, field, asdict
from enum import Enum
import time
from typing import Dict, List, Optional, Any
import uuid

from chakrview.thinking.workspace import ThinkingWorkspace


class CritiqueVerdict(str, Enum):
    """Evaluation verdict from the critique engine."""
    PASS = "PASS"                                 # High quality, consistent, evidence-grounded
    WEAK = "WEAK"                                 # Lacks strong evidence or clarity
    CONTRADICTED = "CONTRADICTED"                 # Conflicts with established facts/evidence
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE" # Missing essential premises
    REQUIRES_REVISION = "REQUIRES_REVISION"       # Specific flaws requiring correction
    UNVERIFIED = "UNVERIFIED"                     # Not yet evaluated against invariants


@dataclass
class CritiqueResult:
    """Structured report emitted by the critique engine."""
    critique_id: str = field(default_factory=lambda: f"crt_{uuid.uuid4().hex[:8]}")
    verdict: CritiqueVerdict = CritiqueVerdict.UNVERIFIED
    score: float = 0.5                            # 0.0 to 1.0
    findings: List[str] = field(default_factory=list)
    recommended_action: str = ""
    contradictions_detected: List[str] = field(default_factory=list)
    missing_evidence: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

    def is_passed(self, threshold: float = 0.75) -> bool:
        return self.verdict == CritiqueVerdict.PASS and self.score >= threshold

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["verdict"] = self.verdict.value
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CritiqueResult":
        d = dict(data)
        if "verdict" in d and isinstance(d["verdict"], str):
            d["verdict"] = CritiqueVerdict(d["verdict"])
        return cls(**d)


class CritiqueEngine:
    """
    Evaluates candidate thoughts, conclusions, and hypotheses against workspace constraints.
    """

    def critique(
        self,
        candidate_text: str,
        workspace: ThinkingWorkspace,
        hypothesis_id: Optional[str] = None,
    ) -> CritiqueResult:
        """
        Run structured multi-criteria critique over a candidate solution or thought.
        """
        findings: List[str] = []
        contradictions: List[str] = []
        missing_evidence: List[str] = []

        clean_text = candidate_text.strip()
        if not clean_text:
            return CritiqueResult(
                verdict=CritiqueVerdict.REQUIRES_REVISION,
                score=0.0,
                findings=["Candidate text is empty or blank."],
                recommended_action="Generate non-empty candidate approach.",
            )

        score = 1.0

        # 1. Objective alignment check
        obj_words = set(w.strip("?,.:;!") for w in workspace.objective.lower().split())
        cand_words = set(w.strip("?,.:;!") for w in clean_text.lower().split())
        overlap = len(obj_words.intersection(cand_words))
        has_numeric = any(c.isdigit() for c in clean_text)
        is_math_obj = any(op in workspace.objective.lower() for op in ("calculate", "compute", "+", "-", "*", "/", "math"))

        if overlap == 0 and len(obj_words) > 2 and not (is_math_obj and has_numeric):
            score -= 0.3
            findings.append("Low vocabulary alignment with primary objective.")

        # 2. Contradiction check against workspace evidence and uncertainty
        for ev in workspace.evidence:
            ev_content = ev.get("content", "").lower()
            # Check for direct negation markers on evidence content
            if "not " in ev_content and any(w in clean_text.lower() for w in ev_content.replace("not ", "").split()):
                contradictions.append(f"Potential conflict with evidence: {ev.get('evidence_id')}")
                score -= 0.35

        # 3. Check against active contradictions in workspace uncertainty
        for k in workspace.uncertainty_state:
            if "contradiction" in k.lower():
                contradictions.append(f"Unresolved workspace conflict: {k}")
                score -= 0.2

        # 4. Check evidence grounding
        if not workspace.evidence and not workspace.hypotheses:
            missing_evidence.append("No supporting evidence or active hypotheses in workspace.")
            score -= 0.25

        # 5. Check for fallback or error tokens
        if "error" in clean_text.lower() or "authorization denied" in clean_text.lower():
            findings.append("Candidate output indicates execution or authorization failure.")
            score -= 0.5

        # Bound score in [0.0, 1.0]
        score = max(0.0, min(1.0, score))

        # Determine verdict
        if contradictions:
            verdict = CritiqueVerdict.CONTRADICTED
            action = "Reconsider candidate approach to resolve detected contradiction."
        elif missing_evidence and score < 0.6:
            verdict = CritiqueVerdict.INSUFFICIENT_EVIDENCE
            action = "Gather additional verified evidence before drawing final conclusion."
        elif score < workspace.policy.min_critique_score:
            verdict = CritiqueVerdict.WEAK
            action = "Revise candidate approach to address identified weaknesses."
        else:
            verdict = CritiqueVerdict.PASS
            action = "Candidate satisfies deliberation critique criteria."

        res = CritiqueResult(
            verdict=verdict,
            score=score,
            findings=findings,
            recommended_action=action,
            contradictions_detected=contradictions,
            missing_evidence=missing_evidence,
        )

        # Record in workspace
        workspace.critiques.append(res.to_dict())
        for f in findings:
            workspace.record_weakness(f)

        return res
