from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from typing import Dict, List, Optional

from chakrview.cognition.repository.semantic_record import RepositorySemanticRecord
from chakrview.cognition.repository.memory_index import RepositoryMemoryIndex


@dataclass(frozen=True)
class ArbitrationWeights:
    """Explicit, documented scoring weights.

    Derivation (see docs/STEP_60_SEMANTIC_MEMORY_ARCHITECTURE.md):
    Step 58 base formula: w_fam=0.40, w_diag=0.40, w_ev=0.20, w_pen=0.50
    Step 60 carves within these buckets without exceeding total positive=1.0:
      task_family    0.30  (was 0.40; carved 0.10 for language)
      language       0.10  (new Step 60 signal)
      framework      0.10  (new Step 60 signal; carved from w_ev)
      symptom_diag   0.20  (partial w_diag)
      dependency     0.10  (new; carved from w_diag)
      module_overlap 0.10  (new; carved from w_diag)
      evidence       0.10  (was 0.20; carved 0.10 for framework)
      penalty        0.50  (unchanged from Step 58)
    """
    w_family: float = 0.30
    w_language: float = 0.10
    w_framework: float = 0.10
    w_symptom: float = 0.20
    w_dependency: float = 0.10
    w_module_overlap: float = 0.10
    w_evidence: float = 0.10
    w_penalty: float = 0.50
    confidence_threshold: float = 0.75
    conflict_delta: float = 0.05


DEFAULT_WEIGHTS = ArbitrationWeights()


@dataclass(frozen=True)
class RepositoryQuery:
    """All information the arbitration layer is allowed to use as query inputs."""
    task_family: str
    language: str
    framework: str
    symptom_signature: str
    root_cause_signature: Optional[str] = None
    dependency_signature: Optional[str] = None
    affected_modules: Optional[List[str]] = None


@dataclass
class ArbitrationSignalBreakdown:
    """Per-candidate, per-signal explainable score record."""
    memory_id: str
    task_family_score: float
    language_score: float
    framework_score: float
    symptom_score: float
    dependency_score: float
    module_overlap_score: float
    evidence_score: float
    penalty: float
    final_score: float
    signal_notes: List[str] = field(default_factory=list)

    def format_trace(self) -> str:
        w = DEFAULT_WEIGHTS
        neg_pen = -self.penalty
        rows = [
            "Candidate " + self.memory_id,
            "-" * 40,
            "  task_family      " + ("%+.4f" % (self.task_family_score * w.w_family)),
            "  language         " + ("%+.4f" % (self.language_score * w.w_language)),
            "  framework        " + ("%+.4f" % (self.framework_score * w.w_framework)),
            "  symptom          " + ("%+.4f" % (self.symptom_score * w.w_symptom)),
            "  dependency       " + ("%+.4f" % (self.dependency_score * w.w_dependency)),
            "  module_overlap   " + ("%+.4f" % (self.module_overlap_score * w.w_module_overlap)),
            "  evidence         " + ("%+.4f" % (self.evidence_score * w.w_evidence)),
            "  boundary_penalty " + ("%+.4f" % neg_pen),
            "-" * 40,
            "  final_score      " + ("%.4f" % self.final_score),
        ]
        for n in self.signal_notes:
            rows.append("  note: " + n)
        return "\n".join(rows)

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ArbitrationCandidate:
    record: RepositorySemanticRecord
    breakdown: ArbitrationSignalBreakdown

    @property
    def score(self) -> float:
        return self.breakdown.final_score

    @property
    def memory_id(self) -> str:
        return self.record.memory_id


class ArbitrationStatus(Enum):
    SELECTED = auto()
    AMBIGUOUS = auto()
    CONFLICTING = auto()
    REJECTED = auto()
    NO_MATCH = auto()


@dataclass
class ArbitrationResult:
    """Complete, deterministic arbitration outcome with full audit trace."""
    status: ArbitrationStatus
    selected: Optional[RepositorySemanticRecord] = None
    candidates: List[ArbitrationCandidate] = field(default_factory=list)
    reason: str = ""
    trace: List[str] = field(default_factory=list)

    def format_report(self) -> str:
        rows = [
            "=" * 60, "Repository Memory Arbitration", "=" * 60,
            "Result: " + self.status.name,
            "Reason: " + self.reason,
            "",
            "Candidates:",
        ]
        for c in self.candidates:
            sel = " (SELECTED)" if (self.selected and c.memory_id == self.selected.memory_id) else ""
            rows.append("  " + c.memory_id + "  score=" + ("%.4f" % c.score) + sel)
        rows.append("")
        for c in self.candidates:
            rows.append(c.breakdown.format_trace())
            rows.append("")
        decision = ("SELECT " + self.selected.memory_id) if self.selected else "ABSTAIN"
        rows.append("Decision: " + decision)
        rows.append("=" * 60)
        return "\n".join(rows)

    def to_dict(self) -> Dict:
        return {
            "status": self.status.name,
            "selected_id": self.selected.memory_id if self.selected else None,
            "reason": self.reason,
            "candidates": [
                {"memory_id": c.memory_id, "score": c.score, "breakdown": c.breakdown.to_dict()}
                for c in self.candidates
            ],
            "trace": self.trace,
        }


def _jaccard(s1: str, s2: str) -> float:
    """Deterministic Jaccard token overlap - matches Step 58 formula exactly."""
    if not s1 or not s2:
        return 0.0
    t = str.maketrans(":.;-_/", "      ")
    t1 = set(s1.lower().translate(t).split())
    t2 = set(s2.lower().translate(t).split())
    if not t1 or not t2:
        return 0.0
    return len(t1 & t2) / len(t1 | t2)


def _score_candidate(
    query: RepositoryQuery,
    record: RepositorySemanticRecord,
    weights: ArbitrationWeights,
) -> ArbitrationSignalBreakdown:
    """Compute full explainable score for one candidate. Every signal is documented."""
    notes: List[str] = []

    # S1 task-family match (binary; Step 58 S_family)
    fam = 1.0 if record.task_family == query.task_family else 0.0
    if not fam:
        notes.append("family mismatch: " + record.task_family)

    # S2 language match (binary; new Step 60)
    lang = 1.0 if record.language.lower() == query.language.lower() else 0.0
    if not lang:
        notes.append("language mismatch: " + record.language)

    # S3 framework match (binary; new Step 60)
    fw = 1.0 if record.framework.lower() == query.framework.lower() else 0.0
    if not fw:
        notes.append("framework mismatch: " + record.framework)

    # S4 symptom similarity (Jaccard; Step 58 S_diag partial)
    sym = _jaccard(record.symptom_signature, query.symptom_signature)
    if sym < 0.1:
        notes.append("low symptom similarity")

    # S5 dependency signature similarity (Jaccard; new Step 60)
    dep = _jaccard(record.dependency_signature, query.dependency_signature) if query.dependency_signature else 0.0

    # S6 affected module overlap (binary any-overlap; new Step 60)
    if query.affected_modules and record.affected_modules:
        mod = 1.0 if any(m in record.affected_modules for m in query.affected_modules) else 0.0
    else:
        mod = 0.0
    if not mod:
        notes.append("no module overlap")

    # S7 evidence-weighted trust (Step 58: min(1, count/5) * success_rate)
    total_ep = record.successful_episodes + record.failed_episodes
    success_rate = record.successful_episodes / max(1, total_ep)
    ev = min(1.0, record.evidence_count / 5.0) * success_rate

    # S8 boundary penalty (Step 58 S_mismatch)
    penalty = 0.0
    for boundary in record.known_boundaries:
        if query.symptom_signature and boundary.lower() in query.symptom_signature.lower():
            penalty += weights.w_penalty
            notes.append("boundary violated: " + boundary)
        if "do not apply" in boundary.lower() and query.task_family in boundary.lower():
            penalty += weights.w_penalty
            notes.append("exclusion boundary: " + boundary)

    raw = (
        weights.w_family * fam
        + weights.w_language * lang
        + weights.w_framework * fw
        + weights.w_symptom * sym
        + weights.w_dependency * dep
        + weights.w_module_overlap * mod
        + weights.w_evidence * ev
        - penalty
    )
    return ArbitrationSignalBreakdown(
        memory_id=record.memory_id,
        task_family_score=fam,
        language_score=lang,
        framework_score=fw,
        symptom_score=sym,
        dependency_score=dep,
        module_overlap_score=mod,
        evidence_score=ev,
        penalty=penalty,
        final_score=max(0.0, round(raw, 6)),
        signal_notes=notes,
    )


def arbitrate(
    query: RepositoryQuery,
    index: RepositoryMemoryIndex,
    weights: ArbitrationWeights = DEFAULT_WEIGHTS,
) -> ArbitrationResult:
    """
    End-to-end deterministic arbitration pipeline:
    QUERY -> candidate_set -> score -> rank -> conflict_check -> threshold -> decision
    """
    trace: List[str] = []

    # 1. Candidate generation (exact field matching via index)
    raw = index.candidate_set(
        task_family=query.task_family,
        language=query.language,
        framework=query.framework,
    )
    trace.append("candidate_set: " + str(len(raw)) + " after exact filters")

    if not raw:
        return ArbitrationResult(
            status=ArbitrationStatus.NO_MATCH,
            reason="No records matched family=" + query.task_family + " lang=" + query.language + " fw=" + query.framework,
            trace=trace,
        )

    # 2. Score every candidate
    scored: List[ArbitrationCandidate] = [
        ArbitrationCandidate(record=r, breakdown=_score_candidate(query, r, weights))
        for r in raw
    ]
    trace.append("scored: " + str(len(scored)) + " candidates")

    # 3. Rank (deterministic: desc score, asc memory_id for tie-break)
    scored.sort(key=lambda c: (-c.score, c.memory_id))
    for i, c in enumerate(scored):
        trace.append("  rank " + str(i + 1) + ": " + c.memory_id + " score=" + ("%.4f" % c.score))

    top = scored[0]

    # 4. Safe abstention: threshold check
    if top.score < weights.confidence_threshold:
        return ArbitrationResult(
            status=ArbitrationStatus.REJECTED,
            candidates=scored,
            reason="Top score " + ("%.4f" % top.score) + " below threshold " + ("%.2f" % weights.confidence_threshold) + ". ABSTAIN.",
            trace=trace,
        )

    # 5. Conflict detection (near-tied, incompatible solutions)
    if len(scored) >= 2:
        second = scored[1]
        close = math.isclose(top.score, second.score, abs_tol=weights.conflict_delta)
        diff_sol = top.record.solution_pattern != second.record.solution_pattern
        if close and diff_sol:
            return ArbitrationResult(
                status=ArbitrationStatus.CONFLICTING,
                candidates=scored,
                reason="CONFLICTING: " + top.memory_id + " vs " + second.memory_id + " near-tied but incompatible solutions. ABSTAIN.",
                trace=trace,
            )

    # 6. Ambiguous (tied scores, same solution)
    tied = [
        c for c in scored
        if math.isclose(c.score, top.score, abs_tol=1e-9)
        and c.record.solution_pattern == top.record.solution_pattern
    ]
    if len(tied) > 1:
        return ArbitrationResult(
            status=ArbitrationStatus.AMBIGUOUS,
            selected=top.record,
            candidates=scored,
            reason=str(len(tied)) + " records tied at " + ("%.4f" % top.score) + " with identical solution.",
            trace=trace,
        )

    # 7. Clear winner
    return ArbitrationResult(
        status=ArbitrationStatus.SELECTED,
        selected=top.record,
        candidates=scored,
        reason="SELECT " + top.memory_id + " score=" + ("%.4f" % top.score),
        trace=trace,
    )
