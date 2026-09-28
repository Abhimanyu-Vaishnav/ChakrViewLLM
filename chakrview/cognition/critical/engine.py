"""
Critical Thinking Engine for ChakrView (Step 23).

Implements the bounded, deterministic critical thinking protocol:
    QUESTION
       ↓
    UNDERSTAND
       ↓
    DECOMPOSE
       ↓
    IDENTIFY ASSUMPTIONS
       ↓
    GENERATE HYPOTHESES
       ↓
    COLLECT / ORGANIZE EVIDENCE
       ↓
    CHECK EVIDENCE QUALITY
       ↓
    SEARCH FOR COUNTER-EVIDENCE
       ↓
    GENERATE ALTERNATIVE EXPLANATIONS
       ↓
    CHECK CONTRADICTIONS
       ↓
    VERIFY
       ↓
    REVISE
       ↓
    DECIDE OR REMAIN UNCERTAIN

Anti-Confirmation-Bias Invariants:
1. The system explicitly challenges its own candidate hypothesis.
2. Formulates supporting and contradicting evidence requirements.
3. Formulates at least one alternative explanation when feasible.
4. Never fabricates evidence: missing evidence has evidence_available = False.
5. Never fabricates counter-evidence: if unavailable, status = NOT_AVAILABLE.
6. Absence of evidence is NEVER converted into evidence.
7. Preserves authority boundaries: CRITICAL THINKING != CAPABILITY AUTHORITY.
"""

from dataclasses import dataclass, field
import time
from typing import Dict, List, Optional, Any, Tuple
import uuid

from chakrview.cognition.critical.models import (
    Hypothesis,
    Evidence,
    Assumption,
    CounterEvidence,
    AlternativeExplanation,
    Contradiction,
    VerificationResult,
    CriticalThinkingTrace,
    EvidenceStatus,
    CounterEvidenceStatus,
    HypothesisStatus,
    ContradictionSeverity,
    AssumptionCriticality,
)


@dataclass
class CriticalThinkingConfig:
    """Bounded operational parameters for critical thinking execution."""
    max_hypotheses: int = 4
    max_evidence_items: int = 15
    max_revision_cycles: int = 2
    max_assumptions: int = 8
    max_alternatives: int = 3
    require_alternative_explanations: bool = True
    strict_falsification: bool = True
    min_confidence_to_decide: float = 0.65


class CriticalThinkingEngine:
    """
    Sovereign, inspectable critical thinking processor.
    """

    def __init__(
        self,
        config: Optional[CriticalThinkingConfig] = None,
        capability_gate: Optional[Any] = None,
    ) -> None:
        self.config = config or CriticalThinkingConfig()
        self.capability_gate = capability_gate

    def execute(
        self,
        question: str,
        initial_evidence: Optional[List[Evidence]] = None,
        context_assumptions: Optional[List[str]] = None,
        owner_id: str = "default_user",
        session_id: str = "default_session",
    ) -> CriticalThinkingTrace:
        """
        Execute the complete 13-stage critical thinking workflow.
        """
        t0 = time.perf_counter()
        trace = CriticalThinkingTrace(
            question=question.strip(),
            metadata={"owner_id": owner_id, "session_id": session_id},
        )

        # Stage 1: QUESTION
        clean_q = question.strip()
        trace.step_records.append({"stage": "QUESTION", "input": clean_q, "valid": bool(clean_q)})
        if not clean_q:
            trace.decision = "No question provided for critical examination."
            trace.uncertainty_acknowledged = True
            trace.execution_time_ms = (time.perf_counter() - t0) * 1000.0
            return trace

        # Stage 2: UNDERSTAND
        problem_domain, target_claim = self._understand(clean_q)
        trace.step_records.append({
            "stage": "UNDERSTAND",
            "domain": problem_domain,
            "target_claim": target_claim,
        })

        # Stage 3: DECOMPOSE
        sub_problems = self._decompose(clean_q, target_claim)
        trace.step_records.append({
            "stage": "DECOMPOSE",
            "sub_problems": sub_problems,
        })

        # Stage 4: IDENTIFY ASSUMPTIONS
        assumptions = self._identify_assumptions(clean_q, context_assumptions)
        trace.assumptions = assumptions[: self.config.max_assumptions]
        trace.step_records.append({
            "stage": "IDENTIFY_ASSUMPTIONS",
            "count": len(trace.assumptions),
            "assumptions": [a.to_dict() for a in trace.assumptions],
        })

        # Stage 5: GENERATE HYPOTHESES (including primary candidate)
        hypotheses = self._generate_hypotheses(clean_q, target_claim, trace.assumptions)
        trace.hypotheses = hypotheses[: self.config.max_hypotheses]
        trace.step_records.append({
            "stage": "GENERATE_HYPOTHESES",
            "count": len(trace.hypotheses),
            "hypotheses": [h.statement for h in trace.hypotheses],
        })

        # Stage 6: COLLECT / ORGANIZE EVIDENCE
        evidence_pool = self._collect_evidence(initial_evidence, clean_q)
        trace.evidence_pool = evidence_pool[: self.config.max_evidence_items]
        trace.step_records.append({
            "stage": "COLLECT_EVIDENCE",
            "count": len(trace.evidence_pool),
            "available_count": sum(1 for e in trace.evidence_pool if e.evidence_available),
        })

        # Stage 7: CHECK EVIDENCE QUALITY
        self._check_evidence_quality(trace.evidence_pool)
        trace.step_records.append({
            "stage": "CHECK_EVIDENCE_QUALITY",
            "verified_count": sum(1 for e in trace.evidence_pool if e.is_verified),
        })

        # Stage 8: SEARCH FOR COUNTER-EVIDENCE (Anti-Confirmation Bias)
        counter_evidence = self._search_counter_evidence(trace.hypotheses, trace.evidence_pool)
        trace.counter_evidence = counter_evidence
        trace.step_records.append({
            "stage": "SEARCH_COUNTER_EVIDENCE",
            "count": len(trace.counter_evidence),
            "statuses": [c.status.value for c in trace.counter_evidence],
        })

        # Stage 9: GENERATE ALTERNATIVE EXPLANATIONS
        alternatives = self._generate_alternative_explanations(trace.hypotheses, trace.evidence_pool)
        trace.alternatives = alternatives[: self.config.max_alternatives]
        trace.step_records.append({
            "stage": "GENERATE_ALTERNATIVE_EXPLANATIONS",
            "count": len(trace.alternatives),
            "alternatives": [alt.explanation for alt in trace.alternatives],
        })

        # Stage 10: CHECK CONTRADICTIONS
        contradictions = self._check_contradictions(
            trace.hypotheses, trace.evidence_pool, trace.assumptions, trace.counter_evidence
        )
        trace.contradictions = contradictions
        trace.step_records.append({
            "stage": "CHECK_CONTRADICTIONS",
            "count": len(trace.contradictions),
            "severities": [c.severity.value for c in trace.contradictions],
        })

        # Stage 11: VERIFY
        verifications = self._verify_hypotheses(
            trace.hypotheses,
            trace.evidence_pool,
            trace.assumptions,
            trace.counter_evidence,
            trace.contradictions,
        )
        trace.verification_results = verifications
        trace.step_records.append({
            "stage": "VERIFY",
            "results": [v.to_dict() for v in trace.verification_results],
        })

        # Stage 12: REVISE (Bounded cycles)
        self._revise_if_needed(trace)

        # Stage 13: DECIDE OR REMAIN UNCERTAIN
        self._decide_or_remain_uncertain(trace)

        trace.execution_time_ms = (time.perf_counter() - t0) * 1000.0
        trace.safe_summary = self._generate_safe_summary(trace)
        return trace

    # -------------------------------------------------------------------------
    # Internal Protocol Stages
    # -------------------------------------------------------------------------

    def _understand(self, query: str) -> Tuple[str, str]:
        """Classify problem domain and isolate target assertion/claim."""
        q_lower = query.lower()
        if any(w in q_lower for w in ["calculate", "math", "add", "sum", "multiply", "=", "+"]):
            domain = "mathematical_deduction"
        elif any(w in q_lower for w in ["why", "cause", "reason", "because"]):
            domain = "causal_inquiry"
        elif any(w in q_lower for w in ["compare", "vs", "versus", "difference"]):
            domain = "comparative_analysis"
        elif any(w in q_lower for w in ["predict", "future", "will"]):
            domain = "predictive_evaluation"
        else:
            domain = "epistemic_assertion"

        target_claim = query.strip()
        return domain, target_claim

    def _decompose(self, query: str, target_claim: str) -> List[str]:
        """Break down the inquiry into explicit falsifiable questions."""
        sub_problems = []
        if "?" in query:
            parts = [p.strip() for p in query.split("?") if p.strip()]
            sub_problems.extend(parts)
        elif " and " in query.lower():
            parts = [p.strip() for p in query.split(" and ") if p.strip()]
            sub_problems.extend(parts)
        else:
            sub_problems.append(f"Is '{target_claim}' logically consistent?")
            sub_problems.append(f"What empirical evidence corroborates or refutes '{target_claim}'?")
        return sub_problems

    def _identify_assumptions(
        self, query: str, context_assumptions: Optional[List[str]] = None
    ) -> List[Assumption]:
        """Extract explicit and implicit assumptions."""
        assumptions: List[Assumption] = []
        if context_assumptions:
            for stmt in context_assumptions:
                assumptions.append(
                    Assumption(
                        statement=stmt,
                        is_explicit=True,
                        plausibility=0.6,
                        criticality=AssumptionCriticality.HIGH,
                        falsifiable=True,
                        falsification_condition=f"Demonstrate condition where '{stmt}' does not hold.",
                    )
                )

        # Natural heuristic assumption extraction
        q_lower = query.lower()
        if "always" in q_lower or "all" in q_lower:
            assumptions.append(
                Assumption(
                    statement="Universal applicability (claim holds without exceptions)",
                    is_explicit=False,
                    plausibility=0.4,
                    criticality=AssumptionCriticality.HIGH,
                    falsifiable=True,
                    falsification_condition="Find a single counter-example.",
                )
            )
        if "best" in q_lower or "optimal" in q_lower:
            assumptions.append(
                Assumption(
                    statement="Single objective optimization metric exists",
                    is_explicit=False,
                    plausibility=0.5,
                    criticality=AssumptionCriticality.MEDIUM,
                    falsifiable=True,
                    falsification_condition="Identify competing trade-offs.",
                )
            )

        if not assumptions:
            assumptions.append(
                Assumption(
                    statement="Premises provided in query accurately describe reality",
                    is_explicit=False,
                    plausibility=0.7,
                    criticality=AssumptionCriticality.MEDIUM,
                    falsifiable=True,
                    falsification_condition="Verify premise against ground truth.",
                )
            )
        return assumptions

    def _generate_hypotheses(
        self, query: str, target_claim: str, assumptions: List[Assumption]
    ) -> List[Hypothesis]:
        """Formulate primary candidate hypothesis and alternative positions."""
        hypotheses: List[Hypothesis] = []

        # Primary candidate
        primary = Hypothesis(
            statement=target_claim,
            prior_plausibility=0.6,
            assumptions=list(assumptions),
            status=HypothesisStatus.CANDIDATE,
            supporting_requirements=[f"Corroborating evidence for: {target_claim}"],
            contradicting_requirements=[f"Any factual counter-evidence against: {target_claim}"],
            verification_requirements=["Logical consistency", "Non-contradiction with evidence"],
        )
        hypotheses.append(primary)

        # Counter-hypothesis / null hypothesis
        negated = Hypothesis(
            statement=f"Negation: {target_claim} is invalid or incomplete",
            prior_plausibility=0.4,
            status=HypothesisStatus.CANDIDATE,
            supporting_requirements=[f"Counter-evidence against: {target_claim}"],
            contradicting_requirements=[f"Definitive proof of: {target_claim}"],
            verification_requirements=["Identify failure mode or boundary exception"],
        )
        hypotheses.append(negated)
        return hypotheses

    def _collect_evidence(
        self, initial_evidence: Optional[List[Evidence]], query: str
    ) -> List[Evidence]:
        """Organize available evidence, preserving evidence_available flag."""
        pool: List[Evidence] = []
        if initial_evidence:
            for ev in initial_evidence:
                pool.append(ev)

        # If pool is empty, explicitly record unavailability of empirical evidence
        if not pool:
            pool.append(
                Evidence(
                    content="No external factual evidence provided in initial pool.",
                    source="runtime_check",
                    reliability=1.0,
                    is_verified=True,
                    evidence_available=False,  # Strict: explicitly marked unavailable
                    metadata={"query": query},
                )
            )
        return pool

    def _check_evidence_quality(self, evidence_pool: List[Evidence]) -> None:
        """Evaluate reliability and provenance of evidence."""
        for ev in evidence_pool:
            if not ev.evidence_available:
                ev.reliability = 0.0
                ev.is_verified = False
            elif ev.source in ["axiom", "math_engine", "state_telemetry"]:
                ev.reliability = max(ev.reliability, 0.95)
                ev.is_verified = True
            elif ev.source in ["user", "unverified_input"]:
                ev.reliability = min(ev.reliability, 0.70)
                ev.is_verified = False

    def _search_counter_evidence(
        self, hypotheses: List[Hypothesis], evidence_pool: List[Evidence]
    ) -> List[CounterEvidence]:
        """
        Anti-Confirmation Bias: Proactively search for counter-evidence against every hypothesis.
        If none found, status is NOT_AVAILABLE or NONE_FOUND (never fabricated).
        """
        counter_list: List[CounterEvidence] = []

        for hyp in hypotheses:
            found_counters = False
            for ev in evidence_pool:
                if not ev.evidence_available:
                    continue

                # Check if evidence contradicts the statement
                ev_lower = ev.content.lower()
                hyp_lower = hyp.statement.lower()

                # Basic heuristic check for contradiction
                if ("not " in ev_lower or "false" in ev_lower or "contradict" in ev_lower) and any(
                    word in ev_lower for word in hyp_lower.split() if len(word) > 4
                ):
                    counter = CounterEvidence(
                        target_hypothesis_id=hyp.hypothesis_id,
                        content=ev.content,
                        source=ev.source,
                        strength=ev.reliability,
                        status=CounterEvidenceStatus.IDENTIFIED,
                        falsifies_target=ev.reliability > 0.85,
                    )
                    counter_list.append(counter)
                    hyp.contradicting_evidence_ids.append(counter.counter_id)
                    found_counters = True

            if not found_counters:
                # Strictly report NOT_AVAILABLE
                counter = CounterEvidence(
                    target_hypothesis_id=hyp.hypothesis_id,
                    content="",
                    source="system_search",
                    strength=0.0,
                    status=CounterEvidenceStatus.NOT_AVAILABLE,
                    falsifies_target=False,
                )
                counter_list.append(counter)

        return counter_list

    def _generate_alternative_explanations(
        self, hypotheses: List[Hypothesis], evidence_pool: List[Evidence]
    ) -> List[AlternativeExplanation]:
        """Formulate at least one alternative account for the primary hypothesis."""
        alternatives: List[AlternativeExplanation] = []
        if not hypotheses:
            return alternatives

        primary = hypotheses[0]
        alt1 = AlternativeExplanation(
            target_hypothesis_id=primary.hypothesis_id,
            explanation=f"Alternative: The observed outcome in '{primary.statement}' is caused by unmodeled latent factors.",
            plausibility=0.5,
            distinguishing_tests=["Isolate independent variables", "Verify boundary conditions"],
            evidence_requirements=["Measurement of latent variables"],
        )
        alternatives.append(alt1)
        primary.alternative_explanation_ids.append(alt1.alt_id)
        return alternatives

    def _check_contradictions(
        self,
        hypotheses: List[Hypothesis],
        evidence_pool: List[Evidence],
        assumptions: List[Assumption],
        counter_evidence: List[CounterEvidence],
    ) -> List[Contradiction]:
        """Detect pairwise tensions between evidence, assumptions, and hypotheses."""
        contradictions: List[Contradiction] = []

        # Check assumptions against counter-evidence
        for cnt in counter_evidence:
            if cnt.status == CounterEvidenceStatus.IDENTIFIED and cnt.falsifies_target:
                for hyp in hypotheses:
                    if hyp.hypothesis_id == cnt.target_hypothesis_id:
                        contradictions.append(
                            Contradiction(
                                source_a_id=hyp.hypothesis_id,
                                source_b_id=cnt.counter_id,
                                description=f"Hypothesis '{hyp.statement}' directly disputed by counter-evidence: {cnt.content}",
                                severity=ContradictionSeverity.HIGH,
                                resolved=False,
                            )
                        )

        # Check assumption fragility
        for asm in assumptions:
            if asm.criticality == AssumptionCriticality.HIGH and asm.plausibility < 0.4:
                contradictions.append(
                    Contradiction(
                        source_a_id=asm.assumption_id,
                        source_b_id="epistemic_standard",
                        description=f"High criticality assumption '{asm.statement}' has low plausibility ({asm.plausibility}).",
                        severity=ContradictionSeverity.MEDIUM,
                        resolved=False,
                    )
                )

        return contradictions

    def _verify_hypotheses(
        self,
        hypotheses: List[Hypothesis],
        evidence_pool: List[Evidence],
        assumptions: List[Assumption],
        counter_evidence: List[CounterEvidence],
        contradictions: List[Contradiction],
    ) -> List[VerificationResult]:
        """Perform multi-criteria evaluation of each hypothesis."""
        results: List[VerificationResult] = []

        for hyp in hypotheses:
            passed = []
            failed = []

            # 1. Evidence availability check
            available_ev = [e for e in evidence_pool if e.evidence_available]
            if not available_ev:
                failed.append("no_available_evidence")
                hyp.missing_information.append("Factual grounding evidence is currently unavailable.")
            else:
                passed.append("evidence_available")

            # 2. Counter-evidence check
            active_counters = [
                c for c in counter_evidence
                if c.target_hypothesis_id == hyp.hypothesis_id and c.status in [CounterEvidenceStatus.IDENTIFIED, CounterEvidenceStatus.CONFIRMED]
            ]
            if active_counters:
                failed.append("counter_evidence_present")
                counter_status = CounterEvidenceStatus.IDENTIFIED
            else:
                passed.append("no_active_counter_evidence")
                counter_status = CounterEvidenceStatus.NOT_AVAILABLE

            # 3. Contradiction count
            hyp_contradictions = [c for c in contradictions if c.source_a_id == hyp.hypothesis_id or c.source_b_id == hyp.hypothesis_id]
            if hyp_contradictions:
                failed.append(f"unresolved_contradictions_{len(hyp_contradictions)}")
            else:
                passed.append("no_contradictions")

            # 4. Assumption validation
            unfalsifiable_assumptions = [a for a in hyp.assumptions if not a.falsifiable]
            if unfalsifiable_assumptions:
                failed.append("unfalsifiable_assumptions")
            else:
                passed.append("assumptions_falsifiable")

            verified = (len(failed) == 0 and len(available_ev) > 0)

            # Compute calibrated confidence
            conf = hyp.prior_plausibility
            if "no_available_evidence" in failed:
                conf = min(conf, 0.45)
            if active_counters:
                conf = max(0.1, conf - 0.35)
            if hyp_contradictions:
                conf = max(0.1, conf - 0.25)
            if verified:
                conf = min(0.95, conf + 0.25)

            hyp.confidence = round(conf, 4)
            if verified:
                hyp.status = HypothesisStatus.SUPPORTED
            elif active_counters:
                hyp.status = HypothesisStatus.CHALLENGED
            elif not available_ev or hyp_contradictions:
                hyp.status = HypothesisStatus.UNCERTAIN

            res = VerificationResult(
                verified=verified,
                passed_checks=passed,
                failed_checks=failed,
                contradictions_detected=len(hyp_contradictions),
                assumptions_validated=len(unfalsifiable_assumptions) == 0,
                counter_evidence_status=counter_status,
                summary=f"Hypothesis '{hyp.statement[:40]}...': status={hyp.status.value}, confidence={hyp.confidence}",
                confidence_score=hyp.confidence,
                details={"passed": passed, "failed": failed},
            )
            results.append(res)

        return results

    def _revise_if_needed(self, trace: CriticalThinkingTrace) -> None:
        """Bounded revision of hypotheses if critical flaws were identified."""
        for cycle in range(self.config.max_revision_cycles):
            for hyp in trace.hypotheses:
                if hyp.status == HypothesisStatus.CHALLENGED:
                    # Narrow scope or revise to incorporate counter-evidence
                    revised_stmt = f"Qualified: {hyp.statement} (subject to counter-evidence exceptions)"
                    hyp.statement = revised_stmt
                    hyp.status = HypothesisStatus.UNCERTAIN
                    trace.step_records.append({
                        "stage": "REVISE",
                        "cycle": cycle + 1,
                        "revised_hypothesis": hyp.hypothesis_id,
                        "new_statement": revised_stmt,
                    })

    def _decide_or_remain_uncertain(self, trace: CriticalThinkingTrace) -> None:
        """Formulate final decision or honestly declare epistemic uncertainty."""
        if not trace.hypotheses:
            trace.decision = "No hypotheses available."
            trace.uncertainty_acknowledged = True
            return

        supported_hypotheses = [h for h in trace.hypotheses if h.status == HypothesisStatus.SUPPORTED and h.confidence >= self.config.min_confidence_to_decide]

        if supported_hypotheses and not any(c.severity in [ContradictionSeverity.HIGH, ContradictionSeverity.FATAL] for c in trace.contradictions):
            best = max(supported_hypotheses, key=lambda h: h.confidence)
            trace.decision = f"DECIDED: {best.statement} (confidence: {best.confidence:.2f})"
            trace.uncertainty_acknowledged = False
        else:
            # Epistemic honesty: UNKNOWN != FALSE
            primary = trace.hypotheses[0]
            reasons = []
            if any(c.severity == ContradictionSeverity.HIGH for c in trace.contradictions):
                reasons.append("unresolved high-severity contradictions")
            if any(not e.evidence_available for e in trace.evidence_pool):
                reasons.append("empirical evidence is unavailable or insufficient")
            if primary.status in [HypothesisStatus.UNCERTAIN, HypothesisStatus.CHALLENGED]:
                reasons.append(f"candidate hypothesis status is {primary.status.value}")

            reason_str = ", ".join(reasons) if reasons else "confidence below decision threshold"
            trace.decision = f"UNCERTAIN: Insufficient certainty to conclude '{primary.statement}'. Key factors: {reason_str}."
            trace.uncertainty_acknowledged = True

    def _generate_safe_summary(self, trace: CriticalThinkingTrace) -> str:
        """Create a safe public summary without exposing raw chain-of-thought."""
        return (
            f"Critical Thinking Summary:\n"
            f"- Question: {trace.question}\n"
            f"- Examined Hypotheses: {len(trace.hypotheses)}\n"
            f"- Assessed Assumptions: {len(trace.assumptions)}\n"
            f"- Counter-Evidence Status: {trace.counter_evidence[0].status.value if trace.counter_evidence else 'NONE'}\n"
            f"- Detected Contradictions: {len(trace.contradictions)}\n"
            f"- Outcome: {trace.decision}\n"
            f"- Uncertainty Explicitly Acknowledged: {trace.uncertainty_acknowledged}"
        )
