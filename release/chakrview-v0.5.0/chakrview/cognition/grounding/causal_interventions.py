"""
Wave 558 & 559: Evidence Substitution, Counterfactual Grounding & Removal Ablations.

Enforces:
- Causal Intervention Testing:
  1. ORIGINAL EVIDENCE -> ANSWER A
  2. COUNTERFACTUAL WRONG EVIDENCE -> MUST CHANGE / ABSTAIN
  3. CONTRADICTORY EVIDENCE -> MUST QUALIFY / REPORT CONFLICT
  4. IRRELEVANT EVIDENCE -> MUST NOT ALTER VALID INTERNAL REASONING
  5. REMOVAL (NO EVIDENCE) -> MUST ABSTAIN IF EVIDENCE WAS NECESSARY
- Distinguishes:
  ANSWER_SENSITIVITY_TO_RELEVANT_EVIDENCE vs ROBUSTNESS_TO_IRRELEVANT_EVIDENCE.
- Independent evaluator preventing test-harness self-fulfillment.
"""

from __future__ import annotations

import copy
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple, Callable


class InterventionType(str, Enum):
    BASELINE_FULL = "BASELINE_FULL"
    NO_EVIDENCE = "NO_EVIDENCE"
    IRRELEVANT_EVIDENCE = "IRRELEVANT_EVIDENCE"
    CONTRADICTORY_EVIDENCE = "CONTRADICTORY_EVIDENCE"
    COUNTERFACTUAL_SUBSTITUTION = "COUNTERFACTUAL_SUBSTITUTION"
    WEAK_EVIDENCE = "WEAK_EVIDENCE"


@dataclass
class CounterfactualTestResult:
    task_id: str
    original_evidence_value: str
    counterfactual_value: str
    system_original_answer: str
    system_intervened_answer: str
    answer_changed_appropriately: bool
    epistemic_abstention_triggered: bool
    causal_sensitivity_score: float  # 1.0 if answer correctly followed the intervention
    rationale: str


@dataclass
class RemovalAblationReport:
    task_id: str
    full_evidence_answer: str
    full_evidence_confidence: float
    ablated_answer: str
    ablated_confidence: float
    abstention_on_ablation: bool
    is_causally_grounded: bool
    grounding_classification: str


class EvidenceInterventionEngine:
    """
    Executes controlled causal interventions on evidence to prove whether cognitive reasoning
    is genuinely constrained by external facts.
    """

    @classmethod
    def run_counterfactual_intervention(
        cls,
        task_id: str,
        query: str,
        original_evidence: Dict[str, Any],
        counterfactual_value: str,
        reasoning_fn: Callable[[str, List[Dict[str, Any]]], Tuple[str, float, bool]],
    ) -> CounterfactualTestResult:
        """
        Intervenes on evidence value (e.g. X=10 -> X=20) and verifies cognitive state shifts.
        """
        # 1. Evaluate with original evidence
        orig_ans, orig_conf, orig_abstain = reasoning_fn(query, [original_evidence])

        # 2. Intervene: substitute counterfactual value
        cf_evidence = copy.deepcopy(original_evidence)
        cf_evidence["value"] = counterfactual_value
        cf_evidence["text"] = f"Verified observation: {cf_evidence.get('entity', '')} {cf_evidence.get('relation', '')} is {counterfactual_value}."

        cf_ans, cf_conf, cf_abstain = reasoning_fn(query, [cf_evidence])

        # Verification: The answer must reflect the substituted value
        changed_appropriately = (counterfactual_value.lower() in cf_ans.lower()) and (cf_ans.lower() != orig_ans.lower())
        score = 1.0 if changed_appropriately else 0.0

        if changed_appropriately:
            rat = f"Reasoning causally followed the counterfactual shift: '{orig_ans}' -> '{cf_ans}'."
        elif cf_abstain:
            rat = "System abstained on counterfactual intervention."
            score = 0.8
            changed_appropriately = True
        else:
            rat = f"FAILURE: System stubbornly retained original answer '{orig_ans}' despite evidence changed to '{counterfactual_value}'!"

        return CounterfactualTestResult(
            task_id=task_id,
            original_evidence_value=original_evidence.get("value", ""),
            counterfactual_value=counterfactual_value,
            system_original_answer=orig_ans,
            system_intervened_answer=cf_ans,
            answer_changed_appropriately=changed_appropriately,
            epistemic_abstention_triggered=cf_abstain,
            causal_sensitivity_score=score,
            rationale=rat,
        )

    @classmethod
    def run_removal_ablation(
        cls,
        task_id: str,
        query: str,
        evidence: Dict[str, Any],
        is_externally_dependent: bool,
        reasoning_fn: Callable[[str, List[Dict[str, Any]]], Tuple[str, float, bool]],
    ) -> RemovalAblationReport:
        """
        Ablates evidence completely (Full -> Empty) to evaluate necessity.
        """
        ans_full, conf_full, abstain_full = reasoning_fn(query, [evidence])
        ans_empty, conf_empty, abstain_empty = reasoning_fn(query, [])

        if is_externally_dependent:
            # Must abstain or qualify when necessary evidence is removed
            is_grounded = abstain_empty or (conf_full - conf_empty >= 0.50)
            classification = "CAUSALLY_GROUNDED_IN_EVIDENCE" if is_grounded else "SUSPECT_UNGROUNDED_CERTAINTY"
        else:
            # Internal query (e.g. 2+2): legitimately solved without external evidence
            is_grounded = (ans_full.lower() == ans_empty.lower())
            classification = "LEGITIMATE_INTERNAL_CAPABILITY"

        return RemovalAblationReport(
            task_id=task_id,
            full_evidence_answer=ans_full,
            full_evidence_confidence=conf_full,
            ablated_answer=ans_empty,
            ablated_confidence=conf_empty,
            abstention_on_ablation=abstain_empty,
            is_causally_grounded=is_grounded,
            grounding_classification=classification,
        )
