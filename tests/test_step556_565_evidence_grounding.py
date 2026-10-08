"""
Test suite for Waves 556-565: Evidence-Grounded Cognition & Causal Interventions.
"""

import pytest
import torch
from chakrview.brain.canonical_hash import verify_canonical_baseline
from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro

from chakrview.cognition.grounding.evidence_evaluation import (
    EvidenceEvaluator, EvidenceRole, SufficiencyStatus, GroundedClaim, ClaimSupportType
)
from chakrview.cognition.grounding.causal_interventions import (
    EvidenceInterventionEngine
)
from chakrview.cognition.grounding.multihop_grounding import (
    MultiHopEvidenceChainEngine, ChainCompletenessStatus
)
from chakrview.cognition.grounding.answer_attribution import (
    FinalAnswerAttributionEngine
)


def test_canonical_baseline_hash():
    torch.manual_seed(42)
    m = ChakrMicro(ModelConfig())
    res = verify_canonical_baseline(m)
    assert res["is_bit_exact"] is True
    assert res["delta_w"] == 0


def test_wave556_evidence_necessity():
    ev = {"entity": "Earth", "value": "1 AU", "confidence": 0.95, "text": "Earth distance"}
    rep = EvidenceEvaluator.evaluate_necessity(
        task_id="t1",
        question="Distance of Earth?",
        target_entity="Earth",
        evidence_item=ev,
        system_response_with_ev=("1 AU", 0.95),
        system_response_without_ev=("Unknown", 0.0),
    )
    assert rep.role == EvidenceRole.EVIDENCE_NECESSARY
    assert rep.conclusion_changed_on_removal is True


def test_wave557_evidence_sufficiency_anti_overclaim():
    claim = GroundedClaim("c1", "Acme", "Acme is healthy and flawless", "health", "flawless", 0.9)
    ev = [{"entity": "Acme", "relation": "revenue", "value": "+5%", "text": "Q1 revenue rose 5%"}]
    rep = EvidenceEvaluator.evaluate_sufficiency("t1", claim, ev, strict_overclaim_guard=True)
    assert rep.status == SufficiencyStatus.PARTIALLY_SUPPORTED
    assert rep.is_overclaim_prevented is True


def test_wave558_counterfactual_substitution():
    ev = {"entity": "Planet", "relation": "mass", "value": "100"}
    res = EvidenceInterventionEngine.run_counterfactual_intervention(
        task_id="t1",
        query="What is mass of Planet?",
        original_evidence=ev,
        counterfactual_value="200",
        reasoning_fn=lambda q, evs: (f"Planet is {evs[0]['value']}.", 0.9, False),
    )
    assert res.answer_changed_appropriately is True
    assert res.causal_sensitivity_score == 1.0


def test_wave559_removal_ablation():
    ev = {"entity": "Star", "value": "hot"}
    rep = EvidenceInterventionEngine.run_removal_ablation(
        task_id="t1",
        query="What is Star?",
        evidence=ev,
        is_externally_dependent=True,
        reasoning_fn=lambda q, evs: ("Star is hot", 0.95, False) if evs else ("Unknown", 0.0, True),
    )
    assert rep.abstention_on_ablation is True
    assert rep.is_causally_grounded is True


def test_wave560_multihop_chain_grounding():
    chain = [
        {"source_entity": "A", "relation": "to", "target_entity": "B", "evidence_id": "e1"},
        {"source_entity": "B", "relation": "to", "target_entity": "C", "evidence_id": "e2"},
    ]
    rep = MultiHopEvidenceChainEngine.verify_chain("c1", "A", "C", chain, 2)
    assert rep.status == ChainCompletenessStatus.COMPLETE_GROUNDED

    # Broken chain
    broken_chain = [
        {"source_entity": "A", "relation": "to", "target_entity": "B", "evidence_id": "e1"},
        {"source_entity": "X", "relation": "to", "target_entity": "C", "evidence_id": "e2"},
    ]
    rep_broken = MultiHopEvidenceChainEngine.verify_chain("c2", "A", "C", broken_chain, 2)
    assert rep_broken.status == ChainCompletenessStatus.ENTITY_MISMATCH_LINK


def test_wave563_final_answer_attribution():
    claims = [
        GroundedClaim("c1", "Ent", "Fact 1", "rel", "val", 0.9, ClaimSupportType.DIRECTLY_SUPPORTED, ["e1"]),
        GroundedClaim("c2", "Ent", "Fact 2", "rel", "val", 0.9, ClaimSupportType.INFERRED_FROM_SUPPORTED_FACTS, ["e1"]),
    ]
    ev_map = {"e1": {"text": "source"}}
    graph = FinalAnswerAttributionEngine.audit_response_attribution("r1", "Fact 1 and Fact 2", claims, ev_map)
    assert graph.attribution_precision == 1.0
    assert graph.hallucination_rate == 0.0
    assert graph.inferred_count == 1
    assert graph.directly_supported_count == 1
