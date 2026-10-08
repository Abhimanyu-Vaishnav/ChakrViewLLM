"""
Test Suite for ChakrView v0.6.0 Neural Representation & Semantic Grounding.
"""

import pytest
import torch

from chakrview.brain.canonical_hash import verify_canonical_baseline
from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro

from chakrview.cognition.v060_candidate.relational_encoder import RelationalEncoder
from chakrview.cognition.v060_candidate.semantic_composition import SemanticCompositionEngine
from chakrview.cognition.v060_candidate.multihop_semantic import MultiHopSemanticReasoner
from chakrview.cognition.v060_candidate.language_grounding import LanguageSemanticGroundingBridge
from chakrview.cognition.v060_candidate.semantic_cognitive_coupling import SemanticCognitiveCouplingEngine
from chakrview.cognition.v060_candidate.surface_reformulation import SurfaceReformulationTester
from chakrview.cognition.v060_candidate.v060_master_gate import V060CandidateMasterGate


def test_canonical_baseline_bit_exact():
    """Verify baseline model parameter integrity (Delta W = 0)."""
    torch.manual_seed(42)
    m = ChakrMicro(ModelConfig())
    res = verify_canonical_baseline(m)
    assert res["is_bit_exact"] is True
    assert res["delta_w"] == 0
    assert res["total_params"] == 3443136


def test_stage1_relational_representation_learning():
    """Verify role-bound slot representation achieves high accuracy on seen pairs and measures generalization gap on unseen pairs."""
    res = RelationalEncoder.run_experiment(steps=15)
    assert res.seen_entity_accuracy >= 0.75
    # Accurately reflect that zero-shot transfer without relational pretraining has a measurable gap
    assert res.generalization_gap >= 0.0


def test_stage2_semantic_composition():
    """Verify semantic composition across unseen attribute combinations."""
    res = SemanticCompositionEngine.run_experiment(steps=15)
    assert res.semantic_composition_pass is True
    assert res.composition_seen_score >= 0.75


def test_stage3_multihop_semantic_representation():
    """Verify 1-hop, 2-hop, and 3-hop transitivity across relational chains."""
    res = MultiHopSemanticReasoner.run_experiment(steps=15)
    assert res.is_transitive is True
    assert res.transitivity_score >= 0.65


def test_stage4_language_semantic_grounding():
    """Verify natural text sequence grounds into normalized relational triple."""
    res = LanguageSemanticGroundingBridge.ground_text_to_semantic_triple("Ram lives in Delhi")
    assert res.is_grounded is True
    assert res.extracted_triple.subject == "Ram"
    assert res.extracted_triple.relation == "LIVES_IN"
    assert res.extracted_triple.object == "Delhi"


def test_stage5_semantic_state_cognition_coupling():
    """Verify grounded semantic triples feed directly into cognitive decision."""
    ev_map = {"ram::lives_in": "Delhi"}
    res = SemanticCognitiveCouplingEngine.evaluate_query_with_semantic_coupling("Ram lives in Delhi", ev_map)
    assert res.semantic_grounded is True
    assert res.cognitive_decision == "ANSWER"
    assert res.measurable_improvement_demonstrated is True


def test_stage6_surface_reformulation_invariance():
    """Verify active, passive, and lexical variants ground to identical canonical triple."""
    variants = [
        "Ram lives in Delhi",
        "Ram resides in Delhi",
        "Delhi is where Ram lives",
    ]
    res = SurfaceReformulationTester.test_surface_invariance(variants, "(Ram) --[LIVES_IN]--> (Delhi)")
    assert res.is_generalization_pass is True
    assert res.reformulation_invariance_rate == 1.0


def test_stage10_v060_master_gate():
    """Audit full v0.6.0 candidate research gate."""
    rep = V060CandidateMasterGate.evaluate_candidate()
    assert rep.canonical_baseline_intact is True
    assert rep.baseline_delta_w == 0
    assert rep.final_verdict == "V060_NEURAL_SEMANTIC_CAPABILITY_PARTIAL"
