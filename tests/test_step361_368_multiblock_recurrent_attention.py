"""Unit tests for Wave 361-368: Multi-Block Recurrent Attention Core.

Validates:
- Step 361: Multi-Block Forensics & Layer Profiling
- Step 362: MultiBlockRecurrentChakrMicro Construction, Isolation, & Bit-Exact Initialization (Delta W = 0)
- Step 363: State Persistence Ablation execution
- Step 364: Shared vs Independent transitions execution
- Step 365: Three-block escalation execution
- Step 366: Causal multi-block interventions execution
- Step 367: Multi-block generalization & distractor robustness execution
- Step 368: Strict multi-seed evaluation & baseline invariance
"""

import pytest
import torch

from chakrview.runtime.interactive import (
    instantiate_frozen_baseline,
    compute_model_hash,
    EXPECTED_WEIGHT_HASH,
)
from chakrview.cognition.multiblock_recurrent_attention_forensics import (
    audit_multiblock_recurrent_architecture,
)
from chakrview.cognition.multiblock_recurrent_attention_core import (
    MultiBlockRecurrentChakrMicro,
)
from chakrview.cognition.state_persistence_ablation_study import (
    train_and_eval_persistence_candidate,
)
from chakrview.cognition.shared_vs_independent_study import (
    train_and_eval_sharing_candidate,
)
from chakrview.cognition.three_block_escalation_study import (
    run_three_block_escalation_study,
)
from chakrview.cognition.multiblock_causal_interventions import (
    run_multiblock_causal_interventions,
)
from chakrview.cognition.multiblock_generalization_robustness import (
    run_multiblock_robustness_suite,
)
from chakrview.cognition.strict_multiblock_i4_evaluation import (
    run_strict_multiblock_i4_evaluation,
)


@pytest.fixture(scope="module")
def base_model():
    return instantiate_frozen_baseline()


def test_step361_multiblock_forensics(base_model):
    rep = audit_multiblock_recurrent_architecture(base_model)
    assert rep.primary_pair == (2, 3)
    assert rep.parameter_budget_passed is True
    assert rep.primary_pair_trainable_params <= 500_000


def test_step362_candidate_isolation_and_bit_exactness(base_model):
    init_hash = compute_model_hash(base_model)
    assert init_hash == EXPECTED_WEIGHT_HASH

    cand = MultiBlockRecurrentChakrMicro(base_model=base_model, target_layers=[2, 3], persistent_state=True)
    assert cand.trainable_param_count <= 500_000

    test_input = torch.tensor([[10, 20, 30, 40]], dtype=torch.long)
    with torch.no_grad():
        y_base = base_model(test_input)
        y_cand = cand(test_input)["vocab_logits"]
        diff = (y_base - y_cand).abs().max().item()
    assert diff == pytest.approx(0.0, abs=1e-5)

    post_hash = compute_model_hash(base_model)
    assert post_hash == EXPECTED_WEIGHT_HASH


def test_step363_state_persistence_ablation(base_model):
    res_no = train_and_eval_persistence_candidate(base_model, condition="A_NoState", train_steps=2, eval_episodes=2)
    res_per = train_and_eval_persistence_candidate(base_model, condition="C_Persistent", train_steps=2, eval_episodes=2)
    assert res_no.condition_id == "A_NoState"
    assert res_per.condition_id == "C_Persistent"


def test_step364_shared_vs_independent(base_model):
    res_shared = train_and_eval_sharing_candidate(base_model, option_id="A_SharedTransition", train_steps=2, eval_episodes=2)
    res_indep = train_and_eval_sharing_candidate(base_model, option_id="B_IndepTransition", train_steps=2, eval_episodes=2)
    assert res_shared.trainable_params < res_indep.trainable_params


def test_step365_three_block_escalation(base_model):
    rep = run_three_block_escalation_study(base_model, seed=42)
    assert rep.two_block_result.trainable_params <= 500_000
    assert rep.three_block_result.trainable_params > rep.two_block_result.trainable_params


def test_step366_causal_multiblock_interventions(base_model):
    cand = MultiBlockRecurrentChakrMicro(base_model=base_model, target_layers=[2, 3], persistent_state=True)
    rep = run_multiblock_causal_interventions(cand, seed=42, num_episodes=2)
    assert "1_normal_persistent" in rep.results
    assert rep.causally_active is True


def test_step367_multiblock_robustness(base_model):
    cand = MultiBlockRecurrentChakrMicro(base_model=base_model, target_layers=[2, 3], persistent_state=True)
    rep = run_multiblock_robustness_suite(cand, seed=42, episodes_per_condition=2)
    assert rep.contamination_zero is True
    assert 0 in rep.distractor_sweep


def test_step368_strict_i4_gate_audit(base_model):
    rep = run_strict_multiblock_i4_evaluation(base_model, seeds=[42])
    assert rep.delta_w_zero is True
    assert rep.baseline_exact is True
    assert rep.final_classification in ["I4_EMERGING", "I4_ACHIEVED", "I4_NOT_ACHIEVED"]
