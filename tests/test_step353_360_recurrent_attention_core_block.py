"""Unit tests for Wave 353-360: Recurrent Attention Core Block.

Validates:
- Step 353: Core Block Forensics & Parameter Budget (<= 250k)
- Step 354: RecurrentAttentionCoreBlock initialization, dimensions & causal attention
- Step 355: RecurrentAttentionChakrMicro clone isolation, bit-exact initialization (Delta W = 0)
- Step 356: State-transition ablation study execution
- Step 357: Causal state interventions execution
- Step 358: Distractor robustness & anti-shortcut suite execution
- Step 359 & 360: Strict multi-seed evaluation & baseline invariance
"""

import pytest
import torch

from chakrview.brain.config import ModelConfig
from chakrview.runtime.interactive import (
    instantiate_frozen_baseline,
    compute_model_hash,
    EXPECTED_WEIGHT_HASH,
)
from chakrview.cognition.recurrent_attention_core_block_forensics import (
    audit_core_block_architecture,
    compute_canonical_block_parameter_counts,
)
from chakrview.cognition.recurrent_attention_core_block import (
    RecurrentAttentionCoreBlock,
)
from chakrview.cognition.recurrent_attention_chakr_micro import (
    RecurrentAttentionChakrMicro,
)
from chakrview.cognition.state_transition_ablation_study import (
    train_and_eval_ablation_candidate,
)
from chakrview.cognition.causal_state_interventions import (
    run_causal_state_interventions,
)
from chakrview.cognition.core_block_anti_shortcut_suite import (
    run_distractor_and_anti_shortcut_suite,
)
from chakrview.cognition.strict_recurrent_i4_evaluation import (
    run_strict_recurrent_i4_evaluation,
)


@pytest.fixture(scope="module")
def base_model():
    return instantiate_frozen_baseline()


def test_step353_core_block_forensics():
    counts = compute_canonical_block_parameter_counts()
    assert counts["total"] == 442_752
    rep = audit_core_block_architecture()
    assert rep.parameter_budget_passed is True
    assert rep.proposed_recurrent_block_params <= 250_000


def test_step354_recurrent_attention_core_block():
    config = ModelConfig()
    block = RecurrentAttentionCoreBlock(config=config, d_state=48, freeze_ffn=True)
    assert block.trainable_param_count <= 250_000

    x = torch.randn(2, 6, config.d_model)
    out, trace = block(x, return_trace=True)
    assert out.shape == x.shape
    assert trace is not None
    assert trace.s1.shape == (2, 48)
    assert trace.q2.shape == (2, 6, config.d_model)


def test_step355_candidate_isolation_and_bit_exactness(base_model):
    init_hash = compute_model_hash(base_model)
    assert init_hash == EXPECTED_WEIGHT_HASH

    candidate = RecurrentAttentionChakrMicro(base_model=base_model)
    assert candidate.trainable_param_count <= 250_000

    test_input = torch.tensor([[10, 20, 30, 40]], dtype=torch.long)
    with torch.no_grad():
        y_base = base_model(test_input)
        y_cand = candidate(test_input)["vocab_logits"]
        diff = (y_base - y_cand).abs().max().item()
    assert diff == pytest.approx(0.0, abs=1e-5)

    post_hash = compute_model_hash(base_model)
    assert post_hash == EXPECTED_WEIGHT_HASH


def test_step356_state_transition_ablation(base_model):
    res_a = train_and_eval_ablation_candidate(base_model, condition="A_SingleCycle", train_steps=2, eval_episodes=2)
    res_c = train_and_eval_ablation_candidate(base_model, condition="C_FullRecurrent", train_steps=2, eval_episodes=2)
    assert res_a.condition_id == "A_SingleCycle"
    assert res_c.condition_id == "C_FullRecurrent"


def test_step357_causal_state_interventions(base_model):
    candidate = RecurrentAttentionChakrMicro(base_model=base_model)
    rep = run_causal_state_interventions(candidate, seed=42, num_episodes=2)
    assert "1_normal_s1" in rep.results
    assert "2_zero_s1" in rep.results
    assert rep.causally_dependent_on_state is True


def test_step358_robustness_and_anti_shortcut(base_model):
    candidate = RecurrentAttentionChakrMicro(base_model=base_model)
    rep = run_distractor_and_anti_shortcut_suite(candidate, seed=42, episodes_per_condition=2)
    assert rep.contamination_zero is True
    assert 0 in rep.distractor_sweep
    assert 5 in rep.distractor_sweep


def test_step359_360_strict_i4_gate_audit(base_model):
    rep = run_strict_recurrent_i4_evaluation(base_model, seeds=[42])
    assert rep.delta_w_zero is True
    assert rep.baseline_exact is True
    assert rep.final_classification in ["I4_EMERGING", "I4_ACHIEVED", "I4_NOT_ACHIEVED"]
