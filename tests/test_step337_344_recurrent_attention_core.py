"""Pytest test suite for Wave 337–344: Compact Recurrent Attention Core."""

import pytest
import torch

from chakrview.brain.model import ChakrMicro
from chakrview.brain.config import ModelConfig
from chakrview.runtime.interactive import compute_model_hash, EXPECTED_WEIGHT_HASH
from chakrview.cognition.integrated_recurrent_attention_forensics import (
    run_integrated_core_forensics,
)
from chakrview.cognition.compact_recurrent_attention_core import (
    CompactRecurrentAttentionCore,
)
from chakrview.cognition.value_ffn_learning import (
    run_value_ffn_ablation_study,
)
from chakrview.cognition.recurrent_integration import (
    run_recurrent_integration_comparison,
)
from chakrview.cognition.causal_neural_path_interventions import (
    run_causal_neural_path_interventions,
)
from chakrview.cognition.integrated_anti_shortcut import (
    run_integrated_anti_shortcut_suite,
)
from chakrview.cognition.strict_integrated_i4_evaluation import (
    run_strict_integrated_i4_evaluation,
)
from chakrview.cognition.compositional_benchmark_environment import (
    CompositionalAssociativeEnvironment,
)


@pytest.fixture
def base_model():
    torch.manual_seed(42)
    cfg = ModelConfig()
    model = ChakrMicro(cfg)
    model.eval()
    return model


def test_step337_integrated_core_forensics(base_model):
    """Verifies integrated core forensics runs and confirms baseline exactness."""
    rep = run_integrated_core_forensics(base_model, target_layer=3, seeds=[42], num_episodes=2)
    assert rep.baseline_exact is True
    assert rep.recommended_target_layer == 3
    assert rep.metrics.h1_key_margin is not None
    assert "CompactRecurrentAttentionCore" in rep.architecture_diagram


def test_step338_compact_recurrent_attention_core_identity_and_params(base_model):
    """Verifies core parameters budget (< 120k preferred, < 150k target) and identity initialization."""
    core = CompactRecurrentAttentionCore(
        base_model, target_layer=3,
    )
    p_count = core.trainable_param_count
    assert p_count < 120000
    assert p_count > 40000

    inp = torch.tensor([[5, 12, 19, 26]], dtype=torch.long)
    with torch.no_grad():
        base_logits = base_model(inp)
        core_logits = core(inp)

    # Zero initialized up-projections guarantee numerical identity at init
    assert torch.allclose(base_logits, core_logits, atol=1e-5)


def test_step339_value_ffn_ablation_study(base_model):
    """Verifies content transformation ablation runs across QK, QKV, and QKV+FFN."""
    rep = run_value_ffn_ablation_study(base_model)
    assert len(rep.metrics_by_config) == 3
    assert "QK_only" in rep.metrics_by_config
    assert "QKV_FFN" in rep.metrics_by_config


def test_step340_recurrent_integration(base_model):
    """Verifies static vs recurrent comparison runs."""
    rep = run_recurrent_integration_comparison(base_model)
    assert rep.metrics_static.mean_g4_acc is not None
    assert rep.metrics_recurrent.mean_g4_acc is not None


def test_step341_causal_neural_path_interventions(base_model):
    """Verifies all 10 causal intervention conditions execute."""
    env = CompositionalAssociativeEnvironment(seed=42)
    core = CompactRecurrentAttentionCore(base_model, target_layer=3)
    rep = run_causal_neural_path_interventions(core, env, num_episodes=2)
    assert len(rep.results) == 10
    assert "normal" in rep.results
    assert "zero_recurrent_state" in rep.results
    assert "swap_recurrent_state" in rep.results


def test_step342_integrated_anti_shortcut_suite(base_model):
    """Verifies distractor sweeps and anti-shortcut conditions execute."""
    core = CompactRecurrentAttentionCore(base_model, target_layer=3)
    rep = run_integrated_anti_shortcut_suite(core, seed=42, num_episodes_per_test=2)
    assert len(rep.distractor_results) == 5
    assert rep.zero_contamination_verified is True


def test_step343_344_strict_evaluation_and_baseline_invariance(base_model):
    """Verifies multi-seed evaluation executes and canonical baseline weights remain bit-exact."""
    init_hash = compute_model_hash(base_model)
    assert init_hash == EXPECTED_WEIGHT_HASH

    rep = run_strict_integrated_i4_evaluation(base_model, seeds=[42])
    assert rep.baseline_exact is True
    assert rep.delta_w_zero is True
    assert rep.final_classification in ("I4_ACHIEVED", "I4_EMERGING", "I4_NOT_ACHIEVED")
