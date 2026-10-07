"""Pytest test suite for Wave 345–352: Complete Transformer Block Training."""

import pytest
import torch

from chakrview.brain.model import ChakrMicro
from chakrview.brain.config import ModelConfig
from chakrview.runtime.interactive import compute_model_hash, EXPECTED_WEIGHT_HASH
from chakrview.cognition.complete_block_training_forensics import (
    run_complete_block_forensics,
)
from chakrview.cognition.trainable_transformer_block import (
    TrainableTransformerBlockCandidate,
)
from chakrview.cognition.training_objective_study import (
    run_training_objective_study,
)
from chakrview.cognition.layer_location_study import (
    run_layer_location_study,
)
from chakrview.cognition.two_block_training_study import (
    run_two_block_controlled_training,
)
from chakrview.cognition.causal_block_interventions import (
    run_core_causal_interventions,
)
from chakrview.cognition.block_anti_shortcut_suite import (
    run_block_anti_shortcut_suite,
)
from chakrview.cognition.strict_block_i4_evaluation import (
    run_strict_block_i4_evaluation,
)
from chakrview.cognition.compositional_benchmark_environment import (
    CompositionalAssociativeEnvironment,
)


@pytest.fixture
def base_model():
    torch.manual_seed(42)
    cfg = ModelConfig()
    model = ChakrMicro(cfg)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    return model


def test_step345_complete_block_forensics(base_model):
    """Verifies complete block forensics executes and parameter audit matches 3,443,136."""
    rep = run_complete_block_forensics(base_model)
    assert rep.baseline_exact is True
    assert rep.total_model_params == 3443136
    assert rep.total_block_params == 442752
    assert len(rep.layer_parameters) == 6
    assert rep.recommended_primary_block == 3


def test_step346_trainable_transformer_block_isolation_and_params(base_model):
    """Verifies TrainableTransformerBlockCandidate leaves baseline frozen and clones correctly."""
    base_hash_before = compute_model_hash(base_model)
    assert base_hash_before == EXPECTED_WEIGHT_HASH

    cand = TrainableTransformerBlockCandidate(base_model, trainable_layers=[3], enable_contextual_binding=True)
    assert cand.trainable_param_count == 442752 + sum(p.numel() for p in cand.binding.parameters())

    # Ensure baseline itself has 0 trainable parameters
    assert sum(p.numel() for p in base_model.parameters() if p.requires_grad) == 0

    inp = torch.tensor([[10, 20, 30, 40]], dtype=torch.long)
    with torch.no_grad():
        b_logits = base_model(inp)
        c_logits = cand(inp)
    assert torch.allclose(b_logits, c_logits, atol=1e-5)


def test_step347_training_objective_study(base_model):
    """Verifies training objective study executes across all 4 objectives."""
    rep = run_training_objective_study(base_model)
    assert len(rep.results) == 4
    assert "Obj_A_Lang_Final" in rep.results
    assert "Obj_C_Hop1_Hop2_Final" in rep.results


def test_step348_layer_location_study(base_model):
    """Verifies layer location comparison across Layer 2, Layer 3, and Layer 4."""
    rep = run_layer_location_study(base_model, candidate_layers=[2, 3, 4], seed=42)
    assert len(rep.layer_results) == 3
    assert rep.best_layer in (2, 3, 4)


def test_step349_two_block_controlled_training(base_model):
    """Verifies single block vs two-block study runs."""
    rep = run_two_block_controlled_training(base_model)
    assert rep.single_block_metrics.g4_acc is not None
    assert rep.two_block_metrics.g4_acc is not None


def test_step350_causal_block_interventions(base_model):
    """Verifies all 8 causal intervention conditions execute on the trained block."""
    cand = TrainableTransformerBlockCandidate(base_model, trainable_layers=[3], enable_contextual_binding=True)
    env = CompositionalAssociativeEnvironment(seed=42)
    rep = run_core_causal_interventions(cand, base_model, env, num_episodes=2, target_layer=3)
    assert len(rep.results) == 8
    assert "normal" in rep.results
    assert "replace_with_baseline" in rep.results
    assert "zero_attn_contribution" in rep.results


def test_step351_block_anti_shortcut_suite(base_model):
    """Verifies distractor sweeps and anti-shortcut conditions execute."""
    cand = TrainableTransformerBlockCandidate(base_model, trainable_layers=[3], enable_contextual_binding=True)
    rep = run_block_anti_shortcut_suite(cand, seed=42, num_episodes_per_cond=2)
    assert len(rep.distractor_results) == 5
    assert rep.zero_contamination_verified is True


def test_step352_strict_block_i4_evaluation(base_model):
    """Verifies multi-seed evaluation executes and canonical baseline weights remain bit-exact."""
    init_hash = compute_model_hash(base_model)
    assert init_hash == EXPECTED_WEIGHT_HASH

    rep = run_strict_block_i4_evaluation(base_model, target_layer=3, seeds=[42])
    assert rep.baseline_exact is True
    assert rep.delta_w_zero is True
    assert rep.final_classification in ("I4_ACHIEVED", "I4_EMERGING", "I4_NOT_ACHIEVED")
