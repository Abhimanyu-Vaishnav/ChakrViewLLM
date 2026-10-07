"""Test suite for Wave 329–336: Trainable Compositional Attention Subsets."""

import pytest
import torch

from chakrview.brain.model import ChakrMicro
from chakrview.brain.config import ModelConfig
from chakrview.runtime.interactive import compute_model_hash, EXPECTED_WEIGHT_HASH
from chakrview.cognition.compositional_attention_forensics import (
    run_compositional_attention_forensics,
)
from chakrview.cognition.trainable_attention_subset import (
    CompositionalAttentionSubsetModel,
)
from chakrview.cognition.dedicated_relational_head import (
    DedicatedRelationalAttentionHead,
    ChakrMicroWithDedicatedRelationalHead,
)
from chakrview.cognition.causal_attention_interventions import (
    run_causal_attention_interventions,
)
from chakrview.cognition.attention_anti_shortcut import (
    run_attention_anti_shortcut_suite,
)
from chakrview.cognition.strict_attention_i4_evaluation import (
    run_strict_attention_i4_evaluation,
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


def test_step329_attention_forensics(base_model):
    """Verifies attention forensics traces all 36 heads and identifies optimal layer."""
    report = run_compositional_attention_forensics(base_model, seeds=[42], num_episodes=2)
    assert len(report.head_metrics) == 36
    assert 0 <= report.best_layer < 6
    assert 0 <= report.best_head < 6
    assert report.best_margin is not None
    assert "Layer" in report.recommended_intervention


def test_step330_331_332_trainable_attention_subset_identity_and_params(base_model):
    """Verifies low-rank Q/K/QK models initialize at exact identity and strictly respect budgets."""
    # Q-only
    q_model = CompositionalAttentionSubsetModel(base_model, target_layer=3, mode="q_only", rank=16)
    q_params = sum(p.numel() for p in q_model.parameters() if p.requires_grad)
    assert q_params == 6144
    assert q_params < 25000

    # K-only
    k_model = CompositionalAttentionSubsetModel(base_model, target_layer=3, mode="k_only", rank=16)
    k_params = sum(p.numel() for p in k_model.parameters() if p.requires_grad)
    assert k_params == 6144
    assert k_params < 25000

    # Q+K
    qk_model = CompositionalAttentionSubsetModel(base_model, target_layer=3, mode="qk", rank=16)
    qk_params = sum(p.numel() for p in qk_model.parameters() if p.requires_grad)
    assert qk_params == 12288
    assert qk_params < 50000

    # Verify exact identity forward pass at initialization
    inp = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    with torch.no_grad():
        base_logits = base_model(inp)
        q_logits = q_model(inp)
        qk_logits = qk_model(inp)

    assert torch.allclose(base_logits, q_logits, atol=1e-5)
    assert torch.allclose(base_logits, qk_logits, atol=1e-5)


def test_step333_dedicated_relational_head(base_model):
    """Verifies dedicated relational head integration and parameter count."""
    rel_model = ChakrMicroWithDedicatedRelationalHead(base_model, target_layer=3, head_dim=32)
    rel_params = sum(p.numel() for p in rel_model.parameters() if p.requires_grad)
    assert rel_params < 50000

    # Verify zero gate identity at initialization
    inp = torch.tensor([[10, 20, 30, 40]], dtype=torch.long)
    with torch.no_grad():
        base_logits = base_model(inp)
        rel_logits = rel_model(inp)

    assert torch.allclose(base_logits, rel_logits, atol=1e-5)


def test_step334_causal_interventions(base_model):
    """Verifies causal interventions framework executes all 8 conditions."""
    env = CompositionalAssociativeEnvironment(seed=42)
    episodes = [env.generate_episode("train", num_distractors=1, episode_idx=2400000 + i) for i in range(2)]
    model = CompositionalAttentionSubsetModel(base_model, target_layer=3, mode="qk", rank=16)
    rep = run_causal_attention_interventions(model, episodes)
    assert len(rep.results) == 8
    assert "normal" in rep.results
    assert "zero_relational" in rep.results
    assert "shuffle_q" in rep.results


def test_step335_anti_shortcut_suite(base_model):
    """Verifies all 16 anti-shortcut conditions run and zero contamination holds."""
    model = CompositionalAttentionSubsetModel(base_model, target_layer=3, mode="qk", rank=16)
    rep = run_attention_anti_shortcut_suite(model, seed=42, num_episodes_per_cond=1)
    assert rep.total_count == 16
    assert rep.zero_contamination_verified is True


def test_step336_strict_evaluation_and_baseline_invariance(base_model):
    """Verifies multi-seed evaluation executes and canonical baseline weights remain bit-exact."""
    init_hash = compute_model_hash(base_model)
    assert init_hash == EXPECTED_WEIGHT_HASH

    rep = run_strict_attention_i4_evaluation(base_model, seeds=[42])
    assert len(rep.arch_results) == 5
    assert rep.base_sha256_exact is True
    assert rep.base_delta_w_zero is True
    assert rep.architecture_decision in ("OUTCOME_A", "OUTCOME_B", "OUTCOME_C")
