"""Unit and Integration Tests for Wave 305-312: Trainable Compositional Backbone Pathway.

Tests:
1. Canonical baseline immutability invariant (3,443,136 params, hash c5571c..., Delta W = 0)
2. Step 305: Internal composition forensics layer-wise diagnostic
3. Step 306: Minimal internal trainable pathway forward pass and zero identity initialization
4. Step 307: Internal location comparison configurations A, B, C, D
5. Step 308: Internal pathway + dynamic token binding integration modes A, B, C, D
6. Step 309: Controlled multi-stage training objective & ablation
7. Step 310: Anti-shortcut & generalization testing across 14 stress conditions
8. Step 311/312: Strict multi-seed I3 preservation and I4 evaluation master execution
"""

from __future__ import annotations

import pytest
import torch

from chakrview.brain.model import ChakrMicro
from chakrview.runtime.interactive import (
    instantiate_frozen_baseline,
    compute_model_hash,
    EXPECTED_WEIGHT_HASH,
)
from chakrview.cognition.internal_trainable_pathway import (
    ChakrMicroWithInternalPathway,
    compute_module_sha256,
)
from chakrview.cognition.internal_composition_forensics import (
    run_internal_composition_forensics,
)
from chakrview.cognition.internal_location_comparison import (
    run_internal_location_comparison,
)
from chakrview.cognition.internal_integration_study import (
    IntegratedInternalBindingModel,
    run_internal_integration_study,
)
from chakrview.cognition.pathway_training_objective import (
    train_pathway_model,
    run_compositional_training_objective_study,
)
from chakrview.cognition.pathway_anti_shortcut import (
    run_pathway_anti_shortcut_suite,
)
from chakrview.cognition.strict_pathway_i4_evaluation import (
    run_strict_pathway_i3_i4_evaluation,
)


def test_baseline_immutability_wave312():
    """Verify canonical baseline remains strictly frozen and bit-exact."""
    base = instantiate_frozen_baseline()
    params = sum(p.numel() for p in base.parameters())
    assert params == 3_443_136
    h = compute_model_hash(base)
    assert h == EXPECTED_WEIGHT_HASH


def test_step305_internal_forensics():
    """Verify Step 305 internal layer-wise forensics."""
    base = instantiate_frozen_baseline()
    rep = run_internal_composition_forensics(base, seeds=[42], num_episodes=2)
    assert rep.mean_critical_layer in range(1, 8)
    assert 42 in rep.per_seed_results


def test_step306_internal_pathway_architecture():
    """Verify Step 306 internal pathway forward pass and zero-op initialization."""
    base = instantiate_frozen_baseline()
    cand = ChakrMicroWithInternalPathway(base, target_layers=[3, 4, 5], bottleneck_dim=32)
    trainable_p = sum(p.numel() for p in cand.parameters() if p.requires_grad)
    assert trainable_p < 50_000

    inp = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    diff = float(torch.norm(cand(inp) - base(inp)).item())
    assert diff < 1e-6


def test_step307_internal_location_comparison():
    """Verify Step 307 internal location configurations."""
    base = instantiate_frozen_baseline()
    rep = run_internal_location_comparison(base, seeds=[42], train_steps=2, eval_episodes=2)
    assert len(rep.location_results) == 4
    assert rep.best_location in rep.location_results


def test_step308_internal_integration_modes():
    """Verify Step 308 integration modes."""
    base = instantiate_frozen_baseline()
    for m in ["B", "C", "D"]:
        model = IntegratedInternalBindingModel(base, config_mode=m)
        inp = torch.tensor([[10, 20, 30]], dtype=torch.long)
        h, _ = model.internal_backbone.forward_hidden_states(inp)
        assert h.shape == (1, 3, 192)


def test_step309_training_objective_and_ablation():
    """Verify Step 309 training objective and ablation."""
    base = instantiate_frozen_baseline()
    _, m_full = train_pathway_model(base, seed=42, train_steps=2, eval_episodes=2, use_hop2_objective=True)
    _, m_abl = train_pathway_model(base, seed=42, train_steps=2, eval_episodes=2, use_hop2_objective=False)
    assert m_full.final_loss >= 0.0
    assert m_abl.final_loss >= 0.0


def test_step309_objective_study_run():
    """Verify Step 309 objective study execution."""
    base = instantiate_frozen_baseline()
    rep = run_compositional_training_objective_study(base, seeds=[42], train_steps=2, eval_episodes=2)
    assert rep.mean_g4_with_hop2_loss >= 0.0


def test_step310_pathway_anti_shortcut():
    """Verify Step 310 14-condition anti-shortcut audit."""
    base = instantiate_frozen_baseline()
    model, _ = train_pathway_model(base, seed=42, train_steps=2, eval_episodes=2)
    rep = run_pathway_anti_shortcut_suite(model, seed=42, episodes_per_cond=2)
    assert len(rep.conditions) == 14
    for c in rep.conditions:
        assert 0.0 <= c.final_tok_acc <= 1.0


def test_step311_312_strict_evaluation():
    """Verify Step 311 and 312 strict multi-seed evaluation and classification."""
    base = instantiate_frozen_baseline()
    rep = run_strict_pathway_i3_i4_evaluation(base, seeds=[42, 101], train_steps=2, eval_episodes=2)
    assert rep.final_classification in ("I4_ACHIEVED", "I4_EMERGING", "I4_NOT_ACHIEVED")
    assert rep.is_base_bit_exact is True
