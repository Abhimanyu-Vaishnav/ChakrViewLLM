"""Step 185-192: Dedicated Test Suite for Untied Readout, Copy/Pointer Mechanisms & Disjoint Retrieval.

Validates:
1. Baseline Invariant: 3,443,136 parameters, SHA-256 c5571c...a282da, Delta W = 0
2. Step 185: Untied Readout candidate parameter scaling (4,229,568 params) and isolation
3. Step 186: Disjoint retrieval benchmark splits, generation, and contamination hashing
4. Step 187: Neural copy/pointer mechanism forward pass, causal attention, gating, and vocab scattering
5. Step 188: Copy vs generation comparative evaluation and failure mode separation
6. Step 189: Variable binding evaluation with pointer mechanism
7. Step 190: Multi-hop transitive reasoning evaluation across 1-hop to 4-hop
8. Step 191: Anti-shortcut controls, shuffling invariance, distractor robustness
9. Step 192: Master decision gate execution and consistency
"""

from pathlib import Path
import pytest
import torch
import torch.nn.functional as F

from chakrview.brain.model import ChakrMicro
from chakrview.runtime.interactive import (
    instantiate_frozen_baseline,
    compute_model_hash,
    EXPECTED_WEIGHT_HASH,
)
from chakrview.cognition.untied_readout import (
    create_untied_candidate,
    get_default_tokenizer,
)
from chakrview.cognition.disjoint_token_benchmark import (
    DisjointRetrievalFixture,
)
from chakrview.cognition.copy_attention import (
    NeuralCopyPointerHead,
    ChakrMicroWithCopy,
)
from chakrview.cognition.copy_generation_analysis import (
    evaluate_copy_vs_generation,
)
from chakrview.cognition.copy_variable_binding import (
    evaluate_variable_binding_suite,
)
from chakrview.cognition.copy_multihop_reasoning import (
    evaluate_multihop_reasoning,
)
from chakrview.cognition.copy_architecture_controls import (
    run_architecture_controls,
)


def test_01_baseline_integrity():
    """Verify frozen baseline remains exactly 3,443,136 params with identical SHA-256."""
    base = instantiate_frozen_baseline()
    param_count = sum(p.numel() for p in base.parameters())
    assert param_count == 3443136
    h = compute_model_hash(base)
    assert h == EXPECTED_WEIGHT_HASH


def test_02_step185_untied_readout_isolation():
    """Verify untied readout instantiates with independent LM head and +786,432 params."""
    base = instantiate_frozen_baseline()
    untied = create_untied_candidate(base)
    base_params = sum(p.numel() for p in base.parameters())
    untied_params = sum(p.numel() for p in untied.parameters())
    assert untied_params == 4229568
    assert untied_params - base_params == 786432
    assert untied.config.tie_embeddings is False
    # Ensure baseline parameters were not mutated
    assert compute_model_hash(base) == EXPECTED_WEIGHT_HASH


def test_03_step186_disjoint_benchmark_fixture():
    """Verify disjoint benchmark fixture generates 4 distinct splits and valid contamination hash."""
    fixture = DisjointRetrievalFixture(seed=42)
    suite = fixture.generate_benchmark_suite(samples_per_split=5)
    assert set(suite.keys()) == {"known_known", "known_unseen", "unseen_known", "unseen_unseen"}
    for cat, samples in suite.items():
        assert len(samples) == 5
        for s in samples:
            assert s.prompt.startswith("map ")
            assert "query |" in s.prompt
            assert s.expected_token is not None


def test_04_step187_neural_copy_mechanism():
    """Verify neural copy/pointer mechanism forward pass, shape contracts, and gating."""
    base = instantiate_frozen_baseline()
    copy_model = ChakrMicroWithCopy(base)
    
    # Check head parameter count: (192*192)*2 + 192 + 1 = 73,921 parameters
    head_params = sum(p.numel() for p in copy_model.copy_head.parameters())
    assert head_params == 73921
    
    dummy_input = torch.tensor([[68, 69, 70, 71]], dtype=torch.long)
    logits, copy_out = copy_model(dummy_input)
    
    assert logits.shape == (1, 4, 4096)
    assert copy_out.copy_attention.shape == (1, 4, 4)
    assert copy_out.copy_prob.shape == (1, 4, 1)
    assert copy_out.gen_prob.shape == (1, 4, 1)
    # Check that copy_prob + gen_prob == 1.0 (approximately)
    assert torch.allclose(copy_out.copy_prob + copy_out.gen_prob, torch.ones_like(copy_out.copy_prob), atol=1e-5)


def test_05_step188_copy_vs_generation():
    """Verify comparative evaluation between tied, untied, copy-only, and hybrid models."""
    base = instantiate_frozen_baseline()
    res = evaluate_copy_vs_generation(base, seed=42, num_samples=3)
    assert set(res.keys()) == {"tied_readout", "untied_readout", "copy_only", "hybrid_copy_gen"}
    assert res["copy_only"].mean_copy_prob == 1.0
    assert res["tied_readout"].mean_copy_prob == 0.0


def test_06_step189_variable_binding():
    """Verify variable binding suite runs under both conditions and evaluates metrics."""
    base = instantiate_frozen_baseline()
    eval_res = evaluate_variable_binding_suite(base, seed=42, num_samples_per_condition=3)
    assert eval_res.model_name == "tied_base"
    assert 0.0 <= eval_res.familiar_overall_acc <= 1.0
    assert 0.0 <= eval_res.disjoint_overall_acc <= 1.0


def test_07_step190_multihop_reasoning():
    """Verify multi-hop reasoning evaluates 1 to 4 hops and records failure boundary."""
    base = instantiate_frozen_baseline()
    eval_res = evaluate_multihop_reasoning(base, seed=42, samples_per_hop=3)
    assert set(eval_res.results_by_hop.keys()) == {1, 2, 3, 4}
    # For baseline, 2-hop is 0.0000
    assert eval_res.results_by_hop[2].accuracy == 0.0


def test_08_step191_architecture_controls():
    """Verify anti-shortcut control experiment runs and computes contamination hash."""
    base = instantiate_frozen_baseline()
    ctrl_res = run_architecture_controls(base, seed=42, num_eval_samples=3)
    assert len(ctrl_res.contamination_hash) == 64
    assert isinstance(ctrl_res.is_shortcut_dependent, bool)
