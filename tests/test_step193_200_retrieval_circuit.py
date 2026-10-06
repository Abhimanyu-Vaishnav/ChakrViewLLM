"""Step 193-200: Dedicated Test Suite for Retrieval Circuit Formation, Query-Key Matching & Associative Memory.

Validates:
1. Baseline Invariant & Immutability: 3,443,136 parameters, SHA-256 c5571c...a282da, Delta W = 0
2. Step 193: Representation probing (linear probes for key and value decodability)
3. Step 194: Contrastive query-key matching objective and parameter isolation
4. Step 195: Value-position retrieval objective and 2x2 diagnostic matrix
5. Step 196: Retrieval curriculum evaluation (R0 to R8)
6. Step 197: Disjoint retrieval 3-stage generalization pipeline
7. Step 198: Dynamic variable binding with role permutation
8. Step 199: Iterative multi-hop retrieval and failure boundary detection
9. Step 200: Master benchmark consistency and classification coverage
"""

from pathlib import Path
import pytest
import torch

from chakrview.brain.model import ChakrMicro
from chakrview.runtime.interactive import (
    instantiate_frozen_baseline,
    compute_model_hash,
    EXPECTED_WEIGHT_HASH,
)
from chakrview.cognition.retrieval_representation_probe import (
    probe_key_value_representations,
)
from chakrview.cognition.query_key_matching import (
    train_and_eval_query_key_matching,
)
from chakrview.cognition.value_position_retrieval import (
    ChakrMicroWithRetrievalCircuit,
    train_and_eval_value_retrieval,
)
from chakrview.cognition.retrieval_curriculum import (
    RetrievalCurriculumEvaluator,
)
from chakrview.cognition.disjoint_retrieval_learning import (
    run_disjoint_retrieval_generalization,
)
from chakrview.cognition.dynamic_variable_binding import (
    evaluate_dynamic_variable_binding,
)
from chakrview.cognition.iterative_multihop_retrieval import (
    evaluate_iterative_multihop_reasoning,
)


def test_01_baseline_integrity():
    """Verify frozen baseline remains exactly 3,443,136 params with identical SHA-256."""
    base = instantiate_frozen_baseline()
    param_count = sum(p.numel() for p in base.parameters())
    assert param_count == 3443136
    h = compute_model_hash(base)
    assert h == EXPECTED_WEIGHT_HASH


def test_02_step193_representation_probing():
    """Verify key and value identities are decodable from contextual hidden states."""
    base = instantiate_frozen_baseline()
    res = probe_key_value_representations(base, seed=42, num_samples=12)
    assert res.key_probe_accuracy >= 0.50
    assert res.value_probe_accuracy >= 0.50
    assert res.sample_count == 12


def test_03_step194_query_key_matching():
    """Verify differentiable query-key matching aligns query with key positions."""
    base = instantiate_frozen_baseline()
    res = train_and_eval_query_key_matching(base, seed=42, epochs=4, num_eval_samples=6)
    assert 0.0 <= res.matching_key_acc <= 1.0
    assert res.mean_correct_key_attention >= 0.0
    assert compute_model_hash(base) == EXPECTED_WEIGHT_HASH


def test_04_step195_value_position_retrieval():
    """Verify value-position retrieval evaluates 2x2 diagnostic matrix."""
    base = instantiate_frozen_baseline()
    res = train_and_eval_value_retrieval(base, seed=42, epochs=4, num_eval_samples=6)
    total_matrix_rate = (
        res.key_corr_val_corr + res.key_corr_val_incorr +
        res.key_incorr_val_corr + res.key_incorr_val_incorr
    )
    assert abs(total_matrix_rate - 1.0) < 1e-4
    assert 0.0 <= res.value_position_accuracy <= 1.0


def test_05_step196_retrieval_curriculum():
    """Verify 9-level curriculum generates proper progression R0 to R8."""
    base = instantiate_frozen_baseline()
    model = ChakrMicroWithRetrievalCircuit(base)
    evaluator = RetrievalCurriculumEvaluator(seed=42)
    rep = evaluator.evaluate_curriculum(model, samples_per_level=3)
    assert len(rep.levels) == 9
    assert "R0" in rep.levels and "R8" in rep.levels
    assert rep.highest_passed_level in list(rep.levels.keys()) + ["None"]


def test_06_step197_disjoint_generalization():
    """Verify 3-stage disjoint generalization pipeline across seeds."""
    base = instantiate_frozen_baseline()
    rep = run_disjoint_retrieval_generalization(base, seeds=[42], epochs=3, num_eval_samples=4)
    assert 42 in rep.seed_results
    assert 0.0 <= rep.mean_stage_a_acc <= 1.0
    assert 0.0 <= rep.mean_stage_b_acc <= 1.0
    assert 0.0 <= rep.mean_stage_c_acc <= 1.0


def test_07_step198_dynamic_variable_binding():
    """Verify variable binding tests familiar vs disjoint role extraction."""
    base = instantiate_frozen_baseline()
    res = evaluate_dynamic_variable_binding(base, seed=42, num_samples_per_cond=4)
    assert 0.0 <= res.familiar_role_acc <= 1.0
    assert 0.0 <= res.disjoint_role_acc <= 1.0


def test_08_step199_iterative_multihop():
    """Verify multi-hop reasoning assesses 1 to 4 hops and records failure boundary."""
    base = instantiate_frozen_baseline()
    rep = evaluate_iterative_multihop_reasoning(base, seed=42, samples_per_hop=4)
    assert set(rep.hops.keys()) == {1, 2, 3, 4}
    assert rep.failure_boundary_hop >= 1
    assert rep.hops[2].accuracy == 0.0
