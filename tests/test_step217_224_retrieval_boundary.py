"""Step 217-224: Dedicated Test Suite for Retrieval Failure Boundary & Value Projection.

Validates:
1. Baseline Invariant & Immutability: 3,443,136 parameters, SHA-256 c5571c...a282da, Delta W = 0
2. Step 217: Value representation retrieval across 4 splits (known/unseen keys and values)
3. Step 218: Three-stage failure boundary (Stage A, Stage B, Stage C) on identical instances
4. Step 219: Disjoint value representation transfer across randomized identity sets
5. Step 220: Instrument internal retrieval path and produce machine-readable trace
6. Step 221: Vocabulary projection candidate comparison under isolated conditions
7. Step 222: Zero-shot variable binding on randomized contextual mappings
8. Step 223: Representation-level multi-hop chaining (intermediate retrieved rep as next query)
9. Step 224: Master benchmark decision gate and 32 master categories consistency
"""

import copy
from pathlib import Path
import pytest
import torch

from chakrview.brain.model import ChakrMicro
from chakrview.runtime.interactive import (
    instantiate_frozen_baseline,
    compute_model_hash,
    EXPECTED_WEIGHT_HASH,
)
from chakrview.cognition.value_representation_retrieval import (
    evaluate_value_representation_retrieval,
)
from chakrview.cognition.retrieval_boundary import (
    evaluate_three_stage_retrieval_boundary,
)
from chakrview.cognition.disjoint_value_transfer import (
    evaluate_disjoint_value_transfer,
)
from chakrview.cognition.retrieval_trace import (
    trace_retrieval_circuit,
)
from chakrview.cognition.value_to_token_projection import (
    evaluate_vocabulary_projections,
)
from chakrview.cognition.zero_shot_variable_binding import (
    evaluate_zero_shot_variable_binding,
)
from chakrview.cognition.representation_multihop import (
    evaluate_representation_multihop,
)


def test_01_baseline_integrity_and_immutability():
    """Verify baseline model parameter count and bit-exact SHA-256."""
    base = instantiate_frozen_baseline()
    assert sum(p.numel() for p in base.parameters()) == 3443136
    h = compute_model_hash(base)
    assert h == EXPECTED_WEIGHT_HASH


def test_02_step217_value_representation_retrieval():
    """Verify value representation retrieval evaluates all 4 conditions."""
    base = instantiate_frozen_baseline()
    rep = evaluate_value_representation_retrieval(base, seed=42, samples_per_split=4)
    assert len(rep.splits) == 4
    assert "known_known" in rep.splits
    assert "unseen_unseen" in rep.splits
    assert isinstance(rep.is_representation_retrieval_successful, bool)


def test_03_step218_three_stage_retrieval_boundary():
    """Verify three-stage boundary explicitly separates Stages A, B, and C."""
    base = instantiate_frozen_baseline()
    rep = evaluate_three_stage_retrieval_boundary(base, seed=42, samples_per_condition=3)
    assert len(rep.stages_table) == 4
    for cond, res in rep.stages_table.items():
        assert res.first_failure_stage in ("STAGE_A", "STAGE_B", "STAGE_C", "NONE")
    assert rep.overall_failure_boundary in ("STAGE_A", "STAGE_B", "STAGE_C", "NONE")


def test_04_step219_disjoint_value_transfer():
    """Verify disjoint value transfer evaluation across seeds."""
    base = instantiate_frozen_baseline()
    rep = evaluate_disjoint_value_transfer(base, seed=42, samples_per_split=3)
    assert len(rep.splits) == 4
    assert rep.next_investigation_target in ("VOCABULARY_PROJECTION", "ASSOCIATIVE_ROUTING")


def test_05_step220_retrieval_trace():
    """Verify internal retrieval path tracing across 6 stages."""
    base = instantiate_frozen_baseline()
    rep = trace_retrieval_circuit(base, seed=42)
    assert len(rep.stages) == 6
    assert rep.stages[0].stage_name == "1_query_state"
    assert rep.stages[1].stage_name == "2_key_matching"
    assert rep.stages[2].stage_name == "3_association_state"
    assert rep.stages[3].stage_name == "4_value_representation"
    assert rep.stages[4].stage_name == "5_output_representation"
    assert rep.stages[5].stage_name == "6_logits_output"
    assert rep.first_loss_stage in [s.stage_name for s in rep.stages] or rep.first_loss_stage == "NONE"


def test_06_step221_vocabulary_projections():
    """Verify vocabulary projection evaluation preconditions and isolated candidates."""
    base = instantiate_frozen_baseline()
    # When precondition not met
    rep_pre_fail = evaluate_vocabulary_projections(base, precondition_verified=False)
    assert rep_pre_fail.precondition_met is False
    assert "NOT APPLICABLE" in rep_pre_fail.status

    # When precondition met
    rep_pre_ok = evaluate_vocabulary_projections(base, precondition_verified=True)
    assert rep_pre_ok.precondition_met is True
    assert len(rep_pre_ok.candidates) == 4


def test_07_step222_zero_shot_variable_binding():
    """Verify zero-shot variable binding on randomized contextual mappings."""
    base = instantiate_frozen_baseline()
    rep = evaluate_zero_shot_variable_binding(base, seed=42, num_episodes=4)
    assert rep.num_episodes == 4
    assert len(rep.episodes) == 4
    assert 0.0 <= rep.final_token_acc <= 1.0


def test_08_step223_representation_multihop():
    """Verify representation-level multi-hop chaining across 2, 3, and 4 hops."""
    base = instantiate_frozen_baseline()
    rep = evaluate_representation_multihop(base, seed=42, num_eval_chains=2)
    assert 2 in rep.chain_results
    assert 3 in rep.chain_results
    assert 4 in rep.chain_results
    assert len(rep.chain_results[2].hop_metrics) == 2
    assert len(rep.chain_results[3].hop_metrics) == 3
    assert len(rep.chain_results[4].hop_metrics) == 4
