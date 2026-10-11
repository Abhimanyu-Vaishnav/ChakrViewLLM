"""Step 261: Multi-Seed Learnability Experiment.

Runs representation adaptation training and evaluation across 3 deterministic seeds:
- Seed 42
- Seed 101
- Seed 2026

Reports per seed:
- train accuracy / loss
- validation accuracy / loss
- unseen/unseen key accuracy
- unseen/unseen value-position accuracy
- unseen/unseen final token accuracy

Compares against previous Wave 249-256 baseline (unseen/unseen = 6.67%).
Computes mean and variance across seeds without cherry-picking.
"""

from __future__ import annotations

import dataclasses
import time
from typing import Dict, List, Optional, Tuple, Any

import torch

from chakrview.brain.model import ChakrMicro
from chakrview.runtime.interactive import compute_model_hash, EXPECTED_WEIGHT_HASH
from chakrview.cognition.neural_representation_adapter import (
    ChakrMicroWithAdaptedRepresentations,
)
from chakrview.cognition.adapter_curriculum_training import (
    run_adapter_training_curriculum,
)
from chakrview.cognition.representation_generalization_test import (
    evaluate_representation_generalization,
)


@dataclasses.dataclass
class SeedLearnabilityMetrics:
    seed: int
    train_loss: float
    val_loss: float
    unseen_unseen_key_acc: float
    unseen_unseen_val_acc: float
    unseen_unseen_tok_acc: float


@dataclasses.dataclass
class MultiSeedLearnabilityReport:
    seeds_tested: List[int]
    per_seed: Dict[int, SeedLearnabilityMetrics]
    mean_train_loss: float
    mean_val_loss: float
    mean_unseen_unseen_key_acc: float
    mean_unseen_unseen_val_acc: float
    mean_unseen_unseen_tok_acc: float
    baseline_unseen_unseen_tok_acc: float # 0.0667 (Wave 249-256)
    delta_vs_wave249_token_acc: float
    is_base_immutable: bool
    base_hash: str
    cpu_runtime_ms: float = 0.0


def run_multiseed_learnability(
    base_model: ChakrMicro,
    seeds: Optional[List[int]] = None,
    adapter_type: str = "gated",
    rank: int = 32,
    steps_per_phase: int = 15,
    eval_episodes: int = 10,
) -> MultiSeedLearnabilityReport:
    """Executes multi-seed learnability evaluation across seeds 42, 101, 2026."""
    if seeds is None:
        seeds = [42, 101, 2026]

    t0 = time.time()
    init_hash = compute_model_hash(base_model)
    per_seed: Dict[int, SeedLearnabilityMetrics] = {}

    for s in seeds:
        cand, cur_rep = run_adapter_training_curriculum(
            base_model=base_model,
            adapter_type=adapter_type,
            rank=rank,
            seed=s,
            steps_per_phase=steps_per_phase,
            eval_episodes_per_phase=eval_episodes,
        )

        gen_rep = evaluate_representation_generalization(
            candidate=cand,
            seed=s,
            num_episodes_per_split=eval_episodes,
        )

        uu = gen_rep.splits["unseen_unseen"]
        per_seed[s] = SeedLearnabilityMetrics(
            seed=s,
            train_loss=cur_rep.overall_train_loss,
            val_loss=cur_rep.overall_val_loss,
            unseen_unseen_key_acc=uu.key_selection_accuracy,
            unseen_unseen_val_acc=uu.val_position_accuracy,
            unseen_unseen_tok_acc=uu.final_token_accuracy,
        )

    mean_tr_loss = sum(m.train_loss for m in per_seed.values()) / len(per_seed)
    mean_vl_loss = sum(m.val_loss for m in per_seed.values()) / len(per_seed)
    mean_uu_k = sum(m.unseen_unseen_key_acc for m in per_seed.values()) / len(per_seed)
    mean_uu_v = sum(m.unseen_unseen_val_acc for m in per_seed.values()) / len(per_seed)
    mean_uu_t = sum(m.unseen_unseen_tok_acc for m in per_seed.values()) / len(per_seed)

    prev_wave_base = 0.0667
    delta = mean_uu_t - prev_wave_base

    post_hash = compute_model_hash(base_model)
    is_clean = (post_hash == init_hash == EXPECTED_WEIGHT_HASH)
    elapsed_ms = (time.time() - t0) * 1000.0

    return MultiSeedLearnabilityReport(
        seeds_tested=seeds,
        per_seed=per_seed,
        mean_train_loss=mean_tr_loss,
        mean_val_loss=mean_vl_loss,
        mean_unseen_unseen_key_acc=mean_uu_k,
        mean_unseen_unseen_val_acc=mean_uu_v,
        mean_unseen_unseen_tok_acc=mean_uu_t,
        baseline_unseen_unseen_tok_acc=prev_wave_base,
        delta_vs_wave249_token_acc=delta,
        is_base_immutable=is_clean,
        base_hash=post_hash,
        cpu_runtime_ms=elapsed_ms,
    )
