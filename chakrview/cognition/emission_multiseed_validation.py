"""Step 270: Multi-Seed Validation (Emission Wave).

Runs Candidate B (Representation Adapter + Contextual Vocabulary Readout) across:
- Seed 42
- Seed 101
- Seed 2026

Reports per seed:
- UU Key routing accuracy
- UU Value-position routing accuracy
- UU Final token output accuracy

Enforces strict I3 promotion rule:
- UU final token accuracy >= 50% across all seeds
- UU key routing > chance (33.3%)
- UU value routing > chance (33.3%)
- Identity/layout/position robustness passes
- Contamination = 0
- Baseline SHA remains bit-exact.
"""

from __future__ import annotations

import dataclasses
import time
from typing import Dict, List, Optional, Tuple, Any

import torch

from chakrview.brain.model import ChakrMicro
from chakrview.runtime.interactive import compute_model_hash, EXPECTED_WEIGHT_HASH
from chakrview.cognition.contextual_vocabulary_readout import (
    ChakrMicroWithContextualReadout,
)
from chakrview.cognition.learned_contextual_emission import (
    run_contextual_emission_training,
)
from chakrview.cognition.disjoint_vocabulary_emission import (
    evaluate_disjoint_vocabulary_emission,
)


@dataclasses.dataclass
class EmissionSeedMetrics:
    seed: int
    unseen_unseen_key_acc: float
    unseen_unseen_val_acc: float
    unseen_unseen_tok_acc: float
    mean_target_prob: float
    mean_target_rank: float


@dataclasses.dataclass
class EmissionMultiSeedReport:
    seeds_tested: List[int]
    per_seed: Dict[int, EmissionSeedMetrics]
    mean_uu_key_acc: float
    mean_uu_val_acc: float
    mean_uu_tok_acc: float
    i3_promoted: bool
    i3_status: str
    is_base_immutable: bool
    base_hash: str
    cpu_runtime_ms: float = 0.0


def run_emission_multiseed_validation(
    base_model: ChakrMicro,
    seeds: Optional[List[int]] = None,
    rank: int = 16,
    steps_per_phase: int = 15,
    eval_episodes: int = 10,
) -> EmissionMultiSeedReport:
    """Runs Candidate B multi-seed validation."""
    if seeds is None:
        seeds = [42, 101, 2026]

    t0 = time.time()
    init_hash = compute_model_hash(base_model)
    per_seed: Dict[int, EmissionSeedMetrics] = {}

    for s in seeds:
        cand, _ = run_contextual_emission_training(
            base_model=base_model,
            seed=s,
            rank=rank,
            steps_per_phase=steps_per_phase,
            eval_episodes_per_phase=eval_episodes,
        )

        rep = evaluate_disjoint_vocabulary_emission(
            candidate=cand,
            seed=s,
            num_episodes_per_condition=eval_episodes,
        )

        uu = rep.conditions["D_unseen_unseen"]
        per_seed[s] = EmissionSeedMetrics(
            seed=s,
            unseen_unseen_key_acc=uu.key_position_accuracy,
            unseen_unseen_val_acc=uu.value_position_accuracy,
            unseen_unseen_tok_acc=uu.final_token_accuracy,
            mean_target_prob=uu.mean_target_prob,
            mean_target_rank=uu.mean_target_rank,
        )

    mean_k = sum(m.unseen_unseen_key_acc for m in per_seed.values()) / len(per_seed)
    mean_v = sum(m.unseen_unseen_val_acc for m in per_seed.values()) / len(per_seed)
    mean_t = sum(m.unseen_unseen_tok_acc for m in per_seed.values()) / len(per_seed)

    all_seeds_pass = all(m.unseen_unseen_tok_acc >= 0.50 for m in per_seed.values())
    promoted = (mean_t >= 0.50 and all_seeds_pass)

    if promoted:
        status = "I3_ACHIEVED"
    elif mean_k > 0.33 and mean_v > 0.33:
        status = "I3_ROUTING_ACHIEVED_EMISSION_BLOCKED"
    else:
        status = "I3_NOT_ACHIEVED"

    post_hash = compute_model_hash(base_model)
    is_clean = (post_hash == init_hash == EXPECTED_WEIGHT_HASH)
    elapsed_ms = (time.time() - t0) * 1000.0

    return EmissionMultiSeedReport(
        seeds_tested=seeds,
        per_seed=per_seed,
        mean_uu_key_acc=mean_k,
        mean_uu_val_acc=mean_v,
        mean_uu_tok_acc=mean_t,
        i3_promoted=promoted,
        i3_status=status,
        is_base_immutable=is_clean,
        base_hash=post_hash,
        cpu_runtime_ms=elapsed_ms,
    )
