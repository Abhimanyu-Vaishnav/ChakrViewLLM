"""Step 294: Strict I4 Re-Evaluation Suite.

Runs full 2-hop benchmark on the strongest stabilized candidate architecture across:
G1: Known identities / Known composition
G2: Unseen identities / Known composition structure
G3: Known identities / Novel unseen composition chains
G4: Unseen identities / Novel unseen composition chains (PRIMARY I4 CAPABILITY GATE)

Runs seeds:
- Seed 42
- Seed 101
- Seed 2026

I4 Promotion Gate:
- G4 unseen/unseen final token accuracy >= 50% mean across seeds
- Stability diagnostic: No individual seed below 40%
- Baseline bit-exact verification
- Zero Python answer lookup
"""

from __future__ import annotations

import dataclasses
import time
from typing import Dict, List, Optional, Tuple, Any

import torch

from chakrview.brain.model import ChakrMicro
from chakrview.runtime.interactive import compute_model_hash, EXPECTED_WEIGHT_HASH
from chakrview.cognition.two_hop_composition_architecture import (
    ChakrMicroCompositionalReasoningModel,
)
from chakrview.cognition.composition_generalization_training import (
    run_compositional_training_and_evaluation,
    CompositionGeneralizationReport,
)


@dataclasses.dataclass
class StrictI4SeedEvaluation:
    seed: int
    g1_tok_acc: float
    g2_tok_acc: float
    g3_tok_acc: float
    g4_tok_acc: float
    hop1_key_acc: float
    hop1_val_acc: float
    intermediate_preservation_acc: float
    hop2_key_acc: float
    hop2_val_acc: float
    mean_target_prob: float
    mean_target_rank: float


@dataclasses.dataclass
class StrictI4ReEvaluationReport:
    per_seed_results: Dict[int, StrictI4SeedEvaluation]
    mean_g1_tok_acc: float
    mean_g2_tok_acc: float
    mean_g3_tok_acc: float
    mean_g4_tok_acc: float
    mean_hop1_key_acc: float
    mean_hop1_val_acc: float
    mean_inter_preservation: float
    mean_hop2_key_acc: float
    mean_hop2_val_acc: float
    i4_gate_passed: bool
    stability_diagnostic_passed: bool
    is_base_bit_exact: bool
    final_classification: str
    summary: str


def run_strict_i4_reevaluation(
    base_model: ChakrMicro,
    seeds: Optional[List[int]] = None,
    train_steps: int = 20,
    eval_episodes_per_split: int = 10,
) -> StrictI4ReEvaluationReport:
    """Executes strict I4 re-evaluation across seeds 42, 101, 2026."""
    if seeds is None:
        seeds = [42, 101, 2026]

    init_hash = compute_model_hash(base_model)
    per_seed: Dict[int, StrictI4SeedEvaluation] = {}

    for s in seeds:
        cand, rep = run_compositional_training_and_evaluation(
            base_model=base_model,
            seed=s,
            rank=16,
            d_bind=64,
            train_steps=train_steps,
            eval_episodes_per_split=eval_episodes_per_split,
        )

        g1 = rep.splits["G1_known_known"]
        g2 = rep.splits["G2_unseen_identities"]
        g3 = rep.splits["G3_unseen_composition"]
        g4 = rep.splits["G4_unseen_unseen"]

        per_seed[s] = StrictI4SeedEvaluation(
            seed=s,
            g1_tok_acc=g1.final_token_acc,
            g2_tok_acc=g2.final_token_acc,
            g3_tok_acc=g3.final_token_acc,
            g4_tok_acc=g4.final_token_acc,
            hop1_key_acc=g4.hop1_key_acc,
            hop1_val_acc=g4.hop1_val_acc,
            intermediate_preservation_acc=g4.intermediate_state_acc,
            hop2_key_acc=g4.hop2_key_acc,
            hop2_val_acc=g4.hop2_val_acc,
            mean_target_prob=g4.mean_target_prob,
            mean_target_rank=g4.mean_target_rank,
        )

    m_g1 = sum(sr.g1_tok_acc for sr in per_seed.values()) / len(per_seed)
    m_g2 = sum(sr.g2_tok_acc for sr in per_seed.values()) / len(per_seed)
    m_g3 = sum(sr.g3_tok_acc for sr in per_seed.values()) / len(per_seed)
    m_g4 = sum(sr.g4_tok_acc for sr in per_seed.values()) / len(per_seed)

    m_h1_k = sum(sr.hop1_key_acc for sr in per_seed.values()) / len(per_seed)
    m_h1_v = sum(sr.hop1_val_acc for sr in per_seed.values()) / len(per_seed)
    m_int = sum(sr.intermediate_preservation_acc for sr in per_seed.values()) / len(per_seed)
    m_h2_k = sum(sr.hop2_key_acc for sr in per_seed.values()) / len(per_seed)
    m_h2_v = sum(sr.hop2_val_acc for sr in per_seed.values()) / len(per_seed)

    post_hash = compute_model_hash(base_model)
    is_bit_exact = (post_hash == init_hash == EXPECTED_WEIGHT_HASH)

    # I4 Promotion Gate: G4 mean >= 50%
    i4_gate = (m_g4 >= 0.50 and is_bit_exact)
    stability_diag = all(sr.g4_tok_acc >= 0.40 for sr in per_seed.values())

    if i4_gate and stability_diag:
        classification = "I4_ACHIEVED"
    elif m_g4 >= 0.25:
        classification = "I4_EMERGING"
    else:
        classification = "I4_NOT_ACHIEVED"

    summary_str = (
        f"Strict I4 Re-Evaluation: G1={m_g1*100:.1f}%, G2={m_g2*100:.1f}%, G3={m_g3*100:.1f}%, G4={m_g4*100:.1f}%. "
        f"Hop1 Key={m_h1_k*100:.1f}%, Hop1 Val={m_h1_v*100:.1f}%, Inter={m_int*100:.1f}%, Hop2 Key={m_h2_k*100:.1f}%, Hop2 Val={m_h2_v*100:.1f}%. "
        f"Classification: {classification}."
    )

    return StrictI4ReEvaluationReport(
        per_seed_results=per_seed,
        mean_g1_tok_acc=m_g1,
        mean_g2_tok_acc=m_g2,
        mean_g3_tok_acc=m_g3,
        mean_g4_tok_acc=m_g4,
        mean_hop1_key_acc=m_h1_k,
        mean_hop1_val_acc=m_h1_v,
        mean_inter_preservation=m_int,
        mean_hop2_key_acc=m_h2_k,
        mean_hop2_val_acc=m_h2_v,
        i4_gate_passed=i4_gate,
        stability_diagnostic_passed=stability_diag,
        is_base_bit_exact=is_bit_exact,
        final_classification=classification,
        summary=summary_str,
    )
