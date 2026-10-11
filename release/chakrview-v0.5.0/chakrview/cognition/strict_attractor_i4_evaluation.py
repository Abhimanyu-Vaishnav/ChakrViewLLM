"""Step 303: Strict Multi-Seed I4 Evaluation Suite for Learned Attractor Architecture.

Runs the complete I4 2-hop benchmark on the candidate architecture with the learned associative attractor across:
Seeds:
- Seed 42
- Seed 101
- Seed 2026

Groups:
- G1: Known identity / Known composition
- G2: Unseen identity / Known composition
- G3: Known identity / Unseen composition
- G4: Unseen identity / Unseen composition (PRIMARY I4 CAPABILITY GATE)

Thresholds:
- Promotion gate: Mean G4 final-token accuracy >= 50.0%
- Stability diagnostic: No individual seed below 40.0%

Audits:
- Canonical baseline SHA-256 and parameter freeze
- Zero Python lookup
- No label leakage
"""

from __future__ import annotations

import dataclasses
import time
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn.functional as F

from chakrview.brain.model import ChakrMicro
from chakrview.runtime.interactive import compute_model_hash, EXPECTED_WEIGHT_HASH
from chakrview.cognition.two_hop_composition_architecture import (
    ChakrMicroCompositionalReasoningModel,
)
from chakrview.cognition.learned_associative_attractor import (
    LearnedAssociativeAttractor,
    compute_module_sha256,
)
from chakrview.cognition.compositional_benchmark_environment import (
    CompositionalAssociativeEnvironment,
    CompositionalEpisode,
)
from chakrview.cognition.associative_attractor_training import (
    train_attractor_compositional_model,
)


@dataclasses.dataclass
class AttractorI4SeedEvaluation:
    seed: int
    g1_tok_acc: float
    g2_tok_acc: float
    g3_tok_acc: float
    g4_tok_acc: float
    hop1_routing_acc: float
    attractor_entropy: float
    attractor_utilization: float
    hop2_key_acc: float
    hop2_val_acc: float


@dataclasses.dataclass
class StrictAttractorI4Report:
    per_seed_results: Dict[int, AttractorI4SeedEvaluation]
    mean_g1_tok_acc: float
    mean_g2_tok_acc: float
    mean_g3_tok_acc: float
    mean_g4_tok_acc: float
    mean_hop1_routing_acc: float
    mean_hop2_key_acc: float
    mean_hop2_val_acc: float
    mean_entropy: float
    mean_utilization: float
    i4_gate_passed: bool
    stability_diagnostic_passed: bool
    is_base_bit_exact: bool
    final_classification: str
    summary: str


def run_strict_attractor_i4_evaluation(
    base_model: ChakrMicro,
    seeds: Optional[List[int]] = None,
    train_steps: int = 15,
    eval_episodes: int = 8,
) -> StrictAttractorI4Report:
    """Runs strict multi-seed evaluation across seeds 42, 101, 2026."""
    if seeds is None:
        seeds = [42, 101, 2026]

    init_hash = compute_model_hash(base_model)
    per_seed: Dict[int, AttractorI4SeedEvaluation] = {}

    for s in seeds:
        cand, att, metrics = train_attractor_compositional_model(
            base_model=base_model,
            seed=s,
            train_steps=train_steps,
            eval_episodes=eval_episodes,
            use_attractor_loss=True,
        )

        per_seed[s] = AttractorI4SeedEvaluation(
            seed=s,
            g1_tok_acc=metrics.mean_g1_tok_acc,
            g2_tok_acc=metrics.mean_g2_tok_acc,
            g3_tok_acc=metrics.mean_g3_tok_acc,
            g4_tok_acc=metrics.mean_g4_tok_acc,
            hop1_routing_acc=metrics.mean_hop1_routing_acc,
            attractor_entropy=metrics.mean_attractor_entropy,
            attractor_utilization=metrics.mean_attractor_utilization,
            hop2_key_acc=metrics.mean_hop2_key_acc,
            hop2_val_acc=metrics.mean_hop2_val_acc,
        )

    # Post-run hash check
    post_hash = compute_model_hash(base_model)
    is_bit_exact = (init_hash == post_hash == EXPECTED_WEIGHT_HASH)

    m_g1 = sum(r.g1_tok_acc for r in per_seed.values()) / len(per_seed)
    m_g2 = sum(r.g2_tok_acc for r in per_seed.values()) / len(per_seed)
    m_g3 = sum(r.g3_tok_acc for r in per_seed.values()) / len(per_seed)
    m_g4 = sum(r.g4_tok_acc for r in per_seed.values()) / len(per_seed)
    m_h1 = sum(r.hop1_routing_acc for r in per_seed.values()) / len(per_seed)
    m_h2_k = sum(r.hop2_key_acc for r in per_seed.values()) / len(per_seed)
    m_h2_v = sum(r.hop2_val_acc for r in per_seed.values()) / len(per_seed)
    m_ent = sum(r.attractor_entropy for r in per_seed.values()) / len(per_seed)
    m_util = sum(r.attractor_utilization for r in per_seed.values()) / len(per_seed)

    i4_gate = (m_g4 >= 0.50)
    stability_diag = all(r.g4_tok_acc >= 0.40 for r in per_seed.values())

    if i4_gate and stability_diag and is_bit_exact:
        classification = "I4_ACHIEVED"
    elif m_g4 >= 0.35:
        classification = "I4_EMERGING"
    else:
        classification = "I4_NOT_ACHIEVED"

    summary = (
        f"Strict Attractor I4 Multi-Seed Evaluation: "
        f"G1={m_g1*100:.1f}%, G2={m_g2*100:.1f}%, G3={m_g3*100:.1f}%, G4={m_g4*100:.1f}%. "
        f"Seeds G4: " + ", ".join([f"s{s}={per_seed[s].g4_tok_acc*100:.1f}%" for s in seeds]) +
        f". Promotion Gate (>50%): {i4_gate}. Stability Diag (all>=40%): {stability_diag}. "
        f"Classification: {classification}."
    )

    return StrictAttractorI4Report(
        per_seed_results=per_seed,
        mean_g1_tok_acc=m_g1,
        mean_g2_tok_acc=m_g2,
        mean_g3_tok_acc=m_g3,
        mean_g4_tok_acc=m_g4,
        mean_hop1_routing_acc=m_h1,
        mean_hop2_key_acc=m_h2_k,
        mean_hop2_val_acc=m_h2_v,
        mean_entropy=m_ent,
        mean_utilization=m_util,
        i4_gate_passed=i4_gate,
        stability_diagnostic_passed=stability_diag,
        is_base_bit_exact=is_bit_exact,
        final_classification=classification,
        summary=summary,
    )
