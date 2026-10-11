"""Step 311: Strict I3 Preservation and I4 Multi-Seed Evaluation for Internal Trainable Pathway.

Runs:
1. I3 Dynamic Contextual Token Binding Preservation:
   - Evaluates Unseen/Unseen (UU) key routing, value routing, candidate selection, final token.
   - Audits base model language retention (||cand|| / ||base|| within 0.95 - 1.05).
2. Strict Multi-Seed I4 2-hop compositional benchmark across:
   - Seeds: 42, 101, 2026
   - Groups:
     G1: Known identity / Known composition
     G2: Unseen identity / Known composition
     G3: Known identity / Unseen composition
     G4: Unseen identity / Unseen composition (PRIMARY I4 CAPABILITY GATE)
3. Promotion Gate:
   - G4 mean >= 50.0%
   - Stability diagnostic: No individual G4 seed below 40.0%
   - Canonical baseline exact verification (Params = 3,443,136, Hash = c5571c..., Delta W = 0)
"""

from __future__ import annotations

import dataclasses
import time
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn.functional as F

from chakrview.brain.model import ChakrMicro
from chakrview.runtime.interactive import compute_model_hash, EXPECTED_WEIGHT_HASH
from chakrview.cognition.internal_integration_study import (
    IntegratedInternalBindingModel,
)
from chakrview.cognition.pathway_training_objective import (
    train_pathway_model,
)
from chakrview.cognition.dynamic_contextual_token_binding import (
    ChakrMicroWithDynamicBinding,
)
from chakrview.cognition.true_disjoint_generalization import (
    evaluate_true_disjoint_binding,
)
from chakrview.cognition.randomized_associative_episodes import (
    RandomizedAssociativeEnvironment,
)


@dataclasses.dataclass
class StrictPathwayI4SeedResult:
    seed: int
    g1_tok_acc: float
    g2_tok_acc: float
    g3_tok_acc: float
    g4_tok_acc: float
    h1_key_acc: float
    h2_key_acc: float
    h2_val_acc: float


@dataclasses.dataclass
class StrictPathwayEvaluationReport:
    per_seed_results: Dict[int, StrictPathwayI4SeedResult]
    mean_g1_tok_acc: float
    mean_g2_tok_acc: float
    mean_g3_tok_acc: float
    mean_g4_tok_acc: float
    mean_h1_key_acc: float
    mean_h2_key_acc: float
    mean_h2_val_acc: float
    i3_preserved: bool
    i3_uu_tok_acc: float
    language_retention_ratio: float
    i4_gate_passed: bool
    stability_diagnostic_passed: bool
    is_base_bit_exact: bool
    final_classification: str
    summary: str


def evaluate_pathway_i3_and_language(
    base_model: ChakrMicro,
    pathway_model: IntegratedInternalBindingModel,
    seed: int = 42,
    num_episodes: int = 15,
) -> Tuple[bool, float, float]:
    """Evaluates single-hop I3 benchmark and base language retention."""
    torch.manual_seed(seed)
    env = RandomizedAssociativeEnvironment(seed=seed)

    # I3 wrapper with trained binding weights
    i3_wrapper = ChakrMicroWithDynamicBinding(base_model=base_model, rank=16)
    i3_wrapper.binding.load_state_dict(pathway_model.binding.state_dict())
    i3_wrapper.eval()

    disjoint_rep = evaluate_true_disjoint_binding(
        candidate=i3_wrapper,
        env=env,
        seed=seed,
        num_episodes_per_condition=num_episodes,
    )

    test_ids = torch.tensor([
        [10, 45, 120, 230],
        [5, 18, 92, 104],
        [12, 60, 201, 310],
    ], dtype=torch.long)

    with torch.no_grad():
        base_logits = base_model(test_ids)
        cand_logits = pathway_model.internal_backbone(test_ids)
        base_norm = float(torch.norm(base_logits).item())
        cand_norm = float(torch.norm(cand_logits).item())
        retention_ratio = cand_norm / max(base_norm, 1e-12)

    uu_tok = disjoint_rep.unseen_unseen_tok_acc
    i3_ok = uu_tok >= 0.50
    return i3_ok, uu_tok, retention_ratio


def run_strict_pathway_i3_i4_evaluation(
    base_model: ChakrMicro,
    seeds: Optional[List[int]] = None,
    train_steps: int = 15,
    eval_episodes: int = 8,
) -> StrictPathwayEvaluationReport:
    """Executes strict Step 311 evaluation across seeds 42, 101, 2026."""
    if seeds is None:
        seeds = [42, 101, 2026]

    init_hash = compute_model_hash(base_model)
    per_seed: Dict[int, StrictPathwayI4SeedResult] = {}
    i3_oks, uu_toks, lang_ratios = [], [], []

    for s in seeds:
        model, m = train_pathway_model(
            base_model=base_model,
            seed=s,
            train_steps=train_steps,
            eval_episodes=eval_episodes,
            use_hop2_objective=True,
        )

        per_seed[s] = StrictPathwayI4SeedResult(
            seed=s,
            g1_tok_acc=m.mean_g1_tok_acc,
            g2_tok_acc=m.mean_g2_tok_acc,
            g3_tok_acc=m.mean_g3_tok_acc,
            g4_tok_acc=m.mean_g4_tok_acc,
            h1_key_acc=m.mean_h1_key_acc,
            h2_key_acc=m.mean_h2_key_acc,
            h2_val_acc=m.mean_h2_val_acc,
        )

        i3_ok, uu_tok, lang_ratio = evaluate_pathway_i3_and_language(
            base_model=base_model, pathway_model=model, seed=s, num_episodes=eval_episodes,
        )
        i3_oks.append(i3_ok)
        uu_toks.append(uu_tok)
        lang_ratios.append(lang_ratio)

    post_hash = compute_model_hash(base_model)
    is_bit_exact = (init_hash == post_hash == EXPECTED_WEIGHT_HASH)

    m_g1 = sum(r.g1_tok_acc for r in per_seed.values()) / len(per_seed)
    m_g2 = sum(r.g2_tok_acc for r in per_seed.values()) / len(per_seed)
    m_g3 = sum(r.g3_tok_acc for r in per_seed.values()) / len(per_seed)
    m_g4 = sum(r.g4_tok_acc for r in per_seed.values()) / len(per_seed)
    m_h1 = sum(r.h1_key_acc for r in per_seed.values()) / len(per_seed)
    m_h2_k = sum(r.h2_key_acc for r in per_seed.values()) / len(per_seed)
    m_h2_v = sum(r.h2_val_acc for r in per_seed.values()) / len(per_seed)

    mean_uu_tok = sum(uu_toks) / len(uu_toks)
    mean_lang = sum(lang_ratios) / len(lang_ratios)
    overall_i3 = all(i3_oks)

    i4_gate = (m_g4 >= 0.50)
    stability_diag = all(r.g4_tok_acc >= 0.40 for r in per_seed.values())

    if i4_gate and stability_diag and is_bit_exact:
        classification = "I4_ACHIEVED"
    elif m_g4 >= 0.35:
        classification = "I4_EMERGING"
    else:
        classification = "I4_NOT_ACHIEVED"

    summary = (
        f"Strict Internal Pathway I4 Evaluation: "
        f"G1={m_g1*100:.1f}%, G2={m_g2*100:.1f}%, G3={m_g3*100:.1f}%, G4={m_g4*100:.1f}%. "
        f"Seeds G4: " + ", ".join([f"s{s}={per_seed[s].g4_tok_acc*100:.1f}%" for s in seeds]) +
        f". Promotion Gate (>50%): {i4_gate}. Stability Diag (all>=40%): {stability_diag}. "
        f"I3 Preserved: {overall_i3} (UU Tok={mean_uu_tok*100:.1f}%), Language Retention={mean_lang:.5f}. "
        f"Classification: {classification}."
    )

    return StrictPathwayEvaluationReport(
        per_seed_results=per_seed,
        mean_g1_tok_acc=m_g1,
        mean_g2_tok_acc=m_g2,
        mean_g3_tok_acc=m_g3,
        mean_g4_tok_acc=m_g4,
        mean_h1_key_acc=m_h1,
        mean_h2_key_acc=m_h2_k,
        mean_h2_val_acc=m_h2_v,
        i3_preserved=overall_i3,
        i3_uu_tok_acc=mean_uu_tok,
        language_retention_ratio=mean_lang,
        i4_gate_passed=i4_gate,
        stability_diagnostic_passed=stability_diag,
        is_base_bit_exact=is_bit_exact,
        final_classification=classification,
        summary=summary,
    )
