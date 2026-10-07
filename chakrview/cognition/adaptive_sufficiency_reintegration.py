"""Step 397: Adaptive Sufficiency Reintegration into Stabilized Relational Core.

Reintegrates the AnswerSufficiencyController with the stabilized orthogonal relational attention
architecture from Steps 393-395.

Compares:
A. Fixed 1 cycle
B. Fixed 2 cycles
C. Fixed 3 cycles
D. Adaptive sufficiency controller (dynamic halting threshold 0.5)

Measures:
- H1 key and value routing
- H2 key routing
- G1, G2, G3, G4 accuracies
- Average reasoning cycles executed
- Premature halt rate on multi-hop tasks
- Unnecessary continuation rate on single-hop tasks
"""

from __future__ import annotations

import dataclasses
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from chakrview.cognition.compositional_benchmark_environment import (
    CompositionalAssociativeEnvironment,
)
from chakrview.cognition.sufficiency_adaptive_reasoning import (
    SufficiencyAdaptiveReasoningCore,
)
from chakrview.cognition.adaptive_learnability import (
    generate_mixed_hop_episode,
)


@dataclasses.dataclass
class ReintegrationVariantResult:
    variant_id: str
    description: str
    g1_acc: float
    g2_acc: float
    g3_acc: float
    g4_acc: float
    h1_routing: float
    h2_routing: float
    mean_cycles: float
    premature_halt_rate: float
    unnecessary_rate: float


@dataclasses.dataclass
class Step397ReintegrationReport:
    variants: Dict[str, ReintegrationVariantResult]
    best_variant: str
    adaptive_preserves_accuracy: bool
    summary: str


def train_and_eval_reintegrated_variant(
    variant_id: str,
    seed: int = 42,
    train_steps: int = 25,
    eval_episodes_per_hop: int = 6,
) -> ReintegrationVariantResult:
    """Trains and tests core under fixed vs adaptive controller configurations."""
    torch.manual_seed(seed)
    env = CompositionalAssociativeEnvironment(seed=seed)

    core = SufficiencyAdaptiveReasoningCore()
    # Apply orthogonal initialization to Q/K projections
    for p in [core.q0_proj.weight, core.k0_proj.weight, core.k_rec_proj.weight]:
        nn.init.orthogonal_(p)

    opt = torch.optim.Adam(core.parameters(), lr=1e-3, weight_decay=1e-4)

    fixed_c = None
    if variant_id == "A_Fixed1": fixed_c = 1
    elif variant_id == "B_Fixed2": fixed_c = 2
    elif variant_id == "C_Fixed3": fixed_c = 3

    core.train()
    hops = [1, 2, 3]

    for step in range(train_steps):
        opt.zero_grad()
        h_k = hops[step % len(hops)]
        ep = generate_mixed_hop_episode(env, hop_count=h_k, split="train", num_distractors=1)
        seq = torch.tensor([ep.prompt_tokens], dtype=torch.long)
        c_pos = torch.tensor([ep.candidate_positions], dtype=torch.long)
        tgt = torch.tensor([ep.target_idx], dtype=torch.long)

        out = core(seq, candidate_positions=c_pos, fixed_cycles=fixed_c, max_reasoning_cycles=4, return_trace=True)
        loss_ans = F.cross_entropy(out["binding_logits"], tgt)

        # Sufficiency loss
        loss_suff = 0.0
        if variant_id == "D_AdaptiveSufficiency":
            traces = out["cycle_traces"]
            for t_idx, trace in enumerate(traces):
                target_suff = 1.0 if (t_idx + 1) >= h_k else 0.0
                loss_suff += F.binary_cross_entropy(trace.sufficiency_score, torch.tensor([[target_suff]]))

        # Relational alignment loss on Cycle 1 & Cycle 2
        loss_align = 0.0
        traces = out["cycle_traces"]
        if len(traces) >= 1 and traces[0].w_attn is not None and len(ep.key_positions) >= 1 and ep.key_positions[0] >= 0:
            attn1 = traces[0].w_attn[0].mean(dim=0)[-1, :]
            loss_align += -torch.log(attn1[ep.key_positions[0]] + 1e-8)
        if len(traces) >= 2 and traces[1].w_attn is not None and len(ep.key_positions) >= 2 and ep.key_positions[1] >= 0:
            attn2 = traces[1].w_attn[0].mean(dim=0)[-1, :]
            loss_align += -torch.log(attn2[ep.key_positions[1]] + 1e-8)

        total_loss = loss_ans + 0.25 * loss_align + 0.2 * loss_suff
        total_loss.backward()
        opt.step()

    # Evaluation
    core.eval()
    def eval_split_metric(split_name: str) -> Tuple[float, float, float, float, float, float]:
        hits = 0
        h1_cnt, h2_cnt = 0, 0
        cyc_list = []
        prem_cnt, unnec_cnt = 0, 0

        with torch.no_grad():
            for h_k in [1, 2, 3]:
                for _ in range(eval_episodes_per_hop):
                    ep = generate_mixed_hop_episode(env, hop_count=h_k, split=split_name, num_distractors=1)
                    seq = torch.tensor([ep.prompt_tokens], dtype=torch.long)
                    c_pos = torch.tensor([ep.candidate_positions], dtype=torch.long)

                    out = core(seq, candidate_positions=c_pos, fixed_cycles=fixed_c, max_reasoning_cycles=5, return_trace=True)
                    c_used = out["total_cycles_executed"]
                    cyc_list.append(c_used)

                    if torch.argmax(out["binding_logits"][0]).item() == ep.target_idx:
                        hits += 1

                    if c_used < h_k: prem_cnt += 1
                    elif c_used > h_k: unnec_cnt += 1

                    traces = out["cycle_traces"]
                    if len(traces) >= 1 and traces[0].w_attn is not None and len(ep.key_positions) >= 1 and ep.key_positions[0] >= 0:
                        if torch.argmax(traces[0].w_attn[0].mean(dim=0)[-1, :]).item() == ep.key_positions[0]:
                            h1_cnt += 1
                    if len(traces) >= 2 and traces[1].w_attn is not None and len(ep.key_positions) >= 2 and ep.key_positions[1] >= 0:
                        if torch.argmax(traces[1].w_attn[0].mean(dim=0)[-1, :]).item() == ep.key_positions[1]:
                            h2_cnt += 1

        tot = 3 * eval_episodes_per_hop
        return (
            hits / tot,
            h1_cnt / tot,
            h2_cnt / tot,
            sum(cyc_list) / tot,
            prem_cnt / tot,
            unnec_cnt / tot,
        )

    g1, _, _, _, _, _ = eval_split_metric("train")
    g2, _, _, _, _, _ = eval_split_metric("val")
    g3, _, _, _, _, _ = eval_split_metric("heldout_composition")
    g4, h1_r, h2_r, m_c, prem, unnec = eval_split_metric("disjoint_test")

    descriptions = {
        "A_Fixed1": "Fixed 1 Reasoning Cycle",
        "B_Fixed2": "Fixed 2 Reasoning Cycles",
        "C_Fixed3": "Fixed 3 Reasoning Cycles",
        "D_AdaptiveSufficiency": "Reintegrated Adaptive Sufficiency Controller",
    }

    return ReintegrationVariantResult(
        variant_id=variant_id,
        description=descriptions[variant_id],
        g1_acc=g1,
        g2_acc=g2,
        g3_acc=g3,
        g4_acc=g4,
        h1_routing=h1_r,
        h2_routing=h2_r,
        mean_cycles=m_c,
        premature_halt_rate=prem,
        unnecessary_rate=unnec,
    )


def run_adaptive_sufficiency_reintegration_study(
    seed: int = 42,
    train_steps: int = 20,
    eval_episodes_per_hop: int = 4,
) -> Step397ReintegrationReport:
    """Executes Step 397 comparison across fixed vs adaptive sufficiency."""
    variants = [
        "A_Fixed1",
        "B_Fixed2",
        "C_Fixed3",
        "D_AdaptiveSufficiency",
    ]

    results: Dict[str, ReintegrationVariantResult] = {}
    for var_id in variants:
        res = train_and_eval_reintegrated_variant(
            var_id, seed=seed, train_steps=train_steps, eval_episodes_per_hop=eval_episodes_per_hop
        )
        results[var_id] = res

    best_v = max(results.keys(), key=lambda k: (results[k].g4_acc, results[k].h2_routing))
    adapt_g4 = results["D_AdaptiveSufficiency"].g4_acc
    fixed_g4 = results["B_Fixed2"].g4_acc
    preserves = (adapt_g4 >= fixed_g4 - 0.10)

    summary = (
        f"Sufficiency Reintegration Study: Best={best_v}. Adaptive G4={adapt_g4:.2%} vs Fixed2 G4={fixed_g4:.2%}. "
        f"Adaptive H1={results['D_AdaptiveSufficiency'].h1_routing:.2%}, "
        f"Adaptive H2={results['D_AdaptiveSufficiency'].h2_routing:.2%}, "
        f"Mean Cycles={results['D_AdaptiveSufficiency'].mean_cycles:.2f}."
    )

    return Step397ReintegrationReport(
        variants=results,
        best_variant=best_v,
        adaptive_preserves_accuracy=preserves,
        summary=summary,
    )


if __name__ == "__main__":
    print("Step 397: Running Adaptive Sufficiency Reintegration Study...")
    rep = run_adaptive_sufficiency_reintegration_study(seed=42, train_steps=15, eval_episodes_per_hop=4)
    print("Report Summary:", rep.summary)
    for v_id, r in rep.variants.items():
        print(f"  [{v_id:22s}] G4={r.g4_acc:.2%}, H1={r.h1_routing:.2%}, H2={r.h2_routing:.2%}, Cycles={r.mean_cycles:.2f}")
