"""Step 379: Adaptive vs Fixed Depth Control Study.

Compares matched candidates under equal parameter budgets:
A. Fixed 1 cycle
B. Fixed 2 cycles
C. Fixed 3 cycles
D. Adaptive controller (dynamic halting threshold 0.5)
E. Adaptive controller with controller disabled (bypass_controller=True, runs to max_cycles)
F. Adaptive controller with random continuation decisions (random_controller=True)

Evaluates:
- G1: Known ID / Known Composition
- G2: Unseen ID / Known Composition
- G3: Known ID / Unseen Composition
- G4: Unseen ID / Unseen Composition (Primary I4 Gate diagnostic)
- H1, H2, H3 routing accuracy
- Final token accuracy
- Mean, median, and max cycles executed across task difficulties
"""

from __future__ import annotations

import dataclasses
import time
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from chakrview.cognition.adaptive_recurrent_reasoning_core import AdaptiveRecurrentReasoningCore
from chakrview.cognition.compositional_benchmark_environment import (
    CompositionalAssociativeEnvironment,
)
from chakrview.cognition.adaptive_learnability import (
    generate_mixed_hop_episode,
    AdaptiveEpisode,
)


@dataclasses.dataclass
class DepthControlVariantResult:
    variant_id: str
    description: str
    train_loss: float
    g1_acc: float
    g2_acc: float
    g3_acc: float
    g4_acc: float
    h1_routing: float
    h2_routing: float
    h3_routing: float
    mean_cycles: float
    max_cycles: int
    cycles_1hop: float
    cycles_2hop: float
    cycles_3hop: float


@dataclasses.dataclass
class Step379DepthControlReport:
    variants: Dict[str, DepthControlVariantResult]
    best_variant: str
    adaptive_outperforms_fixed: bool
    generalization_delta: float
    summary: str


def train_depth_variant(
    core: AdaptiveRecurrentReasoningCore,
    variant_id: str,
    env: CompositionalAssociativeEnvironment,
    train_steps: int = 20,
) -> float:
    """Trains core according to variant mode."""
    optimizer = torch.optim.Adam(core.parameters(), lr=1e-3, weight_decay=1e-4)
    core.train()
    tot_loss = 0.0

    fixed_c = None
    bypass = False
    random_c = False

    if variant_id == "A_Fixed1":
        fixed_c = 1
    elif variant_id == "B_Fixed2":
        fixed_c = 2
    elif variant_id == "C_Fixed3":
        fixed_c = 3
    elif variant_id == "E_BypassController":
        bypass = True
    elif variant_id == "F_RandomController":
        random_c = True

    hops = [1, 2, 3]

    for step in range(train_steps):
        optimizer.zero_grad()
        hop_k = hops[step % len(hops)]
        ep = generate_mixed_hop_episode(env, hop_count=hop_k, split="train", num_distractors=1)

        seq = torch.tensor([ep.prompt_tokens], dtype=torch.long)
        c_pos = torch.tensor([ep.candidate_positions], dtype=torch.long)
        tgt = torch.tensor([ep.target_idx], dtype=torch.long)

        out = core(
            input_ids=seq,
            candidate_positions=c_pos,
            fixed_cycles=fixed_c,
            max_reasoning_cycles=4,
            bypass_controller=bypass,
            random_controller=random_c,
        )

        loss_bind = F.cross_entropy(out["binding_logits"], tgt)

        # Halt loss for adaptive variants
        loss_halt = 0.0
        if variant_id in ["D_Adaptive", "E_BypassController"]:
            for c_i, p_val in enumerate(out["p_continue_per_cycle"]):
                target_p = 1.0 if (c_i + 1) < hop_k else 0.0
                loss_halt += F.binary_cross_entropy(p_val, torch.tensor([[target_p]]))

        loss = loss_bind + 0.3 * loss_halt
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()

    return tot_loss / max(1, train_steps)


def eval_variant_on_split(
    core: AdaptiveRecurrentReasoningCore,
    variant_id: str,
    env: CompositionalAssociativeEnvironment,
    split: str,
    episodes_per_hop: int = 4,
) -> Tuple[float, float, float, float, float, int, float, float, float]:
    """Returns (acc, h1, h2, h3, mean_c, max_c, c_1h, c_2h, c_3h)."""
    core.eval()
    fixed_c = None
    bypass = False
    random_c = False

    if variant_id == "A_Fixed1":
        fixed_c = 1
    elif variant_id == "B_Fixed2":
        fixed_c = 2
    elif variant_id == "C_Fixed3":
        fixed_c = 3
    elif variant_id == "E_BypassController":
        bypass = True
    elif variant_id == "F_RandomController":
        random_c = True

    hits = 0
    h1_hits = 0
    h2_hits = 0
    h3_hits = 0
    cycles_all = []
    cycles_by_hop = {1: [], 2: [], 3: []}

    with torch.no_grad():
        for hop_k in [1, 2, 3]:
            for ep_i in range(episodes_per_hop):
                ep = generate_mixed_hop_episode(env, hop_count=hop_k, split=split, num_distractors=1)
                seq = torch.tensor([ep.prompt_tokens], dtype=torch.long)
                c_pos = torch.tensor([ep.candidate_positions], dtype=torch.long)

                out = core(
                    input_ids=seq,
                    candidate_positions=c_pos,
                    fixed_cycles=fixed_c,
                    max_reasoning_cycles=5,
                    bypass_controller=bypass,
                    random_controller=random_c,
                    return_trace=True,
                )

                c_used = out["total_cycles_executed"]
                cycles_all.append(c_used)
                cycles_by_hop[hop_k].append(c_used)

                pred_idx = torch.argmax(out["binding_logits"][0]).item()
                if pred_idx == ep.target_idx:
                    hits += 1

                # Routing
                traces = out["cycle_traces"]
                if len(traces) >= 1 and traces[0].w_attn is not None:
                    w1 = traces[0].w_attn[0].mean(dim=0)[-1, :]
                    if ep.key_positions[0] >= 0 and torch.argmax(w1).item() == ep.key_positions[0]:
                        h1_hits += 1
                if len(traces) >= 2 and traces[1].w_attn is not None:
                    w2 = traces[1].w_attn[0].mean(dim=0)[-1, :]
                    if len(ep.key_positions) >= 2 and ep.key_positions[1] >= 0 and torch.argmax(w2).item() == ep.key_positions[1]:
                        h2_hits += 1
                if len(traces) >= 3 and traces[2].w_attn is not None:
                    w3 = traces[2].w_attn[0].mean(dim=0)[-1, :]
                    if len(ep.key_positions) >= 3 and ep.key_positions[2] >= 0 and torch.argmax(w3).item() == ep.key_positions[2]:
                        h3_hits += 1

    tot = 3 * episodes_per_hop
    acc = hits / max(1, tot)
    h1 = h1_hits / max(1, tot)
    h2 = h2_hits / max(1, tot)
    h3 = h3_hits / max(1, tot)
    m_c = sum(cycles_all) / max(1, len(cycles_all))
    mx_c = max(cycles_all) if cycles_all else 0
    c_1 = sum(cycles_by_hop[1]) / max(1, len(cycles_by_hop[1]))
    c_2 = sum(cycles_by_hop[2]) / max(1, len(cycles_by_hop[2]))
    c_3 = sum(cycles_by_hop[3]) / max(1, len(cycles_by_hop[3]))

    return acc, h1, h2, h3, m_c, mx_c, c_1, c_2, c_3


def run_adaptive_vs_fixed_depth_study(
    seed: int = 42,
    train_steps: int = 20,
    eval_episodes_per_hop: int = 4,
) -> Step379DepthControlReport:
    """Executes Step 379 comparison across Variants A-F."""
    torch.manual_seed(seed)
    env = CompositionalAssociativeEnvironment(seed=seed)

    variants_config = [
        ("A_Fixed1", "Fixed 1 Reasoning Cycle"),
        ("B_Fixed2", "Fixed 2 Reasoning Cycles"),
        ("C_Fixed3", "Fixed 3 Reasoning Cycles"),
        ("D_Adaptive", "Adaptive Neural Halting Controller"),
        ("E_BypassController", "Adaptive Controller Disabled (Bypassed to Max Cycles)"),
        ("F_RandomController", "Adaptive Controller with Random Continuation"),
    ]

    results: Dict[str, DepthControlVariantResult] = {}

    for var_id, desc in variants_config:
        core = AdaptiveRecurrentReasoningCore()
        loss = train_depth_variant(core, var_id, env, train_steps=train_steps)

        g1, _, _, _, _, _, _, _, _ = eval_variant_on_split(core, var_id, env, "train", eval_episodes_per_hop)
        g2, _, _, _, _, _, _, _, _ = eval_variant_on_split(core, var_id, env, "val", eval_episodes_per_hop)
        g3, _, _, _, _, _, _, _, _ = eval_variant_on_split(core, var_id, env, "heldout_composition", eval_episodes_per_hop)
        g4, h1, h2, h3, m_c, mx_c, c1, c2, c3 = eval_variant_on_split(core, var_id, env, "disjoint_test", eval_episodes_per_hop)

        results[var_id] = DepthControlVariantResult(
            variant_id=var_id,
            description=desc,
            train_loss=loss,
            g1_acc=g1,
            g2_acc=g2,
            g3_acc=g3,
            g4_acc=g4,
            h1_routing=h1,
            h2_routing=h2,
            h3_routing=h3,
            mean_cycles=m_c,
            max_cycles=mx_c,
            cycles_1hop=c1,
            cycles_2hop=c2,
            cycles_3hop=c3,
        )

    # Comparison metrics
    adapt_g4 = results["D_Adaptive"].g4_acc
    best_fixed_g4 = max(results["A_Fixed1"].g4_acc, results["B_Fixed2"].g4_acc, results["C_Fixed3"].g4_acc)
    g_delta = adapt_g4 - best_fixed_g4
    adaptive_wins = (g_delta >= 0.0)

    best_v = max(results.keys(), key=lambda k: results[k].g4_acc)

    summary = (
        f"Depth Control Study: Best={best_v}. Adaptive G4={adapt_g4:.2%}, "
        f"Best Fixed G4={best_fixed_g4:.2%} (Delta={g_delta:+.2%}). "
        f"Adaptive Cycles: 1h={results['D_Adaptive'].cycles_1hop:.1f}, "
        f"2h={results['D_Adaptive'].cycles_2hop:.1f}, 3h={results['D_Adaptive'].cycles_3hop:.1f}."
    )

    return Step379DepthControlReport(
        variants=results,
        best_variant=best_v,
        adaptive_outperforms_fixed=adaptive_wins,
        generalization_delta=g_delta,
        summary=summary,
    )


if __name__ == "__main__":
    print("Step 379: Running Adaptive vs Fixed Depth Control Study...")
    rep = run_adaptive_vs_fixed_depth_study(seed=42, train_steps=15, eval_episodes_per_hop=4)
    print("Report Summary:", rep.summary)
    for v_id, r in rep.variants.items():
        print(f"  [{v_id:20s}] G4={r.g4_acc:.2%}, H1={r.h1_routing:.2%}, H2={r.h2_routing:.2%}, MeanC={r.mean_cycles:.2f}")
