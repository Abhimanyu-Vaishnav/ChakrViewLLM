"""Step 383: Harder Composition (4-Hop) & Resource Adaptation Study.

Tests:
1. 4-Hop Composition Escalation:
   A -> B
   B -> C
   C -> D
   D -> E
   Question: A -> ? Expected: E
   Measures H1, H2, H3, H4 routing and final accuracy.

2. Computation Budget Adaptation:
   Sweeps hard safety budgets:
   max_cycles in [1, 2, 3, 4, 6, 8]
   Measures:
   - Final accuracy under constrained budgets
   - Mean reasoning cycles used
   - Bounded degradation vs opportunity for deeper reasoning
   - Strict resource boundary enforcement
"""

from __future__ import annotations

import dataclasses
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
    train_and_eval_adaptive_learnability,
)


@dataclasses.dataclass
class FourHopResult:
    split_name: str
    final_acc: float
    h1_routing: float
    h2_routing: float
    h3_routing: float
    h4_routing: float
    mean_cycles: float


@dataclasses.dataclass
class ResourceBudgetPoint:
    budget: int
    final_acc: float
    mean_cycles: float
    max_cycles_observed: int
    strictly_bounded: bool


@dataclasses.dataclass
class Step383ResourceAdaptationReport:
    four_hop_results: Dict[str, FourHopResult]
    budget_sweep: Dict[int, ResourceBudgetPoint]
    four_hop_mean_acc: float
    budget_scaling_effective: bool
    summary: str


def run_four_hop_and_budget_adaptation_study(
    core: Optional[AdaptiveRecurrentReasoningCore] = None,
    seed: int = 42,
    episodes: int = 6,
) -> Step383ResourceAdaptationReport:
    """Executes 4-hop composition test and resource budget adaptation sweep."""
    torch.manual_seed(seed)
    env = CompositionalAssociativeEnvironment(seed=seed)
    tok = env.tok

    if core is None:
        core, _ = train_and_eval_adaptive_learnability(seed=seed, train_steps=20)
    core.eval()

    # 1. Four-Hop Composition Diagnostic: A->B->C->D->E
    four_hop_splits = [
        ("known_known", env.TRAIN_KEYS_POOL, env.TRAIN_VALS_POOL),
        ("unseen_unseen", env.DISJOINT_KEYS_POOL, env.DISJOINT_VALS_POOL),
    ]

    four_results: Dict[str, FourHopResult] = {}
    all_4h_acc = []

    for s_name, k_pool, v_pool in four_hop_splits:
        hits = 0
        h1_cnt, h2_cnt, h3_cnt, h4_cnt = 0, 0, 0, 0
        cyc_list = []

        with torch.no_grad():
            for ep_i in range(episodes):
                pool = list(k_pool)
                if len(pool) < 5:
                    pool = pool + list(v_pool)
                sampled = env.rng.sample(pool, 5)
                A, B, C, D, E = sampled[0], sampled[1], sampled[2], sampled[3], sampled[4]
                pairs = [(A, B), (B, C), (C, D), (D, E)]
                env.rng.shuffle(pairs)

                prompt = env.render_prompt(pairs, query_key=A, layout="standard_map")
                tokens = tok.encode(prompt)
                seq = torch.tensor([tokens], dtype=torch.long)

                cand_pos, cand_toks = [], []
                for k, v in pairs:
                    v_enc = tok.encode(v)[0]
                    m = [j for j, t in enumerate(tokens[:-1]) if t == v_enc]
                    if m and m[0] not in cand_pos:
                        cand_pos.append(m[0])
                        cand_toks.append(v_enc)
                if not cand_pos:
                    cand_pos, cand_toks = [0], [tokens[0]]

                c_pos = torch.tensor([cand_pos], dtype=torch.long)
                E_enc = tok.encode(E)[0]
                tgt_idx = cand_toks.index(E_enc) if E_enc in cand_toks else 0

                out = core(seq, candidate_positions=c_pos, max_reasoning_cycles=6, return_trace=True)
                cyc = out["total_cycles_executed"]
                cyc_list.append(cyc)

                pred_idx = torch.argmax(out["binding_logits"][0]).item()
                if pred_idx == tgt_idx:
                    hits += 1

                # Routing positions
                pos_A = [j for j, t in enumerate(tokens) if t == tok.encode(A)[0]]
                pos_B = [j for j, t in enumerate(tokens) if t == tok.encode(B)[0]]
                pos_C = [j for j, t in enumerate(tokens) if t == tok.encode(C)[0]]
                pos_D = [j for j, t in enumerate(tokens) if t == tok.encode(D)[0]]

                traces = out["cycle_traces"]
                if len(traces) >= 1 and traces[0].w_attn is not None and pos_A:
                    w1 = traces[0].w_attn[0].mean(dim=0)[-1, :]
                    if torch.argmax(w1).item() in pos_A:
                        h1_cnt += 1
                if len(traces) >= 2 and traces[1].w_attn is not None and pos_B:
                    w2 = traces[1].w_attn[0].mean(dim=0)[-1, :]
                    if torch.argmax(w2).item() in pos_B:
                        h2_cnt += 1
                if len(traces) >= 3 and traces[2].w_attn is not None and pos_C:
                    w3 = traces[2].w_attn[0].mean(dim=0)[-1, :]
                    if torch.argmax(w3).item() in pos_C:
                        h3_cnt += 1
                if len(traces) >= 4 and traces[3].w_attn is not None and pos_D:
                    w4 = traces[3].w_attn[0].mean(dim=0)[-1, :]
                    if torch.argmax(w4).item() in pos_D:
                        h4_cnt += 1

        N = max(1, episodes)
        acc_4h = hits / N
        all_4h_acc.append(acc_4h)
        four_results[s_name] = FourHopResult(
            split_name=s_name,
            final_acc=acc_4h,
            h1_routing=h1_cnt / N,
            h2_routing=h2_cnt / N,
            h3_routing=h3_cnt / N,
            h4_routing=h4_cnt / N,
            mean_cycles=sum(cyc_list) / N,
        )

    # 2. Computation Budget Adaptation Sweep
    budgets = [1, 2, 3, 4, 6, 8]
    budget_results: Dict[int, ResourceBudgetPoint] = {}

    for b in budgets:
        b_hits = 0
        b_cycles = []
        with torch.no_grad():
            for ep_i in range(episodes):
                ep = generate_mixed_hop_episode(env, hop_count=3, split="disjoint_test", num_distractors=1)
                seq = torch.tensor([ep.prompt_tokens], dtype=torch.long)
                c_pos = torch.tensor([ep.candidate_positions], dtype=torch.long)

                out = core(seq, candidate_positions=c_pos, max_reasoning_cycles=b)
                c_used = out["total_cycles_executed"]
                b_cycles.append(c_used)

                pred_idx = torch.argmax(out["binding_logits"][0]).item()
                if pred_idx == ep.target_idx:
                    b_hits += 1

        N = max(1, episodes)
        max_c = max(b_cycles) if b_cycles else 0
        bounded = (max_c <= b)
        budget_results[b] = ResourceBudgetPoint(
            budget=b,
            final_acc=b_hits / N,
            mean_cycles=sum(b_cycles) / N,
            max_cycles_observed=max_c,
            strictly_bounded=bounded,
        )

    m_4h = sum(all_4h_acc) / len(all_4h_acc)
    all_bounded = all(pt.strictly_bounded for pt in budget_results.values())

    summary = (
        f"4-Hop & Resource Adaptation: 4-Hop Acc={m_4h:.2%}. "
        f"Budgets evaluated={budgets}. Strict Boundary Respected={all_bounded}. "
        f"Budget 1 Acc={budget_results[1].final_acc:.2%}, Budget 8 Acc={budget_results[8].final_acc:.2%}."
    )

    return Step383ResourceAdaptationReport(
        four_hop_results=four_results,
        budget_sweep=budget_results,
        four_hop_mean_acc=m_4h,
        budget_scaling_effective=all_bounded,
        summary=summary,
    )


if __name__ == "__main__":
    print("Step 383: Running 4-Hop Escalation and Resource Budget Adaptation...")
    rep = run_four_hop_and_budget_adaptation_study(episodes=6)
    print("Report Summary:", rep.summary)
    for s, r in rep.four_hop_results.items():
        print(f"  4-Hop Split {s:15s}: Acc={r.final_acc:.2%}, H1={r.h1_routing:.2%}, H2={r.h2_routing:.2%}, H3={r.h3_routing:.2%}, H4={r.h4_routing:.2%}")
    for b, pt in rep.budget_sweep.items():
        print(f"  Budget {b}: Acc={pt.final_acc:.2%}, MeanC={pt.mean_cycles:.2f}, MaxObs={pt.max_cycles_observed}, Bounded={pt.strictly_bounded}")
