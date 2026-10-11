"""Step 375: Three-Hop Composition Escalation Diagnostic.

Tests 3-hop composition chain:
A -> B
B -> C
C -> D
Question: A -> ?
Expected: D

Evaluates whether the core's recurrent-attention architecture can generalize to 3 hops:
- known/known
- unseen/known
- known/unseen
- unseen/unseen

Measures:
- H1 routing (A -> B)
- H2 routing (B -> C)
- H3 routing (C -> D)
- Final token accuracy
- Exact degradation boundary identification

Records boundary without claiming I5 capability prematurely.
"""

from __future__ import annotations

import dataclasses
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from chakrview.cognition.unified_compositional_core import UnifiedCompositionalCore
from chakrview.cognition.compositional_benchmark_environment import (
    CompositionalAssociativeEnvironment,
)


@dataclasses.dataclass
class ThreeHopSplitResult:
    split_name: str
    h1_routing_acc: float
    h2_routing_acc: float
    h3_routing_acc: float
    final_token_acc: float
    sample_count: int


@dataclasses.dataclass
class ThreeHopEscalationReport:
    splits: Dict[str, ThreeHopSplitResult]
    mean_h1: float
    mean_h2: float
    mean_h3: float
    mean_final_acc: float
    degradation_boundary: str
    eligible_for_three_hop: bool
    summary: str


def run_three_hop_escalation_diagnostic(
    core: UnifiedCompositionalCore,
    seed: int = 42,
    episodes_per_split: int = 6,
) -> ThreeHopEscalationReport:
    """Evaluates UnifiedCompositionalCore on 3-hop chains A->B->C->D."""
    torch.manual_seed(seed)
    env = CompositionalAssociativeEnvironment(seed=seed)
    tok = env.tok
    core.eval()

    splits_to_test = [
        ("known_known", env.TRAIN_KEYS_POOL, env.TRAIN_VALS_POOL),
        ("unseen_known", env.DISJOINT_KEYS_POOL, env.TRAIN_VALS_POOL),
        ("known_unseen", env.TRAIN_KEYS_POOL, env.DISJOINT_VALS_POOL),
        ("unseen_unseen", env.DISJOINT_KEYS_POOL, env.DISJOINT_VALS_POOL),
    ]

    split_results: Dict[str, ThreeHopSplitResult] = {}

    all_h1 = []
    all_h2 = []
    all_h3 = []
    all_acc = []

    for split_name, key_pool, val_pool in splits_to_test:
        h1_corr = 0
        h2_corr = 0
        h3_corr = 0
        tok_corr = 0

        with torch.no_grad():
            for ep_i in range(episodes_per_split):
                # Sample 4 symbols A, B, C, D
                pool = list(key_pool)
                if len(pool) < 4:
                    pool = pool + list(val_pool)
                sampled = env.rng.sample(pool, 4)
                A, B, C, D = sampled[0], sampled[1], sampled[2], sampled[3]

                pairs = [(A, B), (B, C), (C, D)]
                env.rng.shuffle(pairs)

                prompt = env.render_prompt(pairs, query_key=A, layout="standard_map")
                tokens = tok.encode(prompt)
                seq = torch.tensor([tokens], dtype=torch.long)

                cand_positions, cand_tokens = [], []
                for k, v in pairs:
                    v_enc = tok.encode(v)[0]
                    pos_list = [j for j, t in enumerate(tokens[:-1]) if t == v_enc]
                    if pos_list and pos_list[0] not in cand_positions:
                        cand_positions.append(pos_list[0])
                        cand_tokens.append(v_enc)
                if not cand_positions:
                    cand_positions, cand_tokens = [0], [tokens[0]]

                c_pos = torch.tensor([cand_positions], dtype=torch.long)
                D_enc = tok.encode(D)[0]
                tgt_idx = cand_tokens.index(D_enc) if D_enc in cand_tokens else 0

                out = core(
                    input_ids=seq,
                    candidate_positions=c_pos,
                    return_trace=True,
                )

                if "binding_logits" in out and out["binding_logits"] is not None:
                    pred_idx = torch.argmax(out["binding_logits"][0]).item()
                    if pred_idx == tgt_idx:
                        tok_corr += 1
                else:
                    pred_tok = torch.argmax(out["vocab_logits"][0, -1, :]).item()
                    if pred_tok == D_enc:
                        tok_corr += 1

                # Routing positions
                A_enc = tok.encode(A)[0]
                B_enc = tok.encode(B)[0]
                C_enc = tok.encode(C)[0]

                pos_A = [j for j, t in enumerate(tokens) if t == A_enc]
                pos_B = [j for j, t in enumerate(tokens) if t == B_enc]
                pos_C = [j for j, t in enumerate(tokens) if t == C_enc]

                tr = out.get("trace")
                if tr is not None:
                    if tr.w1 is not None:
                        w1 = tr.w1[0].mean(dim=0)[-1, :]
                        top1 = torch.argmax(w1).item()
                        if pos_A and top1 in pos_A:
                            h1_corr += 1
                    if tr.w2 is not None:
                        w2 = tr.w2[0].mean(dim=0)[-1, :]
                        top2 = torch.argmax(w2).item()
                        if pos_B and top2 in pos_B:
                            h2_corr += 1
                        if pos_C and top2 in pos_C:
                            h3_corr += 1

        n = max(episodes_per_split, 1)
        h1_acc = h1_corr / n
        h2_acc = h2_corr / n
        h3_acc = h3_corr / n
        final_acc = tok_corr / n

        split_results[split_name] = ThreeHopSplitResult(
            split_name=split_name,
            h1_routing_acc=h1_acc,
            h2_routing_acc=h2_acc,
            h3_routing_acc=h3_acc,
            final_token_acc=final_acc,
            sample_count=n,
        )

        all_h1.append(h1_acc)
        all_h2.append(h2_acc)
        all_h3.append(h3_acc)
        all_acc.append(final_acc)

    mean_h1 = sum(all_h1) / len(all_h1)
    mean_h2 = sum(all_h2) / len(all_h2)
    mean_h3 = sum(all_h3) / len(all_h3)
    mean_final = sum(all_acc) / len(all_acc)

    # Identify degradation boundary
    if mean_h1 < 0.30:
        boundary = "Hop-1 Routing Collapse (unable to bind initial query key)"
    elif mean_h2 < 0.25:
        boundary = "Hop-2 Transition Collapse (state transition loses intermediate reference B)"
    elif mean_h3 < 0.20:
        boundary = "Hop-3 Capacity Boundary (2-cycle core architecture lacks third recurrence cycle for C->D)"
    else:
        boundary = "Preserved 3-Hop Capability"

    eligible = (mean_h2 >= 0.35) and (mean_final >= 0.25)

    summary = (
        f"Three-Hop Diagnostic: mean H1={mean_h1:.2%}, H2={mean_h2:.2%}, H3={mean_h3:.2%}, "
        f"final_acc={mean_final:.2%}. Degradation Boundary: {boundary}."
    )

    return ThreeHopEscalationReport(
        splits=split_results,
        mean_h1=mean_h1,
        mean_h2=mean_h2,
        mean_h3=mean_h3,
        mean_final_acc=mean_final,
        degradation_boundary=boundary,
        eligible_for_three_hop=eligible,
        summary=summary,
    )


if __name__ == "__main__":
    from chakrview.cognition.compositional_curriculum_training import train_compositional_curriculum

    print("Step 375: Evaluating 3-hop escalation diagnostic...")
    core = UnifiedCompositionalCore()
    train_compositional_curriculum(core=core, steps_per_level=5, seed=42)
    rep = run_three_hop_escalation_diagnostic(core, seed=42, episodes_per_split=6)
    print("Report Summary:", rep.summary)
    for name, r in rep.splits.items():
        print(f"Split {name}: final_acc={r.final_token_acc:.2%}, h1={r.h1_routing_acc:.2%}, h2={r.h2_routing_acc:.2%}, h3={r.h3_routing_acc:.2%}")
