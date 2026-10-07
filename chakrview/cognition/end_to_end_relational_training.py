"""Step 395: End-to-End Two-Hop Relational Training.

Trains the complete relational chain jointly:
H1 query-key retrieval
 ↓
intermediate state transition
 ↓
H2 query formulation
 ↓
H2 key-value retrieval
 ↓
dynamic contextual token binding
 ↓
final answer

Tracks individual loss components:
- L_H1: Auxiliary Hop-1 query-key alignment loss
- L_intermediate: Intermediate state regularity loss
- L_H2: Auxiliary Hop-2 query-key alignment loss
- L_final: Cross-entropy dynamic candidate token binding loss
- L_total: Joint end-to-end composite loss
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
from chakrview.cognition.relational_initialization_study import (
    InitializedRelationalCore,
)
from chakrview.cognition.adaptive_learnability import (
    generate_mixed_hop_episode,
)


@dataclasses.dataclass
class EndToEndTrainingCurves:
    step: int
    l_h1: float
    l_inter: float
    l_h2: float
    l_final: float
    l_total: float


@dataclasses.dataclass
class Step395EndToEndReport:
    final_h1_routing: float
    final_h2_routing: float
    final_g1_acc: float
    final_g4_acc: float
    loss_reduction_pct: float
    curves: List[EndToEndTrainingCurves]
    convergence_stable: bool
    summary: str


def train_and_eval_end_to_end_core(
    core: Optional[InitializedRelationalCore] = None,
    seed: int = 42,
    train_steps: int = 30,
    eval_episodes: int = 6,
) -> Tuple[InitializedRelationalCore, Step395EndToEndReport]:
    """Trains the end-to-end two-hop relational chain with joint losses."""
    torch.manual_seed(seed)
    env = CompositionalAssociativeEnvironment(seed=seed)

    if core is None:
        core = InitializedRelationalCore(mode="D_orthogonal")

    opt = torch.optim.Adam(core.parameters(), lr=1e-3, weight_decay=1e-4)
    curves: List[EndToEndTrainingCurves] = []

    core.train()
    init_loss = 0.0
    final_loss = 0.0

    for step in range(train_steps):
        opt.zero_grad()
        ep = generate_mixed_hop_episode(env, hop_count=2, split="train", num_distractors=1)
        seq = torch.tensor([ep.prompt_tokens], dtype=torch.long)
        c_pos = torch.tensor([ep.candidate_positions], dtype=torch.long)
        tgt = torch.tensor([ep.target_idx], dtype=torch.long)

        out = core(seq, candidate_positions=c_pos)
        loss_final = F.cross_entropy(out["binding_logits"], tgt)

        w1 = out["w1"]  # [B, n_heads, T, T]
        w2 = out["w2"]  # [B, n_heads, T, T]
        s1 = out["s1"]  # [B, d_state]

        # L_H1: routing from query pos (-1) to Premise 1 key
        l_h1 = 0.0
        if len(ep.key_positions) >= 1 and ep.key_positions[0] >= 0:
            target_k1 = ep.key_positions[0]
            attn1 = w1[0].mean(dim=0)[-1, :]
            l_h1 = -torch.log(attn1[target_k1] + 1e-8)

        # L_H2: routing from query pos (-1) to Premise 2 key
        l_h2 = 0.0
        if len(ep.key_positions) >= 2 and ep.key_positions[1] >= 0:
            target_k2 = ep.key_positions[1]
            attn2 = w2[0].mean(dim=0)[-1, :]
            l_h2 = -torch.log(attn2[target_k2] + 1e-8)

        # L_inter: intermediate state regularity (unit sphere norm penalty)
        l_inter = (torch.norm(s1, p=2, dim=-1) - 1.0).pow(2).mean()

        l_total = loss_final + 0.25 * l_h1 + 0.25 * l_h2 + 0.05 * l_inter
        l_total.backward()
        opt.step()

        if step == 0:
            init_loss = l_total.item()
        if step == train_steps - 1:
            final_loss = l_total.item()

        if step % 5 == 0 or step == train_steps - 1:
            curves.append(
                EndToEndTrainingCurves(
                    step=step,
                    l_h1=float(l_h1.item() if isinstance(l_h1, torch.Tensor) else l_h1),
                    l_inter=float(l_inter.item()),
                    l_h2=float(l_h2.item() if isinstance(l_h2, torch.Tensor) else l_h2),
                    l_final=float(loss_final.item()),
                    l_total=float(l_total.item()),
                )
            )

    # Evaluation
    core.eval()
    h1_hits = 0
    h2_hits = 0
    g1_hits = 0
    g4_hits = 0

    with torch.no_grad():
        # G1 check
        for _ in range(eval_episodes):
            ep = generate_mixed_hop_episode(env, hop_count=2, split="train", num_distractors=1)
            seq = torch.tensor([ep.prompt_tokens], dtype=torch.long)
            c_pos = torch.tensor([ep.candidate_positions], dtype=torch.long)
            out = core(seq, candidate_positions=c_pos)
            if torch.argmax(out["binding_logits"][0]).item() == ep.target_idx:
                g1_hits += 1

        # G4 & Routing check
        for _ in range(eval_episodes):
            ep = generate_mixed_hop_episode(env, hop_count=2, split="disjoint_test", num_distractors=1)
            seq = torch.tensor([ep.prompt_tokens], dtype=torch.long)
            c_pos = torch.tensor([ep.candidate_positions], dtype=torch.long)
            out = core(seq, candidate_positions=c_pos)
            if torch.argmax(out["binding_logits"][0]).item() == ep.target_idx:
                g4_hits += 1

            w1 = out["w1"][0].mean(dim=0)[-1, :]
            if len(ep.key_positions) >= 1 and ep.key_positions[0] >= 0:
                if torch.argmax(w1).item() == ep.key_positions[0]:
                    h1_hits += 1

            w2 = out["w2"][0].mean(dim=0)[-1, :]
            if len(ep.key_positions) >= 2 and ep.key_positions[1] >= 0:
                if torch.argmax(w2).item() == ep.key_positions[1]:
                    h2_hits += 1

    N = max(1, eval_episodes)
    g1_acc = g1_hits / N
    g4_acc = g4_hits / N
    h1_acc = h1_hits / N
    h2_acc = h2_hits / N

    loss_drop = ((init_loss - final_loss) / max(1e-4, init_loss)) * 100.0 if init_loss > 0 else 0.0
    stable = (final_loss < init_loss) and (g4_acc >= 0.30)

    summary = (
        f"End-to-End Two-Hop Training: G1={g1_acc:.2%}, G4={g4_acc:.2%}. "
        f"H1 Key Routing={h1_acc:.2%}, H2 Key Routing={h2_acc:.2%}. "
        f"Loss Reduction={loss_drop:+.1f}% ({init_loss:.4f} -> {final_loss:.4f}). "
        f"Convergence Stable={stable}."
    )

    rep = Step395EndToEndReport(
        final_h1_routing=h1_acc,
        final_h2_routing=h2_acc,
        final_g1_acc=g1_acc,
        final_g4_acc=g4_acc,
        loss_reduction_pct=loss_drop,
        curves=curves,
        convergence_stable=stable,
        summary=summary,
    )
    return core, rep


if __name__ == "__main__":
    print("Step 395: Running End-to-End Two-Hop Training...")
    core, rep = train_and_eval_end_to_end_core(train_steps=25, eval_episodes=6)
    print("Report Summary:", rep.summary)
    for c in rep.curves:
        print(f"  Step {c.step:2d} -> L_total={c.l_total:.4f} (L_final={c.l_final:.4f}, L_H1={c.l_h1:.4f}, L_H2={c.l_h2:.4f})")
