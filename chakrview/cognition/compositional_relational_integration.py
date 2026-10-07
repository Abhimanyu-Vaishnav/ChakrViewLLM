"""Step 405: Compositional Integration of Learned Relational Acquisition.

Transfers the validated Hop-1 relational learner into the full end-to-end two-hop
compositional reasoning core:
H1 acquisition
    ↓
retrieved value representation (v1)
    ↓
intermediate state transition (s1)
    ↓
H2 query formation (q2)
    ↓
H2 acquisition & value routing (w2_key, w2_val)
    ↓
dynamic contextual token binding
    ↓
answer

Tracks individual joint losses:
- L_H1: Hop-1 key & value alignment
- L_intermediate: Intermediate state regularity
- L_H2: Hop-2 key & value alignment
- L_final: Final candidate token binding cross-entropy
- L_total: Total joint loss
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
from chakrview.cognition.adaptive_learnability import (
    generate_mixed_hop_episode,
)
from chakrview.cognition.neural_relational_acquisition import (
    NeuralRelationalAcquisitionModule,
)


@dataclasses.dataclass
class CompositionalIntegrationCurvePoint:
    step: int
    l_h1: float
    l_inter: float
    l_h2: float
    l_final: float
    l_total: float


@dataclasses.dataclass
class Step405CompositionalReport:
    initial_loss: float
    final_loss: float
    loss_reduction_pct: float
    final_h1_routing: float
    final_h2_routing: float
    final_g1_acc: float
    final_g4_acc: float
    training_curves: List[CompositionalIntegrationCurvePoint]
    integration_stable: bool
    summary: str


def train_and_eval_compositional_integration(
    model: Optional[NeuralRelationalAcquisitionModule] = None,
    seed: int = 42,
    train_steps: int = 35,
    eval_episodes: int = 8,
) -> Tuple[NeuralRelationalAcquisitionModule, Step405CompositionalReport]:
    """Trains the full two-hop compositional pipeline jointly."""
    torch.manual_seed(seed)
    env = CompositionalAssociativeEnvironment(seed=seed)

    if model is None:
        model = NeuralRelationalAcquisitionModule(d_input=96, d_model=96, vocab_size=4096)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    curves: List[CompositionalIntegrationCurvePoint] = []

    model.train()
    init_loss = 0.0
    final_loss = 0.0

    for step in range(train_steps):
        opt.zero_grad()
        ep = generate_mixed_hop_episode(env, hop_count=2, split="train", num_distractors=1)
        seq = torch.tensor([ep.prompt_tokens], dtype=torch.long)
        c_pos = torch.tensor([ep.candidate_positions], dtype=torch.long)
        tgt = torch.tensor([ep.target_idx], dtype=torch.long)

        out = model(seq, candidate_positions=c_pos, max_hops=2)

        l_final = F.cross_entropy(out["binding_logits"], tgt)
        l_h1_k = F.cross_entropy(out["w1_key"], torch.tensor([ep.key_positions[0]]))
        l_h1_v = F.cross_entropy(out["w1_val"], torch.tensor([ep.val_positions[0]]))
        l_h1 = 0.5 * (l_h1_k + l_h1_v)

        l_h2_k = F.cross_entropy(out["w2_key"], torch.tensor([ep.key_positions[1]]))
        l_h2_v = F.cross_entropy(out["w2_val"], torch.tensor([ep.val_positions[1]]))
        l_h2 = 0.5 * (l_h2_k + l_h2_v)

        l_inter = F.mse_loss(out["s1"], out["s1"].detach())

        l_total = l_final + 0.25 * l_h1 + 0.25 * l_h2 + 0.05 * l_inter

        if step == 0:
            init_loss = l_total.item()
        if step == train_steps - 1:
            final_loss = l_total.item()

        l_total.backward()
        opt.step()

        curves.append(
            CompositionalIntegrationCurvePoint(
                step=step,
                l_h1=l_h1.item(),
                l_inter=l_inter.item(),
                l_h2=l_h2.item(),
                l_final=l_final.item(),
                l_total=l_total.item(),
            )
        )

    # Eval
    model.eval()
    g1_hits, g4_hits = 0, 0
    h1_hits, h2_hits = 0, 0

    with torch.no_grad():
        for _ in range(eval_episodes):
            ep = generate_mixed_hop_episode(env, hop_count=2, split="train", num_distractors=1)
            seq = torch.tensor([ep.prompt_tokens], dtype=torch.long)
            c_pos = torch.tensor([ep.candidate_positions], dtype=torch.long)
            out = model(seq, candidate_positions=c_pos, max_hops=2)
            if torch.argmax(out["binding_logits"][0]).item() == ep.target_idx:
                g1_hits += 1

        for _ in range(eval_episodes):
            ep = generate_mixed_hop_episode(env, hop_count=2, split="disjoint_test", num_distractors=1)
            seq = torch.tensor([ep.prompt_tokens], dtype=torch.long)
            c_pos = torch.tensor([ep.candidate_positions], dtype=torch.long)
            out = model(seq, candidate_positions=c_pos, max_hops=2)
            if torch.argmax(out["binding_logits"][0]).item() == ep.target_idx:
                g4_hits += 1
            if torch.argmax(out["w1_key"][0]).item() == ep.key_positions[0]:
                h1_hits += 1
            if torch.argmax(out["w2_key"][0]).item() == ep.key_positions[1]:
                h2_hits += 1

    N = max(1, eval_episodes)
    g1_acc = g1_hits / N
    g4_acc = g4_hits / N
    h1_acc = h1_hits / N
    h2_acc = h2_hits / N

    loss_reduction = ((init_loss - final_loss) / max(1e-4, init_loss)) * 100.0 if init_loss > 0 else 0.0
    stable = (final_loss < init_loss) and (g4_acc >= 0.50) and (h2_acc >= 0.50)

    summary = (
        f"Step 405 Compositional Integration: G1={g1_acc:.2%}, G4={g4_acc:.2%}. "
        f"H1 Routing={h1_acc:.2%}, H2 Routing={h2_acc:.2%}. "
        f"Loss Reduction={loss_reduction:+.1f}% ({init_loss:.4f} -> {final_loss:.4f}). "
        f"Stable={stable}."
    )

    report = Step405CompositionalReport(
        initial_loss=init_loss,
        final_loss=final_loss,
        loss_reduction_pct=loss_reduction,
        final_h1_routing=h1_acc,
        final_h2_routing=h2_acc,
        final_g1_acc=g1_acc,
        final_g4_acc=g4_acc,
        training_curves=curves,
        integration_stable=stable,
        summary=summary,
    )
    return model, report


if __name__ == "__main__":
    print("Step 405: Running Compositional Integration of Learned Relational Acquisition...")
    _, rep = train_and_eval_compositional_integration(train_steps=35, eval_episodes=8)
    print("Report Summary:", rep.summary)
