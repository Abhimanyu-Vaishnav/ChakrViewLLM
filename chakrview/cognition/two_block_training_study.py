"""Step 349: Two-Block Controlled Training Study.

Compares:
Option A: Best Single Trainable Block (Layer 3, exactly 442,752 block params)
Option B: Adjacent Two-Block Candidate (Layer 2 + Layer 3, exactly 885,504 block params)

Investigates:
Does training a second complete transformer block enable true compositional transformation,
or does it merely increase in-sample memorization without improving unseen generalization (G4)?

Audits across:
- G1 (Known ID / Known Comp)
- G2 (Unseen ID / Known Comp)
- G3 (Known ID / Unseen Comp)
- G4 (Unseen ID / Unseen Comp)
"""

from __future__ import annotations

import dataclasses
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from chakrview.brain.model import ChakrMicro
from chakrview.cognition.trainable_transformer_block import (
    TrainableTransformerBlockCandidate,
)
from chakrview.cognition.training_objective_study import (
    evaluate_candidate_performance,
)
from chakrview.cognition.compositional_benchmark_environment import (
    CompositionalAssociativeEnvironment,
)
from chakrview.cognition.identity_invariant_objective import (
    IdentityInvariantRepresentationLoss,
)


@dataclasses.dataclass
class BlockComparisonMetrics:
    name: str
    trainable_layers: List[int]
    trainable_params: int
    g1_acc: float
    g2_acc: float
    g3_acc: float
    g4_acc: float
    h1_key_acc: float
    h2_key_acc: float
    h2_val_acc: float
    summary: str


@dataclasses.dataclass
class TwoBlockComparisonReport:
    single_block_metrics: BlockComparisonMetrics
    two_block_metrics: BlockComparisonMetrics
    two_block_improves_g4: bool
    g4_delta: float
    summary: str


def train_block_candidate(
    base_model: ChakrMicro,
    trainable_layers: List[int],
    seed: int = 42,
    train_steps: int = 15,
) -> TrainableTransformerBlockCandidate:
    """Trains TrainableTransformerBlockCandidate on specified layer indices."""
    torch.manual_seed(seed)
    env = CompositionalAssociativeEnvironment(seed=seed)
    tok = env.tok

    candidate = TrainableTransformerBlockCandidate(
        base_model=base_model,
        trainable_layers=trainable_layers,
        enable_contextual_binding=True,
    )

    train_params = [p for p in candidate.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(train_params, lr=1.0e-3, weight_decay=0.01)
    route_loss_fn = IdentityInvariantRepresentationLoss(temperature=0.15)

    candidate.train()
    for st in range(train_steps):
        ep = env.generate_episode(split="train", num_distractors=1, episode_idx=349000 + st)
        inp = torch.tensor([ep.prompt_tokens], dtype=torch.long)
        opt.zero_grad()

        final_h, _ = candidate.forward_hidden_states(inp)
        h_norm = F.normalize(final_h[0], p=2, dim=-1)

        premise_key_pos = []
        for k, v in ep.all_premise_pairs:
            k_enc = tok.encode(k)[0]
            m = [j for j, t in enumerate(ep.prompt_tokens[:-1]) if t == k_enc]
            if m: premise_key_pos.append(m[0])

        r_loss = route_loss_fn(
            adapted_hidden=final_h, logits=final_h,
            query_key_pos=ep.query_key_pos, matching_key_pos=ep.hop1_key_pos,
            distractor_key_positions=premise_key_pos, associated_val_pos=ep.hop1_val_pos,
            target_token=ep.intermediate_token,
        )

        v1_rep = final_h[0, ep.hop1_val_pos : ep.hop1_val_pos + 1]
        v1_norm = F.normalize(v1_rep[0], p=2, dim=-1)
        k2_logits = torch.matmul(v1_norm.unsqueeze(0), h_norm.transpose(0, 1)) / 0.15
        l_k2 = F.cross_entropy(k2_logits, torch.tensor([ep.hop2_key_pos], dtype=torch.long))

        cand_positions, cand_tokens = [], []
        for k, v in ep.all_premise_pairs:
            v_enc = tok.encode(v)[0]
            pos_list = [j for j, t in enumerate(ep.prompt_tokens[:-1]) if t == v_enc]
            if pos_list and pos_list[0] not in cand_positions:
                cand_positions.append(pos_list[0])
                cand_tokens.append(v_enc)
        if not cand_positions:
            cand_positions, cand_tokens = [0], [ep.prompt_tokens[0]]

        cand_states = torch.stack([final_h[0, p] for p in cand_positions], dim=0).unsqueeze(0)
        cand_mask = torch.ones((1, len(cand_positions)), dtype=torch.bool, device=final_h.device)
        tgt_idx = cand_tokens.index(ep.target_token) if ep.target_token in cand_tokens else 0
        val_final_rep = final_h[0, ep.hop2_val_pos : ep.hop2_val_pos + 1]
        bind_logits, _ = candidate.compute_binding_scores(val_final_rep, cand_states, cand_mask)
        l_bind = F.cross_entropy(bind_logits, torch.tensor([tgt_idx], dtype=torch.long))

        loss = r_loss.total_loss + 1.5 * l_k2 + 2.0 * l_bind
        loss.backward()
        torch.nn.utils.clip_grad_norm_(train_params, max_norm=1.0)
        opt.step()

    return candidate


def evaluate_block_candidate_multigroup(
    candidate: TrainableTransformerBlockCandidate,
    name: str,
    trainable_layers: List[int],
    seed: int = 42,
    num_episodes: int = 6,
) -> BlockComparisonMetrics:
    """Evaluates candidate across G1, G2, G3, G4."""
    env = CompositionalAssociativeEnvironment(seed=seed)
    _, _, _, _, g1 = evaluate_candidate_performance(candidate, env, split="train", num_episodes=num_episodes)
    _, _, _, _, g2 = evaluate_candidate_performance(candidate, env, split="disjoint_test", num_episodes=num_episodes)
    _, _, _, _, g3 = evaluate_candidate_performance(candidate, env, split="heldout_composition", num_episodes=num_episodes)
    h1_k, h1_v, h2_k, h2_v, g4 = evaluate_candidate_performance(candidate, env, split="disjoint_test", num_episodes=num_episodes)

    return BlockComparisonMetrics(
        name=name,
        trainable_layers=trainable_layers,
        trainable_params=candidate.trainable_param_count,
        g1_acc=g1,
        g2_acc=g2,
        g3_acc=g3,
        g4_acc=g4,
        h1_key_acc=h1_k,
        h2_key_acc=h2_k,
        h2_val_acc=h2_v,
        summary=f"{name}: Params={candidate.trainable_param_count:,}, G4={g4*100:.1f}%, H1 Key={h1_k*100:.1f}%, H2 Key={h2_k*100:.1f}%",
    )


def run_two_block_controlled_training(base_model: ChakrMicro) -> TwoBlockComparisonReport:
    """Executes Step 349 study comparing 1-block (L3) vs 2-block (L2+L3)."""
    cand_single = train_block_candidate(base_model, trainable_layers=[3], seed=42)
    single_metrics = evaluate_block_candidate_multigroup(cand_single, "Single Block (L3)", [3])

    cand_two = train_block_candidate(base_model, trainable_layers=[2, 3], seed=42)
    two_metrics = evaluate_block_candidate_multigroup(cand_two, "Two Blocks (L2+L3)", [2, 3])

    delta = two_metrics.g4_acc - single_metrics.g4_acc
    improves = (delta > 0.0)

    summary = (
        f"Two-Block Comparison: Single Block G4 = {single_metrics.g4_acc * 100:.1f}% vs "
        f"Two Blocks G4 = {two_metrics.g4_acc * 100:.1f}% (Δ = {delta * 100:+.1f}%). "
        f"Two Blocks Beneficial: {improves}."
    )

    return TwoBlockComparisonReport(
        single_block_metrics=single_metrics,
        two_block_metrics=two_metrics,
        two_block_improves_g4=improves,
        g4_delta=delta,
        summary=summary,
    )
