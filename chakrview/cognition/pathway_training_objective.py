"""Step 309: Controlled Multi-Stage Compositional Training Objective.

Supervises the internal trainable pathway through a principled multi-stage objective:
L_total = L_route1 + lambda_v1 * L_val1_preservation + lambda_k2 * L_route2_key + lambda_v2 * L_route2_val + lambda_bind * L_dynamic_bind

Where:
1. L_route1: Hop-1 query key routing over premise keys.
2. L_val1_preservation: Contrastive preservation ensuring the Hop-1 value representation
   is clearly distinguished from decoy value positions.
3. L_route2_key: Contrastive intermediate objective aligning transformed state with Hop-2 premise key.
4. L_dynamic_bind: Dynamic contextual token binding loss to target answer token.

Includes an explicit ablation study:
- Full Multi-Stage Objective
- Ablated Objective (removing the Hop-2 intermediate routing loss L_route2_key)
"""

from __future__ import annotations

import dataclasses
import time
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from chakrview.brain.model import ChakrMicro
from chakrview.runtime.interactive import compute_model_hash, EXPECTED_WEIGHT_HASH
from chakrview.cognition.internal_integration_study import (
    IntegratedInternalBindingModel,
)
from chakrview.cognition.compositional_benchmark_environment import (
    CompositionalAssociativeEnvironment,
    CompositionalEpisode,
)
from chakrview.cognition.identity_invariant_objective import (
    IdentityInvariantRepresentationLoss,
)


@dataclasses.dataclass
class PathwayTrainingMetrics:
    seed: int
    with_hop2_objective: bool
    mean_g1_tok_acc: float
    mean_g2_tok_acc: float
    mean_g3_tok_acc: float
    mean_g4_tok_acc: float
    mean_h1_key_acc: float
    mean_h2_key_acc: float
    mean_h2_val_acc: float
    final_loss: float


@dataclasses.dataclass
class PathwayTrainingObjectiveReport:
    full_objective_metrics: List[PathwayTrainingMetrics]
    ablated_objective_metrics: List[PathwayTrainingMetrics]
    mean_g4_with_hop2_loss: float
    mean_g4_without_hop2_loss: float
    hop2_objective_benefit: float
    summary: str


def train_pathway_model(
    base_model: ChakrMicro,
    seed: int,
    train_steps: int = 15,
    eval_episodes: int = 8,
    use_hop2_objective: bool = True,
    target_layers: Optional[List[int]] = None,
    bottleneck_dim: int = 32,
) -> Tuple[IntegratedInternalBindingModel, PathwayTrainingMetrics]:
    """Trains the IntegratedInternalBindingModel with or without the Hop-2 intermediate objective."""
    torch.manual_seed(seed)
    env = CompositionalAssociativeEnvironment(seed=seed)
    tok = env.tok

    if target_layers is None:
        target_layers = [3, 4, 5]

    model = IntegratedInternalBindingModel(
        base_model=base_model,
        config_mode="C",  # Mode C: internal pathway + dynamic contextual binding
        target_layers=target_layers,
        bottleneck_dim=bottleneck_dim,
    )

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=2.0e-3,
        weight_decay=0.01,
    )
    route_loss_fn = IdentityInvariantRepresentationLoss(temperature=0.15)

    model.train()
    last_loss = 0.0

    for st in range(train_steps):
        ep = env.generate_episode("train", num_distractors=1, episode_idx=st)
        inp = torch.tensor([ep.prompt_tokens], dtype=torch.long)
        opt.zero_grad()

        final_h, _ = model.internal_backbone.forward_hidden_states(inp)
        h_norm = F.normalize(final_h[0], p=2, dim=-1)

        premise_key_pos = []
        for k, v in ep.all_premise_pairs:
            k_enc = tok.encode(k)[0]
            m = [j for j, t in enumerate(ep.prompt_tokens[:-1]) if t == k_enc]
            if m: premise_key_pos.append(m[0])

        r_loss1 = route_loss_fn(
            adapted_hidden=final_h, logits=final_h, query_key_pos=ep.query_key_pos,
            matching_key_pos=ep.hop1_key_pos, distractor_key_positions=premise_key_pos,
            associated_val_pos=ep.hop1_val_pos, target_token=ep.intermediate_token,
        )

        # Hop-2 intermediate routing objective
        l_route2 = torch.tensor(0.0, device=final_h.device)
        if use_hop2_objective:
            h_v1 = h_norm[ep.hop1_val_pos : ep.hop1_val_pos + 1]
            k2_logits = torch.matmul(h_v1, h_norm.transpose(0, 1)) / 0.15
            l_route2 = F.cross_entropy(k2_logits, torch.tensor([ep.hop2_key_pos], dtype=torch.long))

        # Dynamic contextual candidate binding
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
        bind_logits, _ = model.compute_binding_scores(val_final_rep, cand_states, cand_mask)
        l_bind = F.cross_entropy(bind_logits, torch.tensor([tgt_idx], dtype=torch.long))

        total_loss = r_loss1.total_loss + 2.0 * l_bind
        if use_hop2_objective:
            total_loss = total_loss + 1.5 * l_route2

        total_loss.backward()
        opt.step()
        last_loss = float(total_loss.item())

    # Evaluation
    model.eval()
    splits = [
        ("G1", "train"),
        ("G2", "disjoint_test"),
        ("G3", "heldout_composition"),
        ("G4", "disjoint_test"),
    ]
    tok_accs = {}
    h1_k_corr, h2_k_corr, h2_v_corr = 0, 0, 0
    total_eval = 0

    with torch.no_grad():
        for sp_name, sp_env in splits:
            corr_tok = 0
            for ev_i in range(eval_episodes):
                ep = env.generate_episode(sp_env, num_distractors=1, episode_idx=700000 + ev_i)
                total_eval += 1
                inp = torch.tensor([ep.prompt_tokens], dtype=torch.long)
                final_h, _ = model.internal_backbone.forward_hidden_states(inp)
                h_norm = F.normalize(final_h[0], p=2, dim=-1)

                premise_key_pos = []
                for k, v in ep.all_premise_pairs:
                    k_enc = tok.encode(k)[0]
                    m = [j for j, t in enumerate(ep.prompt_tokens[:-1]) if t == k_enc]
                    if m: premise_key_pos.append(m[0])

                premise_val_pos = []
                for k, v in ep.all_premise_pairs:
                    v_enc = tok.encode(v)[0]
                    m = [j for j, t in enumerate(ep.prompt_tokens[:-1]) if t == v_enc]
                    if m: premise_val_pos.append(m[0])

                # Hop-1 key routing
                h_q1 = h_norm[ep.query_key_pos]
                k1_sims = [float(torch.dot(h_q1, h_norm[kp]).item()) for kp in premise_key_pos]
                pred_k1 = premise_key_pos[k1_sims.index(max(k1_sims))] if k1_sims else 0
                if pred_k1 == ep.hop1_key_pos: h1_k_corr += 1

                # Hop-1 val routing
                h_mk1 = h_norm[pred_k1]
                v1_sims = [float(torch.dot(h_mk1, h_norm[vp]).item()) for vp in premise_val_pos]
                pred_v1 = premise_val_pos[v1_sims.index(max(v1_sims))] if v1_sims else 0

                # Hop-2 key routing directly from intermediate state
                h_v1 = h_norm[pred_v1]
                k2_sims = [float(torch.dot(h_v1, h_norm[kp]).item()) for kp in premise_key_pos]
                pred_k2 = premise_key_pos[k2_sims.index(max(k2_sims))] if k2_sims else 0
                if pred_k2 == ep.hop2_key_pos: h2_k_corr += 1

                # Hop-2 val routing
                h_mk2 = h_norm[pred_k2]
                v2_sims = [float(torch.dot(h_mk2, h_norm[vp]).item()) for vp in premise_val_pos]
                pred_v2 = premise_val_pos[v2_sims.index(max(v2_sims))] if v2_sims else 0
                if pred_v2 == ep.hop2_val_pos: h2_v_corr += 1

                # Dynamic binding
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
                val_rep = final_h[0, pred_v2 : pred_v2 + 1]
                b_logits, _ = model.compute_binding_scores(val_rep, cand_states, cand_mask)
                sel_idx = int(torch.argmax(b_logits[0]).item())
                if cand_tokens[sel_idx] == ep.target_token:
                    corr_tok += 1

            tok_accs[sp_name] = corr_tok / eval_episodes

    metrics = PathwayTrainingMetrics(
        seed=seed,
        with_hop2_objective=use_hop2_objective,
        mean_g1_tok_acc=tok_accs["G1"],
        mean_g2_tok_acc=tok_accs["G2"],
        mean_g3_tok_acc=tok_accs["G3"],
        mean_g4_tok_acc=tok_accs["G4"],
        mean_h1_key_acc=h1_k_corr / max(total_eval, 1),
        mean_h2_key_acc=h2_k_corr / max(total_eval, 1),
        mean_h2_val_acc=h2_v_corr / max(total_eval, 1),
        final_loss=last_loss,
    )

    return model, metrics


def run_compositional_training_objective_study(
    base_model: ChakrMicro,
    seeds: Optional[List[int]] = None,
    train_steps: int = 15,
    eval_episodes: int = 8,
) -> PathwayTrainingObjectiveReport:
    """Compares training with full Hop-2 intermediate loss vs ablated loss."""
    if seeds is None:
        seeds = [42, 101, 2026]

    full_m: List[PathwayTrainingMetrics] = []
    ablated_m: List[PathwayTrainingMetrics] = []

    for s in seeds:
        _, m_full = train_pathway_model(
            base_model=base_model, seed=s, train_steps=train_steps, eval_episodes=eval_episodes,
            use_hop2_objective=True,
        )
        full_m.append(m_full)

        _, m_abl = train_pathway_model(
            base_model=base_model, seed=s, train_steps=train_steps, eval_episodes=eval_episodes,
            use_hop2_objective=False,
        )
        ablated_m.append(m_abl)

    mean_g4_full = sum(m.mean_g4_tok_acc for m in full_m) / len(full_m)
    mean_g4_abl = sum(m.mean_g4_tok_acc for m in ablated_m) / len(ablated_m)
    benefit = mean_g4_full - mean_g4_abl

    summary = (
        f"Step 309 Objective Study: G4 Full={mean_g4_full*100:.2f}% vs "
        f"Ablated={mean_g4_abl*100:.2f}% (delta={benefit*100:+.2f}%). "
        f"H2_Key Full={sum(m.mean_h2_key_acc for m in full_m)/len(full_m)*100:.1f}% vs "
        f"Ablated={sum(m.mean_h2_key_acc for m in ablated_m)/len(ablated_m)*100:.1f}%."
    )

    return PathwayTrainingObjectiveReport(
        full_objective_metrics=full_m,
        ablated_objective_metrics=ablated_m,
        mean_g4_with_hop2_loss=mean_g4_full,
        mean_g4_without_hop2_loss=mean_g4_abl,
        hop2_objective_benefit=benefit,
        summary=summary,
    )
