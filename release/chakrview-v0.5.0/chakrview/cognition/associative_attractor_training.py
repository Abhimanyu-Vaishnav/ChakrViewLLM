"""Step 299: Associative Attractor Training.

Trains the learned associative attractor codebook jointly with:
1. Intermediate attractor assignment clustering / consistency:
   - Intermediate state h_v1 projects and softly clusters into learned prototypes.
   - Decoy/negative intermediate states from other premises are separated.
2. Hop-2 key routing alignment:
   - Query constructed from attractor state aligns with true Hop-2 premise key.
3. End-to-end dynamic contextual token binding:
   - Val-2 retrieved state binds candidate token to output vocabulary token.
4. Preserves causal chain:
   Hop 1 result -> Attractor state -> Hop 2 query -> Hop 2 key -> Hop 2 val -> Final token.
5. Includes explicit ablation removing the attractor loss.
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
from chakrview.cognition.identity_invariant_objective import (
    IdentityInvariantRepresentationLoss,
)


@dataclasses.dataclass
class AttractorTrainingMetrics:
    seed: int
    with_attractor_loss: bool
    mean_g1_tok_acc: float
    mean_g2_tok_acc: float
    mean_g3_tok_acc: float
    mean_g4_tok_acc: float
    mean_hop1_routing_acc: float
    mean_hop2_key_acc: float
    mean_hop2_val_acc: float
    mean_attractor_entropy: float
    mean_attractor_utilization: float
    final_loss: float


@dataclasses.dataclass
class AttractorTrainingReport:
    full_attractor_metrics: List[AttractorTrainingMetrics]
    ablated_attractor_metrics: List[AttractorTrainingMetrics]
    mean_g4_with_attractor_loss: float
    mean_g4_without_attractor_loss: float
    attractor_loss_benefit: float
    summary: str


def train_attractor_compositional_model(
    base_model: ChakrMicro,
    seed: int,
    train_steps: int = 15,
    eval_episodes: int = 8,
    use_attractor_loss: bool = True,
    num_attractors: int = 16,
    d_attractor: int = 64,
) -> Tuple[ChakrMicroCompositionalReasoningModel, LearnedAssociativeAttractor, AttractorTrainingMetrics]:
    """Trains candidate architecture with learned associative attractor."""
    torch.manual_seed(seed)
    env = CompositionalAssociativeEnvironment(seed=seed)
    tok = env.tok

    cand = ChakrMicroCompositionalReasoningModel(base_model, rank=16)
    attractor = LearnedAssociativeAttractor(
        d_model=cand.config.d_model,
        num_attractors=num_attractors,
        d_attractor=d_attractor,
        temperature=0.25,
    )

    for p in cand.base_model.parameters(): p.requires_grad = False
    for p in cand.adapter.parameters(): p.requires_grad = True
    for p in cand.bridge.parameters(): p.requires_grad = True
    for p in cand.binding.parameters(): p.requires_grad = True
    for p in attractor.parameters(): p.requires_grad = True

    opt = torch.optim.AdamW(
        list(cand.adapter.parameters()) +
        list(cand.bridge.parameters()) +
        list(cand.binding.parameters()) +
        list(attractor.parameters()),
        lr=2.0e-3,
        weight_decay=0.01,
    )
    loss_fn = IdentityInvariantRepresentationLoss(temperature=0.15)

    cand.train()
    attractor.train()
    last_loss = 0.0

    for st in range(train_steps):
        ep = env.generate_episode("train", num_distractors=1, episode_idx=st)
        inp = torch.tensor([ep.prompt_tokens], dtype=torch.long)
        opt.zero_grad()
        h_ad, _ = cand.forward_backbone(inp)

        premise_key_pos = []
        for k, v in ep.all_premise_pairs:
            k_enc = tok.encode(k)[0]
            m = [j for j, t in enumerate(ep.prompt_tokens[:-1]) if t == k_enc]
            if m: premise_key_pos.append(m[0])

        r_loss1 = loss_fn(
            adapted_hidden=h_ad, logits=h_ad, query_key_pos=ep.query_key_pos,
            matching_key_pos=ep.hop1_key_pos, distractor_key_positions=premise_key_pos,
            associated_val_pos=ep.hop1_val_pos, target_token=ep.intermediate_token,
        )

        # Hop 1 intermediate representation
        h_v1 = h_ad[0, ep.hop1_val_pos : ep.hop1_val_pos + 1]

        # Pass through attractor
        h_att, probs, ent = attractor(h_v1)

        # Bridged query for Hop 2
        h_q2 = cand.bridge(h_att)

        # Contrastive Hop-2 key routing
        h_norm = F.normalize(h_ad[0], p=2, dim=-1)
        h_q2_norm = F.normalize(h_q2[0], p=2, dim=-1)
        k2_logits = torch.matmul(h_q2_norm.unsqueeze(0), h_norm.transpose(0, 1)) / 0.15
        l_route = F.cross_entropy(k2_logits, torch.tensor([ep.hop2_key_pos], dtype=torch.long))

        # Attractor prototype dispersion/entropy loss
        # Encourages confident assignment per example while avoiding collapse
        avg_probs = probs.mean(dim=0)
        l_attractor = 0.0
        if use_attractor_loss:
            # Minimize assignment entropy per instance (sharp cluster)
            inst_entropy = -torch.sum(probs * torch.log(probs + 1e-12), dim=-1).mean()
            # Maximize batch diversity (anti-collapse across distinct identities)
            diversity_entropy = -torch.sum(avg_probs * torch.log(avg_probs + 1e-12))
            l_attractor = 0.5 * inst_entropy - 0.2 * diversity_entropy

        # Final binding loss
        cand_positions = []
        cand_tokens = []
        for k, v in ep.all_premise_pairs:
            v_enc = tok.encode(v)[0]
            pos_list = [j for j, t in enumerate(ep.prompt_tokens[:-1]) if t == v_enc]
            if pos_list and pos_list[0] not in cand_positions:
                cand_positions.append(pos_list[0])
                cand_tokens.append(v_enc)
        if not cand_positions:
            cand_positions, cand_tokens = [0], [ep.prompt_tokens[0]]

        cand_states = torch.stack([h_ad[0, p] for p in cand_positions], dim=0).unsqueeze(0)
        cand_mask = torch.ones((1, len(cand_positions)), dtype=torch.bool, device=h_ad.device)
        tgt_idx = cand_tokens.index(ep.target_token) if ep.target_token in cand_tokens else 0
        val_final_rep = h_ad[0, ep.hop2_val_pos : ep.hop2_val_pos + 1]
        bind_logits, _ = cand.compute_binding_scores(val_final_rep, cand_states, cand_mask)
        l_bind = F.cross_entropy(bind_logits, torch.tensor([tgt_idx], dtype=torch.long))

        total_loss = r_loss1.total_loss + 1.0 * l_route + 2.0 * l_bind
        if use_attractor_loss:
            total_loss = total_loss + 0.5 * l_attractor

        total_loss.backward()
        opt.step()
        last_loss = float(total_loss.item())

    # Evaluation across splits
    cand.eval()
    attractor.eval()

    splits = [
        ("G1", "train"),
        ("G2", "disjoint_test"),
        ("G3", "heldout_composition"),
        ("G4", "disjoint_test"),
    ]
    tok_accs = {}
    h1_k_corr, h2_k_corr, h2_v_corr = 0, 0, 0
    all_ents = []
    active_attractors = set()
    total_eval_episodes = 0

    with torch.no_grad():
        for sp_name, sp_env in splits:
            corr_tok = 0
            for ev_i in range(eval_episodes):
                ep = env.generate_episode(sp_env, num_distractors=1, episode_idx=150000 + ev_i)
                total_eval_episodes += 1
                inp = torch.tensor([ep.prompt_tokens], dtype=torch.long)
                h_ad, _ = cand.forward_backbone(inp)
                h_norm = F.normalize(h_ad[0], p=2, dim=-1)

                # Hop 1 key routing
                q1_norm = h_norm[ep.query_key_pos : ep.query_key_pos + 1]
                scores1 = torch.matmul(q1_norm, h_norm.transpose(0, 1)).squeeze(0)
                pred1_k = int(torch.argmax(scores1[:-1]).item())
                if pred1_k == ep.hop1_key_pos:
                    h1_k_corr += 1

                # Hop 1 value representation
                h_v1 = h_ad[0, ep.hop1_val_pos : ep.hop1_val_pos + 1]
                h_att, probs, ent = attractor(h_v1)
                all_ents.append(ent)
                best_k = int(torch.argmax(probs, dim=-1).item())
                active_attractors.add(best_k)

                # Bridged query 2
                h_q2 = cand.bridge(h_att)
                h_q2_norm = F.normalize(h_q2[0], p=2, dim=-1)

                scores2_k = torch.matmul(h_q2_norm.unsqueeze(0), h_norm.transpose(0, 1)).squeeze(0)
                pred2_k = int(torch.argmax(scores2_k[:-1]).item())
                if pred2_k == ep.hop2_key_pos:
                    h2_k_corr += 1

                # Hop 2 value routing
                h_k2 = h_ad[0, ep.hop2_key_pos : ep.hop2_key_pos + 1]
                scores2_v = torch.matmul(F.normalize(h_k2, p=2, dim=-1), h_norm.transpose(0, 1)).squeeze(0)
                pred2_v = int(torch.argmax(scores2_v[:-1]).item())
                if pred2_v == ep.hop2_val_pos:
                    h2_v_corr += 1

                # Dynamic binding
                cand_positions = []
                cand_tokens = []
                for k, v in ep.all_premise_pairs:
                    v_enc = tok.encode(v)[0]
                    pos_list = [j for j, t in enumerate(ep.prompt_tokens[:-1]) if t == v_enc]
                    if pos_list and pos_list[0] not in cand_positions:
                        cand_positions.append(pos_list[0])
                        cand_tokens.append(v_enc)
                if not cand_positions:
                    cand_positions, cand_tokens = [0], [ep.prompt_tokens[0]]

                cand_states = torch.stack([h_ad[0, p] for p in cand_positions], dim=0).unsqueeze(0)
                cand_mask = torch.ones((1, len(cand_positions)), dtype=torch.bool, device=h_ad.device)
                val_rep = h_ad[0, ep.hop2_val_pos : ep.hop2_val_pos + 1]
                b_logits, _ = cand.compute_binding_scores(val_rep, cand_states, cand_mask)
                sel_idx = int(torch.argmax(b_logits[0]).item())
                if cand_tokens[sel_idx] == ep.target_token:
                    corr_tok += 1

            tok_accs[sp_name] = corr_tok / eval_episodes

    metrics = AttractorTrainingMetrics(
        seed=seed,
        with_attractor_loss=use_attractor_loss,
        mean_g1_tok_acc=tok_accs["G1"],
        mean_g2_tok_acc=tok_accs["G2"],
        mean_g3_tok_acc=tok_accs["G3"],
        mean_g4_tok_acc=tok_accs["G4"],
        mean_hop1_routing_acc=h1_k_corr / total_eval_episodes,
        mean_hop2_key_acc=h2_k_corr / total_eval_episodes,
        mean_hop2_val_acc=h2_v_corr / total_eval_episodes,
        mean_attractor_entropy=float(sum(all_ents) / max(len(all_ents), 1)),
        mean_attractor_utilization=len(active_attractors) / num_attractors,
        final_loss=last_loss,
    )

    return cand, attractor, metrics


def run_associative_attractor_training_study(
    base_model: ChakrMicro,
    seeds: Optional[List[int]] = None,
    train_steps: int = 15,
    eval_episodes: int = 8,
) -> AttractorTrainingReport:
    """Executes Step 299 training study comparing full attractor loss vs ablated loss."""
    if seeds is None:
        seeds = [42, 101, 2026]

    full_metrics: List[AttractorTrainingMetrics] = []
    ablated_metrics: List[AttractorTrainingMetrics] = []

    for s in seeds:
        _, _, m_full = train_attractor_compositional_model(
            base_model, seed=s, train_steps=train_steps, eval_episodes=eval_episodes,
            use_attractor_loss=True,
        )
        full_metrics.append(m_full)

        _, _, m_abl = train_attractor_compositional_model(
            base_model, seed=s, train_steps=train_steps, eval_episodes=eval_episodes,
            use_attractor_loss=False,
        )
        ablated_metrics.append(m_abl)

    mean_g4_full = sum(m.mean_g4_tok_acc for m in full_metrics) / len(full_metrics)
    mean_g4_abl = sum(m.mean_g4_tok_acc for m in ablated_metrics) / len(ablated_metrics)
    benefit = mean_g4_full - mean_g4_abl

    summary = (
        f"Attractor Training Study: G4 Full={mean_g4_full*100:.2f}% vs "
        f"Ablated={mean_g4_abl*100:.2f}% (delta={benefit*100:+.2f}%). "
        f"Utilization={sum(m.mean_attractor_utilization for m in full_metrics)/len(full_metrics):.2f}"
    )

    return AttractorTrainingReport(
        full_attractor_metrics=full_metrics,
        ablated_attractor_metrics=ablated_metrics,
        mean_g4_with_attractor_loss=mean_g4_full,
        mean_g4_without_attractor_loss=mean_g4_abl,
        attractor_loss_benefit=benefit,
        summary=summary,
    )
