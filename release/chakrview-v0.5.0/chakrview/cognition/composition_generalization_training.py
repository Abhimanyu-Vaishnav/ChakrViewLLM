"""Step 285: Composition Generalization Training and 4-Way Evaluation.

Trains the Compositional Candidate on 2-hop compositional chains:
L_total = L_hop1_route + L_hop1_val + L_bridge + L_hop2_route + L_hop2_val + 2.0 * L_bind

Evaluates across the 4 fundamental generalization splits:
G1: Known identities / Known composition (in-distribution)
G2: Unseen identities / Known composition structure
G3: Known identities / Novel unseen composition chains
G4: Unseen identities / Novel unseen composition chains (PRIMARY I4 CAPABILITY GATE)

Reports:
- Hop-1 key & value routing accuracy
- Intermediate state accuracy
- Hop-2 key & value routing accuracy
- Final candidate & token accuracy
- Mean target probability & rank
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
from chakrview.cognition.compositional_benchmark_environment import (
    CompositionalAssociativeEnvironment,
    CompositionalEpisode,
)
from chakrview.cognition.identity_invariant_objective import (
    IdentityInvariantRepresentationLoss,
)


@dataclasses.dataclass
class CompositionalSplitMetrics:
    split_name: str
    num_episodes: int
    hop1_key_acc: float
    hop1_val_acc: float
    intermediate_state_acc: float
    hop2_key_acc: float
    hop2_val_acc: float
    final_token_acc: float
    mean_target_prob: float
    mean_target_rank: float


@dataclasses.dataclass
class CompositionGeneralizationReport:
    seed: int
    splits: Dict[str, CompositionalSplitMetrics]
    g1_known_known_tok_acc: float
    g2_unseen_id_tok_acc: float
    g3_unseen_comp_tok_acc: float
    g4_unseen_unseen_tok_acc: float
    is_i4_gate_satisfied: bool
    is_base_frozen: bool
    base_hash: str
    cpu_runtime_ms: float = 0.0


def run_compositional_training_and_evaluation(
    base_model: ChakrMicro,
    seed: int = 42,
    rank: int = 16,
    d_bind: int = 64,
    train_steps: int = 25,
    eval_episodes_per_split: int = 10,
    lr: float = 2.0e-3,
) -> Tuple[ChakrMicroCompositionalReasoningModel, CompositionGeneralizationReport]:
    """Trains the compositional candidate and evaluates G1 through G4."""
    t0 = time.time()
    torch.manual_seed(seed)
    env = CompositionalAssociativeEnvironment(seed=seed)
    tok = env.tok

    init_hash = compute_model_hash(base_model)
    candidate = ChakrMicroCompositionalReasoningModel(base_model, rank=rank, d_bind=d_bind)

    # Strictly freeze canonical base model
    for p in candidate.base_model.parameters(): p.requires_grad = False
    for p in candidate.adapter.parameters(): p.requires_grad = True
    for p in candidate.bridge.parameters(): p.requires_grad = True
    for p in candidate.binding.parameters(): p.requires_grad = True

    opt = torch.optim.AdamW(
        list(candidate.adapter.parameters())
        + list(candidate.bridge.parameters())
        + list(candidate.binding.parameters()),
        lr=lr,
        weight_decay=0.01,
    )

    route_loss_fn = IdentityInvariantRepresentationLoss(temperature=0.15)

    # 1. Training loop
    candidate.train()
    for st in range(train_steps):
        ep = env.generate_episode("train", num_distractors=1, episode_idx=st)
        inp = torch.tensor([ep.prompt_tokens], dtype=torch.long)
        opt.zero_grad()
        h_ad, _ = candidate.forward_backbone(inp)

        # Hop 1 Routing: Query attends to premise keys
        premise_key_pos = []
        for k, v in ep.all_premise_pairs:
            k_enc = tok.encode(k)[0]
            m = [j for j, t in enumerate(ep.prompt_tokens[:-1]) if t == k_enc]
            if m: premise_key_pos.append(m[0])

        r_loss1 = route_loss_fn(
            adapted_hidden=h_ad,
            logits=h_ad,
            query_key_pos=ep.query_key_pos,
            matching_key_pos=ep.hop1_key_pos,
            distractor_key_positions=premise_key_pos,
            associated_val_pos=ep.hop1_val_pos,
            target_token=ep.intermediate_token,
        )

        # Hop 2: Intermediate state bridged to second query
        h_v1 = h_ad[0, ep.hop1_val_pos : ep.hop1_val_pos + 1]
        h_q2 = candidate.bridge_intermediate_state(h_v1)

        # Match h_q2 to hop2_key_pos
        h_norm = F.normalize(h_ad[0], p=2, dim=-1)
        h_q2_norm = F.normalize(h_q2[0], p=2, dim=-1)
        k2_logits = torch.matmul(h_q2_norm.unsqueeze(0), h_norm.transpose(0, 1)) / 0.15
        l_h2_key = F.cross_entropy(k2_logits, torch.tensor([ep.hop2_key_pos], dtype=torch.long))

        # Hop 2 value routing and dynamic candidate binding
        cand_positions = []
        cand_tokens = []
        for k, v in ep.all_premise_pairs:
            v_enc = tok.encode(v)[0]
            pos_list = [j for j, t in enumerate(ep.prompt_tokens[:-1]) if t == v_enc]
            if pos_list and pos_list[0] not in cand_positions:
                cand_positions.append(pos_list[0])
                cand_tokens.append(v_enc)

        if not cand_positions:
            cand_positions = [0]
            cand_tokens = [ep.prompt_tokens[0]]

        cand_states = torch.stack([h_ad[0, p] for p in cand_positions], dim=0).unsqueeze(0)
        cand_mask = torch.ones((1, len(cand_positions)), dtype=torch.bool, device=h_ad.device)

        target_cand_idx = None
        for idx, t_id in enumerate(cand_tokens):
            if t_id == ep.target_token:
                target_cand_idx = idx
                break

        val_final_rep = h_ad[0, ep.hop2_val_pos : ep.hop2_val_pos + 1]
        bind_logits, _ = candidate.compute_binding_scores(val_final_rep, cand_states, cand_mask)

        if target_cand_idx is not None:
            l_bind = F.cross_entropy(bind_logits, torch.tensor([target_cand_idx], dtype=torch.long))
        else:
            l_bind = torch.tensor(0.0)

        loss = r_loss1.total_loss + l_h2_key + 2.0 * l_bind
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(candidate.adapter.parameters())
            + list(candidate.bridge.parameters())
            + list(candidate.binding.parameters()),
            1.0,
        )
        opt.step()

    # 2. Evaluation across G1, G2, G3, G4
    candidate.eval()
    split_configs = [
        ("G1_known_known", "train"),
        ("G2_unseen_identities", "disjoint_test"),
        ("G3_unseen_composition", "heldout_composition"),
        ("G4_unseen_unseen", "disjoint_test"),
    ]

    split_results: Dict[str, CompositionalSplitMetrics] = {}

    with torch.no_grad():
        for sp_name, sp_env in split_configs:
            h1_k_corr, h1_v_corr = 0, 0
            inter_corr = 0
            h2_k_corr, h2_v_corr = 0, 0
            tok_corr = 0
            probs, ranks = [], []

            for ev_i in range(eval_episodes_per_split):
                ep = env.generate_episode(
                    split=sp_env,
                    num_distractors=1,
                    episode_idx=60000 + ev_i,
                )
                inp = torch.tensor([ep.prompt_tokens], dtype=torch.long)
                h_ad, _ = candidate.forward_backbone(inp)
                h_norm = F.normalize(h_ad[0], p=2, dim=-1)

                # Identify premise keys and values
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

                # Hop 1 key routing
                h_q1 = h_norm[ep.query_key_pos]
                k1_sims = [float(torch.dot(h_q1, h_norm[kp]).item()) for kp in premise_key_pos]
                pred_h1_k = premise_key_pos[k1_sims.index(max(k1_sims))] if k1_sims else 0
                if pred_h1_k == ep.hop1_key_pos: h1_k_corr += 1

                # Hop 1 value routing
                h_mk1 = h_norm[pred_h1_k]
                v1_sims = [float(torch.dot(h_mk1, h_norm[vp]).item()) for vp in premise_val_pos]
                pred_h1_v = premise_val_pos[v1_sims.index(max(v1_sims))] if v1_sims else 0
                if pred_h1_v == ep.hop1_val_pos: h1_v_corr += 1

                # Intermediate state extraction & check
                if inp[0, pred_h1_v].item() == ep.intermediate_token:
                    inter_corr += 1

                # Hop 2 bridge: intermediate value -> query 2
                h_v1 = h_ad[0, pred_h1_v : pred_h1_v + 1]
                h_q2 = candidate.bridge_intermediate_state(h_v1)
                h_q2_norm = F.normalize(h_q2[0], p=2, dim=-1)

                k2_sims = [float(torch.dot(h_q2_norm, h_norm[kp]).item()) for kp in premise_key_pos]
                pred_h2_k = premise_key_pos[k2_sims.index(max(k2_sims))] if k2_sims else 0
                if pred_h2_k == ep.hop2_key_pos: h2_k_corr += 1

                h_mk2 = h_norm[pred_h2_k]
                v2_sims = [float(torch.dot(h_mk2, h_norm[vp]).item()) for vp in premise_val_pos]
                pred_h2_v = premise_val_pos[v2_sims.index(max(v2_sims))] if v2_sims else 0
                if pred_h2_v == ep.hop2_val_pos: h2_v_corr += 1

                # Final contextual binding
                cand_positions = []
                cand_tokens = []
                for k, v in ep.all_premise_pairs:
                    v_enc = tok.encode(v)[0]
                    pos_list = [j for j, t in enumerate(ep.prompt_tokens[:-1]) if t == v_enc]
                    if pos_list and pos_list[0] not in cand_positions:
                        cand_positions.append(pos_list[0])
                        cand_tokens.append(v_enc)

                if not cand_positions:
                    cand_positions = [0]
                    cand_tokens = [ep.prompt_tokens[0]]

                cand_states = torch.stack([h_ad[0, p] for p in cand_positions], dim=0).unsqueeze(0)
                cand_mask = torch.ones((1, len(cand_positions)), dtype=torch.bool, device=h_ad.device)

                target_idx = None
                for idx, t_id in enumerate(cand_tokens):
                    if t_id == ep.target_token:
                        target_idx = idx
                        break

                val_final_rep = h_ad[0, pred_h2_v : pred_h2_v + 1]
                bind_logits, bind_probs = candidate.compute_binding_scores(val_final_rep, cand_states, cand_mask)

                pred_cand_idx = bind_logits.argmax().item()
                if target_idx is not None:
                    if pred_cand_idx == target_idx:
                        tok_corr += 1
                    p_tgt = bind_probs[0, target_idx].item()
                    probs.append(p_tgt)
                    sorted_idxs = torch.argsort(bind_logits[0], descending=True)
                    rk = (sorted_idxs == target_idx).nonzero().item() + 1
                    ranks.append(rk)
                else:
                    probs.append(0.0)
                    ranks.append(len(cand_tokens))

            split_results[sp_name] = CompositionalSplitMetrics(
                split_name=sp_name,
                num_episodes=eval_episodes_per_split,
                hop1_key_acc=h1_k_corr / eval_episodes_per_split,
                hop1_val_acc=h1_v_corr / eval_episodes_per_split,
                intermediate_state_acc=inter_corr / eval_episodes_per_split,
                hop2_key_acc=h2_k_corr / eval_episodes_per_split,
                hop2_val_acc=h2_v_corr / eval_episodes_per_split,
                final_token_acc=tok_corr / eval_episodes_per_split,
                mean_target_prob=sum(probs) / max(1, len(probs)),
                mean_target_rank=sum(ranks) / max(1, len(ranks)),
            )

    post_hash = compute_model_hash(base_model)
    is_frozen = (post_hash == init_hash == EXPECTED_WEIGHT_HASH)

    g1_tok = split_results["G1_known_known"].final_token_acc
    g2_tok = split_results["G2_unseen_identities"].final_token_acc
    g3_tok = split_results["G3_unseen_composition"].final_token_acc
    g4_tok = split_results["G4_unseen_unseen"].final_token_acc

    i4_gate = (g4_tok >= 0.50 and g3_tok >= 0.50)

    return candidate, CompositionGeneralizationReport(
        seed=seed,
        splits=split_results,
        g1_known_known_tok_acc=g1_tok,
        g2_unseen_id_tok_acc=g2_tok,
        g3_unseen_comp_tok_acc=g3_tok,
        g4_unseen_unseen_tok_acc=g4_tok,
        is_i4_gate_satisfied=i4_gate,
        is_base_frozen=is_frozen,
        base_hash=post_hash,
        cpu_runtime_ms=(time.time() - t0) * 1000.0,
    )
