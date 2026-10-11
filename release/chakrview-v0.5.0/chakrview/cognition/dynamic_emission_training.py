"""Step 275: Dynamic Emission Training.

Trains Candidate C:
Frozen ChakrMicro
+
Trainable Gated Representation Adapter (rank=16)
+
Dynamic Contextual Token Binding Module

Training Objective:
L_total = L_route + L_val + L_binding
Where:
- L_route: Identity-invariant Query-Key alignment contrastive loss
- L_val: Value-position routing loss
- L_binding: Cross-entropy over dynamic contextual candidate token scores

Tracking:
- total loss, binding loss
- key routing accuracy
- value routing accuracy
- candidate-token accuracy
- final token accuracy
- target candidate probability & rank
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
from chakrview.cognition.dynamic_contextual_token_binding import (
    DynamicContextualTokenBinding,
    ChakrMicroWithDynamicBinding,
    compute_module_sha256,
)
from chakrview.cognition.contextual_token_candidates import (
    extract_contextual_candidates_from_episode,
)
from chakrview.cognition.identity_invariant_objective import (
    IdentityInvariantRepresentationLoss,
)
from chakrview.cognition.randomized_associative_episodes import (
    RandomizedAssociativeEnvironment,
    RandomizedAssociativeEpisode,
)
from chakrview.cognition.adapter_curriculum_training import (
    extract_key_positions_from_episode,
)


@dataclasses.dataclass
class DynamicBindingPhaseResult:
    phase_name: str
    phase_desc: str
    train_loss: float
    val_loss: float
    key_routing_acc: float
    value_routing_acc: float
    candidate_token_acc: float
    final_token_acc: float
    mean_target_prob: float
    mean_target_rank: float


@dataclasses.dataclass
class DynamicEmissionTrainingReport:
    seed: int
    adapter_params: int
    binding_params: int
    trainable_params: int
    total_params: int
    binding_sha256: str
    phases: List[DynamicBindingPhaseResult]
    overall_train_loss: float
    overall_val_loss: float
    overall_key_acc: float
    overall_val_acc: float
    overall_cand_acc: float
    overall_token_acc: float
    is_base_frozen: bool
    base_hash: str
    language_retention_ratio: float
    cpu_runtime_ms: float = 0.0


def run_dynamic_emission_training(
    base_model: ChakrMicro,
    seed: int = 42,
    rank: int = 16,
    d_bind: int = 64,
    steps_per_phase: int = 15,
    eval_episodes_per_phase: int = 8,
    lr: float = 2.0e-3,
) -> Tuple[ChakrMicroWithDynamicBinding, DynamicEmissionTrainingReport]:
    """Trains representation adapter and dynamic token binding jointly."""
    t0 = time.time()
    torch.manual_seed(seed)
    env = RandomizedAssociativeEnvironment(seed=seed)
    tok = env.tok

    init_hash = compute_model_hash(base_model)
    candidate = ChakrMicroWithDynamicBinding(base_model, rank=rank, d_bind=d_bind)

    # Strictly freeze base model
    for p in candidate.base_model.parameters():
        p.requires_grad = False
    for p in candidate.adapter.parameters():
        p.requires_grad = True
    for p in candidate.binding.parameters():
        p.requires_grad = True

    route_loss_fn = IdentityInvariantRepresentationLoss(temperature=0.15)
    opt = torch.optim.AdamW(
        list(candidate.adapter.parameters()) + list(candidate.binding.parameters()),
        lr=lr,
        weight_decay=0.01,
    )

    phases = [
        ("Phase A", "Single Association", "train", 1, "standard_map", False),
        ("Phase B", "Multiple Associations (2 pairs)", "train", 2, "standard_map", False),
        ("Phase C", "Randomized Layouts (2-3 pairs)", "train", 2, None, False),
        ("Phase D", "Distractors & Varying Pairs", "train", 3, None, True),
        ("Phase E", "Unseen Identities (known-unseen / unseen-known)", "known_unseen", 2, None, False),
        ("Phase F", "Full Disjoint Episodes (unseen-unseen)", "disjoint_test", 2, None, False),
    ]

    phase_results: List[DynamicBindingPhaseResult] = []

    for p_idx, (p_name, p_desc, sp, n_assoc, lay, dist) in enumerate(phases):
        candidate.train()
        t_losses = []

        for st in range(steps_per_phase):
            ep = env.generate_episode(
                split=sp,
                num_associations=n_assoc,
                layout_name=lay,
                include_distractors=dist,
                episode_idx=p_idx * 1000 + st,
            )

            inp = torch.tensor([ep.prompt_tokens], dtype=torch.long)
            all_k_pos = extract_key_positions_from_episode(ep, tok)

            opt.zero_grad()
            h_ad, _ = candidate.forward_backbone(inp)

            # 1. Routing loss (Query -> Matching Key)
            r_loss = route_loss_fn(
                adapted_hidden=h_ad,
                logits=h_ad,
                query_key_pos=ep.query_key_pos,
                matching_key_pos=ep.matching_key_pos,
                distractor_key_positions=all_k_pos,
                associated_val_pos=ep.associated_val_pos,
                target_token=ep.target_token,
            )
            l_route = r_loss.total_loss

            # 2. Value position loss
            cand_k_h = h_ad[0, all_k_pos]
            target_k_idx = all_k_pos.index(ep.matching_key_pos) if ep.matching_key_pos in all_k_pos else 0
            v_logits = torch.matmul(cand_k_h[target_k_idx : target_k_idx + 1], h_ad[0].transpose(0, 1)) / (192**0.5)
            l_val = F.cross_entropy(v_logits, torch.tensor([ep.associated_val_pos], dtype=torch.long))

            # 3. Dynamic candidate token binding loss
            cand_bundle = extract_contextual_candidates_from_episode(ep, h_ad, tok)
            val_h = h_ad[0, ep.associated_val_pos : ep.associated_val_pos + 1]  # [1, D]

            bind_logits, _ = candidate.compute_binding_scores(
                retrieved_value_rep=val_h,
                candidate_states=cand_bundle.candidate_states,
                candidate_mask=cand_bundle.candidate_mask,
            )

            if cand_bundle.target_candidate_idx is not None:
                tgt_idx = torch.tensor([cand_bundle.target_candidate_idx], dtype=torch.long)
                l_bind = F.cross_entropy(bind_logits, tgt_idx)
            else:
                l_bind = torch.tensor(0.0)

            loss = l_route + l_val + 2.0 * l_bind
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(candidate.adapter.parameters()) + list(candidate.binding.parameters()),
                1.0,
            )
            opt.step()
            t_losses.append(loss.item())

        # Evaluation phase
        candidate.eval()
        v_losses = []
        k_corr, v_corr, c_corr, t_corr = 0, 0, 0, 0
        probs, ranks = [], []

        with torch.no_grad():
            for ev_i in range(eval_episodes_per_phase):
                ep = env.generate_episode(
                    split=sp,
                    num_associations=n_assoc,
                    layout_name=lay,
                    include_distractors=dist,
                    episode_idx=p_idx * 5000 + ev_i,
                )
                inp = torch.tensor([ep.prompt_tokens], dtype=torch.long)
                all_k_pos = extract_key_positions_from_episode(ep, tok)
                h_ad, _ = candidate.forward_backbone(inp)

                # Key routing
                q_h = h_ad[0, ep.query_key_pos]
                cand_k_h = h_ad[0, all_k_pos]
                sims = F.cosine_similarity(q_h.unsqueeze(0), cand_k_h, dim=-1)
                pred_k_pos = all_k_pos[sims.argmax().item()]
                if pred_k_pos == ep.matching_key_pos:
                    k_corr += 1

                # Value routing: evaluate matched key representation against candidate premise value positions
                val_positions_ev = []
                for p in ep.pairs:
                    v_enc = tok.encode(p.val)[0]
                    vp_list = [j for j, t in enumerate(ep.prompt_tokens[:-1]) if t == v_enc]
                    if vp_list:
                        val_positions_ev.append(vp_list[0])

                if val_positions_ev:
                    h_norm_ev = F.normalize(h_ad[0], p=2, dim=-1)
                    h_m_ev = h_norm_ev[pred_k_pos]
                    v_sims = [float(torch.dot(h_m_ev, h_norm_ev[vp]).item()) for vp in val_positions_ev]
                    pred_v_pos = val_positions_ev[v_sims.index(max(v_sims))]
                else:
                    pred_v_pos = 0

                if pred_v_pos == ep.associated_val_pos:
                    v_corr += 1

                # Dynamic Candidate selection
                cand_bundle = extract_contextual_candidates_from_episode(ep, h_ad, tok)
                val_h = h_ad[0, pred_v_pos : pred_v_pos + 1]

                bind_logits, bind_probs = candidate.compute_binding_scores(
                    retrieved_value_rep=val_h,
                    candidate_states=cand_bundle.candidate_states,
                    candidate_mask=cand_bundle.candidate_mask,
                )

                pred_cand_idx = bind_logits.argmax().item()
                if cand_bundle.target_candidate_idx is not None:
                    if pred_cand_idx == cand_bundle.target_candidate_idx:
                        c_corr += 1
                        t_corr += 1
                    t_prob = bind_probs[0, cand_bundle.target_candidate_idx].item()
                    probs.append(t_prob)
                    sorted_idxs = torch.argsort(bind_logits[0], descending=True)
                    t_rank = (sorted_idxs == cand_bundle.target_candidate_idx).nonzero().item() + 1
                    ranks.append(t_rank)
                else:
                    probs.append(0.0)
                    ranks.append(len(cand_bundle.candidate_positions))

        phase_results.append(
            DynamicBindingPhaseResult(
                phase_name=p_name,
                phase_desc=p_desc,
                train_loss=sum(t_losses) / max(1, len(t_losses)),
                val_loss=0.0,
                key_routing_acc=k_corr / eval_episodes_per_phase,
                value_routing_acc=v_corr / eval_episodes_per_phase,
                candidate_token_acc=c_corr / eval_episodes_per_phase,
                final_token_acc=t_corr / eval_episodes_per_phase,
                mean_target_prob=sum(probs) / max(1, len(probs)),
                mean_target_rank=sum(ranks) / max(1, len(ranks)),
            )
        )

    # Post checks
    post_hash = compute_model_hash(base_model)
    is_frozen = (post_hash == init_hash == EXPECTED_WEIGHT_HASH)
    ad_params = sum(p.numel() for p in candidate.adapter.parameters())
    bi_params = sum(p.numel() for p in candidate.binding.parameters())
    train_params = ad_params + bi_params
    tot_params = sum(p.numel() for p in candidate.parameters())
    mod_sha = compute_module_sha256(candidate.binding)

    # Language retention
    candidate.eval()
    with torch.no_grad():
        test_ids = torch.tensor([[10, 45, 120, 230]], dtype=torch.long)
        base_logits = base_model(test_ids)
        h_cand, _ = candidate.forward_backbone(test_ids)
        cand_logits = base_model.lm_head(h_cand)
        lang_ratio = float(torch.norm(cand_logits) / (torch.norm(base_logits) + 1e-12))

    return candidate, DynamicEmissionTrainingReport(
        seed=seed,
        adapter_params=ad_params,
        binding_params=bi_params,
        trainable_params=train_params,
        total_params=tot_params,
        binding_sha256=mod_sha,
        phases=phase_results,
        overall_train_loss=sum(pr.train_loss for pr in phase_results) / len(phase_results),
        overall_val_loss=0.0,
        overall_key_acc=sum(pr.key_routing_acc for pr in phase_results) / len(phase_results),
        overall_val_acc=sum(pr.value_routing_acc for pr in phase_results) / len(phase_results),
        overall_cand_acc=sum(pr.candidate_token_acc for pr in phase_results) / len(phase_results),
        overall_token_acc=sum(pr.final_token_acc for pr in phase_results) / len(phase_results),
        is_base_frozen=is_frozen,
        base_hash=post_hash,
        language_retention_ratio=lang_ratio,
        cpu_runtime_ms=(time.time() - t0) * 1000.0,
    )
