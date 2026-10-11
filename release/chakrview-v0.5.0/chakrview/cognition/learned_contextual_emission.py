"""Step 266: Learned Contextual Emission Training.

Trains Candidate Architecture:
Frozen ChakrMicro Backbone
+
Trainable Gated Representation Adapter (rank=16)
+
Trainable Contextual Vocabulary Readout

Training Objective:
L_total = L_route + L_val + L_emission
Where:
- L_route: Identity-invariant Query-Key role alignment contrastive loss
- L_val: Key->Value role association loss
- L_emission: Cross-entropy of contextual readout logits against ground-truth episode target token

Tracks:
- Train loss & validation loss
- Key-position accuracy
- Value-position accuracy
- Final token accuracy
- Target token probability
- Target token rank
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
from chakrview.cognition.contextual_vocabulary_readout import (
    ContextualVocabularyReadout,
    ChakrMicroWithContextualReadout,
    compute_module_sha256,
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
class EmissionCurriculumPhaseResult:
    phase_name: str
    phase_desc: str
    train_loss: float
    val_loss: float
    key_position_accuracy: float
    value_position_accuracy: float
    final_token_accuracy: float
    mean_target_prob: float
    mean_target_rank: float


@dataclasses.dataclass
class ContextualEmissionTrainingReport:
    seed: int
    adapter_params: int
    readout_params: int
    trainable_params: int
    total_params: int
    readout_sha256: str
    phases: List[EmissionCurriculumPhaseResult]
    overall_train_loss: float
    overall_val_loss: float
    overall_key_acc: float
    overall_val_pos_acc: float
    overall_token_acc: float
    is_base_frozen: bool
    base_hash: str
    language_retention_ratio: float
    cpu_runtime_ms: float = 0.0


def run_contextual_emission_training(
    base_model: ChakrMicro,
    seed: int = 42,
    rank: int = 16,
    steps_per_phase: int = 20,
    eval_episodes_per_phase: int = 10,
    lr: float = 1.5e-3,
) -> Tuple[ChakrMicroWithContextualReadout, ContextualEmissionTrainingReport]:
    """Trains representation adapter and contextual readout jointly on multi-phase curriculum."""
    t0 = time.time()
    torch.manual_seed(seed)
    env = RandomizedAssociativeEnvironment(seed=seed)
    tok = env.tok

    init_hash = compute_model_hash(base_model)
    candidate = ChakrMicroWithContextualReadout(base_model, rank=rank)

    # Strictly freeze canonical base model
    for p in candidate.base_model.parameters():
        p.requires_grad = False
    for p in candidate.adapter.parameters():
        p.requires_grad = True
    for p in candidate.readout.parameters():
        p.requires_grad = True

    opt = torch.optim.AdamW(
        list(candidate.adapter.parameters()) + list(candidate.readout.parameters()),
        lr=lr,
        weight_decay=1e-4,
    )
    role_loss_fn = IdentityInvariantRepresentationLoss()

    # Base language loss reference
    ref_enc = tok.encode("The quick brown fox jumps over the lazy dog.")
    ref_inp = torch.tensor([ref_enc[:-1]], dtype=torch.long)
    ref_tgt = torch.tensor([ref_enc[1:]], dtype=torch.long)
    with torch.no_grad():
        x = base_model.embedding(ref_inp)
        for l in base_model.layers:
            x = l(x)
        h = base_model.final_norm(x)
        b_log = base_model.lm_head(h)
        init_lang_loss = float(F.cross_entropy(b_log.view(-1, 4096), ref_tgt.view(-1)).item())

    phase_configs = [
        {"name": "Phase_A", "desc": "Single association", "split": "train", "n": 1, "dist": False, "lay": "standard_map"},
        {"name": "Phase_B", "desc": "Multiple associations (2 pairs)", "split": "train", "n": 2, "dist": False, "lay": "standard_map"},
        {"name": "Phase_C", "desc": "Multiple associations + distractors", "split": "train", "n": 2, "dist": True, "lay": "standard_map"},
        {"name": "Phase_D", "desc": "Randomized layouts", "split": "train", "n": 2, "dist": False, "lay": None},
        {"name": "Phase_E", "desc": "Unseen identities (partial)", "split": "known_unseen", "n": 2, "dist": False, "lay": "standard_map"},
        {"name": "Phase_F", "desc": "Unseen key + unseen value", "split": "disjoint_test", "n": 2, "dist": False, "lay": "standard_map"},
    ]

    phase_results: List[EmissionCurriculumPhaseResult] = []

    for cfg in phase_configs:
        candidate.train()
        train_loss_acc = 0.0

        for st in range(steps_per_phase):
            ep = env.generate_episode(
                split=cfg["split"] if cfg["split"] != "disjoint_test" else "train",
                num_associations=cfg["n"],
                layout_name=cfg["lay"],
                include_distractors=cfg["dist"],
                episode_idx=st,
            )

            inp = torch.tensor([ep.prompt_tokens], dtype=torch.long)
            all_k_pos = extract_key_positions_from_episode(ep, tok)

            opt.zero_grad()
            adapted_hidden, raw_hidden = candidate.forward_backbone(inp)

            # 1. Representation role loss
            r_loss = role_loss_fn(
                adapted_hidden=adapted_hidden,
                logits=raw_hidden, # pass hidden for dummy
                query_key_pos=ep.query_key_pos,
                matching_key_pos=ep.matching_key_pos,
                distractor_key_positions=all_k_pos,
                associated_val_pos=ep.associated_val_pos,
                target_token=ep.target_token,
            )

            # 2. Contextual vocabulary emission loss
            # Soft selected value representation from routing:
            h_norm = F.normalize(adapted_hidden[0], p=2, dim=-1)
            h_q = h_norm[ep.query_key_pos]
            k_sims = torch.stack([torch.dot(h_q, h_norm[kp]) for kp in all_k_pos])
            k_attn = F.softmax(k_sims / 0.1, dim=-1)
            h_matched_k = sum(k_attn[i] * adapted_hidden[0, kp] for i, kp in enumerate(all_k_pos))

            # Matched key attends to candidate values in context
            val_positions = []
            for p in ep.pairs:
                v_enc = tok.encode(p.val)[0]
                vp_list = [j for j, t in enumerate(ep.prompt_tokens[:-1]) if t == v_enc]
                if vp_list:
                    val_positions.append(vp_list[0])

            h_mk_norm = F.normalize(h_matched_k, p=2, dim=-1)
            v_sims = torch.stack([torch.dot(h_mk_norm, h_norm[vp]) for vp in val_positions])
            v_attn = F.softmax(v_sims / 0.1, dim=-1)
            selected_v_rep = sum(v_attn[i] * adapted_hidden[0, vp] for i, vp in enumerate(val_positions))

            # Forward through Contextual Vocabulary Readout
            emission_logits = candidate.forward_from_value_rep(selected_v_rep.unsqueeze(0))
            tgt_t = torch.tensor([ep.target_token], dtype=torch.long)
            l_emission = F.cross_entropy(emission_logits, tgt_t)

            loss = r_loss.total_loss + l_emission
            loss.backward()
            opt.step()

            train_loss_acc += loss.item()

        avg_train_loss = train_loss_acc / max(1, steps_per_phase)

        # Validation on distinct split
        candidate.eval()
        val_loss_acc = 0.0
        val_k_corr = 0
        val_p_corr = 0
        val_t_corr = 0
        val_probs = []
        val_ranks = []

        with torch.no_grad():
            for e_idx in range(eval_episodes_per_phase):
                ep_val = env.generate_episode(
                    split=cfg["split"],
                    num_associations=cfg["n"],
                    layout_name=cfg["lay"],
                    include_distractors=cfg["dist"],
                    episode_idx=1000 + e_idx,
                )

                inp_v = torch.tensor([ep_val.prompt_tokens], dtype=torch.long)
                all_k_pos_v = extract_key_positions_from_episode(ep_val, tok)
                adapted_hidden_v, _ = candidate.forward_backbone(inp_v)

                h_norm_v = F.normalize(adapted_hidden_v[0], p=2, dim=-1)
                h_qv = h_norm_v[ep_val.query_key_pos]
                k_sims_v = [float(torch.dot(h_qv, h_norm_v[kp]).item()) for kp in all_k_pos_v]
                if k_sims_v and all_k_pos_v[k_sims_v.index(max(k_sims_v))] == ep_val.matching_key_pos:
                    val_k_corr += 1

                val_positions_v = []
                for p in ep_val.pairs:
                    v_enc = tok.encode(p.val)[0]
                    vp_list = [j for j, t in enumerate(ep_val.prompt_tokens[:-1]) if t == v_enc]
                    if vp_list:
                        val_positions_v.append(vp_list[0])

                h_mv = h_norm_v[ep_val.matching_key_pos]
                v_sims_v = [float(torch.dot(h_mv, h_norm_v[vp]).item()) for vp in val_positions_v]
                best_vp = val_positions_v[v_sims_v.index(max(v_sims_v))]
                if best_vp == ep_val.associated_val_pos:
                    val_p_corr += 1

                # Forward through readout from selected value state
                selected_v_state = adapted_hidden_v[0, best_vp].unsqueeze(0)
                em_logits = candidate.forward_from_value_rep(selected_v_state)
                tgt_v_tok = ep_val.target_token
                l_v_em = F.cross_entropy(em_logits, torch.tensor([tgt_v_tok], dtype=torch.long))
                val_loss_acc += l_v_em.item()

                pred_tok = torch.argmax(em_logits[0]).item()
                if pred_tok == tgt_v_tok:
                    val_t_corr += 1

                em_probs = F.softmax(em_logits[0], dim=-1)
                val_probs.append(float(em_probs[tgt_v_tok].item()))
                rk = int((em_logits[0] > em_logits[0, tgt_v_tok]).sum().item()) + 1
                val_ranks.append(rk)

        k_acc = val_k_corr / max(1, eval_episodes_per_phase)
        v_acc = val_p_corr / max(1, eval_episodes_per_phase)
        t_acc = val_t_corr / max(1, eval_episodes_per_phase)
        avg_val_loss = val_loss_acc / max(1, eval_episodes_per_phase)

        phase_results.append(
            EmissionCurriculumPhaseResult(
                phase_name=cfg["name"],
                phase_desc=cfg["desc"],
                train_loss=avg_train_loss,
                val_loss=avg_val_loss,
                key_position_accuracy=k_acc,
                value_position_accuracy=v_acc,
                final_token_accuracy=t_acc,
                mean_target_prob=sum(val_probs) / max(1, len(val_probs)),
                mean_target_rank=sum(val_ranks) / max(1, len(val_ranks)),
            )
        )

    # Base language retention
    with torch.no_grad():
        x = base_model.embedding(ref_inp)
        for l in base_model.layers:
            x = l(x)
        h = base_model.final_norm(x)
        post_log = base_model.lm_head(h)
        post_lang_loss = float(F.cross_entropy(post_log.view(-1, 4096), ref_tgt.view(-1)).item())

    retention_ratio = post_lang_loss / max(1e-12, init_lang_loss)
    final_hash = compute_model_hash(base_model)
    is_frozen = (final_hash == init_hash == EXPECTED_WEIGHT_HASH)

    p_ad = sum(p.numel() for p in candidate.adapter.parameters())
    p_ro = sum(p.numel() for p in candidate.readout.parameters())
    p_tr = p_ad + p_ro
    p_tot = sum(p.numel() for p in candidate.parameters())
    ro_sha = compute_module_sha256(candidate.readout)
    elapsed_ms = (time.time() - t0) * 1000.0

    report = ContextualEmissionTrainingReport(
        seed=seed,
        adapter_params=p_ad,
        readout_params=p_ro,
        trainable_params=p_tr,
        total_params=p_tot,
        readout_sha256=ro_sha,
        phases=phase_results,
        overall_train_loss=sum(p.train_loss for p in phase_results) / len(phase_results),
        overall_val_loss=sum(p.val_loss for p in phase_results) / len(phase_results),
        overall_key_acc=sum(p.key_position_accuracy for p in phase_results) / len(phase_results),
        overall_val_pos_acc=sum(p.value_position_accuracy for p in phase_results) / len(phase_results),
        overall_token_acc=sum(p.final_token_accuracy for p in phase_results) / len(phase_results),
        is_base_frozen=is_frozen,
        base_hash=final_hash,
        language_retention_ratio=retention_ratio,
        cpu_runtime_ms=elapsed_ms,
    )

    return candidate, report
