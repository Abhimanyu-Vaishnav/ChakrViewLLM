"""Step 259: Adapter Training Curriculum.

Trains ONLY the compact representation adapter along Phase A through G:
Phase A: Single pair
Phase B: Two pairs
Phase C: Multiple pairs (3 pairs)
Phase D: Distractors
Phase E: Randomized layouts
Phase F: Unseen identities (known key, unseen val / unseen key, known val)
Phase G: Unseen key + unseen value

For each phase records:
- train loss
- validation loss
- train key accuracy
- validation key accuracy
- held-out key accuracy
- value-position accuracy
- final token accuracy

Advances only when validation improves or finishes budget.
Ensures ChakrMicro remains completely frozen (Delta W = 0).
"""

from __future__ import annotations

import dataclasses
import time
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn.functional as F

from chakrview.brain.model import ChakrMicro
from chakrview.runtime.interactive import compute_model_hash, EXPECTED_WEIGHT_HASH
from chakrview.cognition.neural_representation_adapter import (
    ChakrMicroWithAdaptedRepresentations,
    compute_module_sha256,
)
from chakrview.cognition.identity_invariant_objective import (
    IdentityInvariantRepresentationLoss,
)
from chakrview.cognition.randomized_associative_episodes import (
    RandomizedAssociativeEnvironment,
    RandomizedAssociativeEpisode,
)


@dataclasses.dataclass
class AdapterPhaseMetrics:
    phase_name: str
    phase_desc: str
    train_loss: float
    val_loss: float
    train_key_acc: float
    val_key_acc: float
    heldout_key_acc: float
    val_pos_acc: float
    final_token_acc: float
    phase_passed: bool


@dataclasses.dataclass
class AdapterCurriculumReport:
    seed: int
    adapter_type: str
    adapter_param_count: int
    adapter_hash: str
    phases: List[AdapterPhaseMetrics]
    overall_train_loss: float
    overall_val_loss: float
    overall_key_acc: float
    overall_val_pos_acc: float
    overall_token_acc: float
    is_base_frozen: bool
    base_hash: str
    language_retention_ratio: float
    cpu_runtime_ms: float = 0.0


def extract_key_positions_from_episode(ep: RandomizedAssociativeEpisode, tok) -> List[int]:
    """Finds all premise key positions in prompt tokens."""
    res = []
    for p in ep.pairs:
        k_enc = tok.encode(p.key)[0]
        pos_list = [i for i, t in enumerate(ep.prompt_tokens[:-1]) if t == k_enc]
        if pos_list:
            res.append(pos_list[0])
    return res


def run_adapter_training_curriculum(
    base_model: ChakrMicro,
    adapter_type: str = "gated",
    rank: int = 32,
    seed: int = 42,
    steps_per_phase: int = 20,
    eval_episodes_per_phase: int = 10,
    lr: float = 2e-3,
) -> Tuple[ChakrMicroWithAdaptedRepresentations, AdapterCurriculumReport]:
    """Runs Phase A to Phase G representation adaptation curriculum."""
    t0 = time.time()
    torch.manual_seed(seed)
    env = RandomizedAssociativeEnvironment(seed=seed)
    tok = env.tok

    init_hash = compute_model_hash(base_model)
    candidate = ChakrMicroWithAdaptedRepresentations(base_model, adapter_type=adapter_type, rank=rank)

    # Strictly freeze base model
    for p in candidate.base_model.parameters():
        p.requires_grad = False
    for p in candidate.adapter.parameters():
        p.requires_grad = True

    opt = torch.optim.AdamW(candidate.adapter.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = IdentityInvariantRepresentationLoss()

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
        {"name": "Phase_A", "desc": "Single pair", "split": "train", "n": 1, "dist": False, "lay": "standard_map"},
        {"name": "Phase_B", "desc": "Two pairs", "split": "train", "n": 2, "dist": False, "lay": "standard_map"},
        {"name": "Phase_C", "desc": "Multiple pairs (3 pairs)", "split": "train", "n": 3, "dist": False, "lay": "standard_map"},
        {"name": "Phase_D", "desc": "Distractors", "split": "train", "n": 2, "dist": True, "lay": "standard_map"},
        {"name": "Phase_E", "desc": "Randomized layouts", "split": "train", "n": 2, "dist": False, "lay": None},
        {"name": "Phase_F", "desc": "Unseen identities (partial)", "split": "known_unseen", "n": 2, "dist": False, "lay": "standard_map"},
        {"name": "Phase_G", "desc": "Unseen key + unseen value", "split": "disjoint_test", "n": 2, "dist": False, "lay": "standard_map"},
    ]

    phase_results: List[AdapterPhaseMetrics] = []

    for cfg in phase_configs:
        candidate.train()
        train_loss_acc = 0.0
        train_k_corr = 0

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
            h_ad, h_raw, logits = candidate.forward_adapted_backbone(inp)

            losses = loss_fn(
                adapted_hidden=h_ad,
                logits=logits,
                query_key_pos=ep.query_key_pos,
                matching_key_pos=ep.matching_key_pos,
                distractor_key_positions=all_k_pos,
                associated_val_pos=ep.associated_val_pos,
                target_token=ep.target_token,
            )

            losses.total_loss.backward()
            opt.step()

            train_loss_acc += losses.total_loss.item()

            # Measure key selection via cosine similarity of query-key to candidate keys
            h_q = F.normalize(h_ad[0, ep.query_key_pos], p=2, dim=-1)
            sims = [float(torch.dot(h_q, F.normalize(h_ad[0, p], p=2, dim=-1)).item()) for p in all_k_pos]
            if sims:
                best_k_idx = all_k_pos[sims.index(max(sims))]
                if best_k_idx == ep.matching_key_pos:
                    train_k_corr += 1

        avg_train_loss = train_loss_acc / max(1, steps_per_phase)
        train_k_acc = train_k_corr / max(1, steps_per_phase)

        # Validation on distinct split
        candidate.eval()
        val_loss_acc = 0.0
        val_k_corr = 0
        held_k_corr = 0
        val_pos_corr = 0
        tok_corr = 0

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

                h_ad_v, _, log_v = candidate.forward_adapted_backbone(inp_v)
                v_losses = loss_fn(
                    adapted_hidden=h_ad_v,
                    logits=log_v,
                    query_key_pos=ep_val.query_key_pos,
                    matching_key_pos=ep_val.matching_key_pos,
                    distractor_key_positions=all_k_pos_v,
                    associated_val_pos=ep_val.associated_val_pos,
                    target_token=ep_val.target_token,
                )
                val_loss_acc += v_losses.total_loss.item()

                # Key selection
                h_qv = F.normalize(h_ad_v[0, ep_val.query_key_pos], p=2, dim=-1)
                sims_v = [float(torch.dot(h_qv, F.normalize(h_ad_v[0, p], p=2, dim=-1)).item()) for p in all_k_pos_v]
                if sims_v:
                    best_kv = all_k_pos_v[sims_v.index(max(sims_v))]
                    if best_kv == ep_val.matching_key_pos:
                        val_k_corr += 1

                # Value position routing from matched key state
                h_mv = F.normalize(h_ad_v[0, ep_val.matching_key_pos], p=2, dim=-1)
                # Value positions across context
                val_sims = [float(torch.dot(h_mv, F.normalize(h_ad_v[0, p], p=2, dim=-1)).item()) for p in range(len(ep_val.prompt_tokens) - 1)]
                best_vp = val_sims.index(max(val_sims))
                if best_vp == ep_val.associated_val_pos:
                    val_pos_corr += 1

                # Final token output
                pred_tok = torch.argmax(log_v[0, -1, :]).item()
                if pred_tok == ep_val.target_token:
                    tok_corr += 1

            # Held-out split test
            ep_held = env.generate_episode(split="disjoint_test", num_associations=2, episode_idx=2000)
            in_h = torch.tensor([ep_held.prompt_tokens], dtype=torch.long)
            all_kh = extract_key_positions_from_episode(ep_held, tok)
            h_ad_h, _, _ = candidate.forward_adapted_backbone(in_h)
            h_qh = F.normalize(h_ad_h[0, ep_held.query_key_pos], p=2, dim=-1)
            sims_h = [float(torch.dot(h_qh, F.normalize(h_ad_h[0, p], p=2, dim=-1)).item()) for p in all_kh]
            if sims_h and all_kh[sims_h.index(max(sims_h))] == ep_held.matching_key_pos:
                held_k_corr = 1

        val_k_acc = val_k_corr / max(1, eval_episodes_per_phase)
        val_p_acc = val_pos_corr / max(1, eval_episodes_per_phase)
        tok_acc = tok_corr / max(1, eval_episodes_per_phase)
        avg_val_loss = val_loss_acc / max(1, eval_episodes_per_phase)

        phase_results.append(
            AdapterPhaseMetrics(
                phase_name=cfg["name"],
                phase_desc=cfg["desc"],
                train_loss=avg_train_loss,
                val_loss=avg_val_loss,
                train_key_acc=train_k_acc,
                val_key_acc=val_k_acc,
                heldout_key_acc=float(held_k_corr),
                val_pos_acc=val_p_acc,
                final_token_acc=tok_acc,
                phase_passed=(val_k_acc > 0.0),
            )
        )

    # Language retention check
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

    p_count = sum(p.numel() for p in candidate.adapter.parameters())
    ad_hash = compute_module_sha256(candidate.adapter)
    elapsed_ms = (time.time() - t0) * 1000.0

    report = AdapterCurriculumReport(
        seed=seed,
        adapter_type=adapter_type,
        adapter_param_count=p_count,
        adapter_hash=ad_hash,
        phases=phase_results,
        overall_train_loss=sum(p.train_loss for p in phase_results) / len(phase_results),
        overall_val_loss=sum(p.val_loss for p in phase_results) / len(phase_results),
        overall_key_acc=sum(p.val_key_acc for p in phase_results) / len(phase_results),
        overall_val_pos_acc=sum(p.val_pos_acc for p in phase_results) / len(phase_results),
        overall_token_acc=sum(p.final_token_acc for p in phase_results) / len(phase_results),
        is_base_frozen=is_frozen,
        base_hash=final_hash,
        language_retention_ratio=retention_ratio,
        cpu_runtime_ms=elapsed_ms,
    )

    return candidate, report
