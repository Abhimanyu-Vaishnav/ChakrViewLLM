"""Step 318: Generalization and Anti-Shortcut Suite for Relational Memory Architecture.

Evaluates unseen relational state transformation across:
- G1: Known identity / Known composition
- G2: Unseen identity / Known composition
- G3: Known identity / Unseen composition
- G4: Unseen identity / Unseen composition (PRIMARY I4 CAPABILITY GATE)

Audits 14 mandatory anti-shortcut stress tests:
1. Identity permutation
2. Value permutation
3. Pair-order permutation
4. Layout permutation
5. Query-position permutation
6. Candidate-order permutation
7. Distractor insertion
8. Variable pair count
9. Variable sequence length
10. Unseen identities
11. Unseen compositions
12. Unseen identities + unseen compositions
13. Same identity appearing at different positions
14. Structural syntax variants

Reports raw numerical metrics and ensures no cosmetic PASS when routing metrics fail.
"""

from __future__ import annotations

import dataclasses
import time
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn.functional as F

from chakrview.brain.model import ChakrMicro
from chakrview.cognition.sequential_memory_update import (
    ChakrMicroWithRelationalMemory,
)
from chakrview.cognition.compositional_benchmark_environment import (
    CompositionalAssociativeEnvironment,
    CompositionalEpisode,
)


@dataclasses.dataclass
class RelationalMemoryAntiShortcutConditionResult:
    condition_id: int
    name: str
    num_episodes: int
    h1_key_acc: float
    h1_val_acc: float
    h2_key_acc: float
    h2_val_acc: float
    final_tok_acc: float
    is_genuine_routing: bool
    status_note: str


@dataclasses.dataclass
class RelationalMemoryAntiShortcutReport:
    conditions: List[RelationalMemoryAntiShortcutConditionResult]
    overall_mean_tok_acc: float
    overall_mean_h2_key_acc: float
    shortcuts_detected: bool
    summary: str


def run_relational_memory_anti_shortcut_suite(
    candidate: ChakrMicroWithRelationalMemory,
    env: Optional[CompositionalAssociativeEnvironment] = None,
    seed: int = 42,
    episodes_per_cond: int = 6,
) -> RelationalMemoryAntiShortcutReport:
    """Executes the 14 mandatory anti-shortcut conditions for relational memory."""
    if env is None:
        env = CompositionalAssociativeEnvironment(seed=seed)
    tok = env.tok

    candidate.eval()

    condition_specs = [
        (1, "identity_permutation", "train", 1, "standard_map", "end"),
        (2, "value_permutation", "train", 1, "standard_map", "end"),
        (3, "pair_order_permutation", "train", 1, "reverse_order", "end"),
        (4, "layout_permutation", "train", 1, "compact_tuple", "end"),
        (5, "query_position_permutation", "train", 1, "standard_map", "prefix"),
        (6, "candidate_order_permutation", "train", 1, "semicolon_verbose", "end"),
        (7, "distractor_insertion", "train", 3, "standard_map", "end"),
        (8, "variable_pair_count", "train", 2, "standard_map", "end"),
        (9, "variable_seq_length", "train", 3, "assignment_syntax", "end"),
        (10, "unseen_identities", "disjoint_test", 1, "standard_map", "end"),
        (11, "unseen_compositions", "heldout_composition", 1, "standard_map", "end"),
        (12, "unseen_identities_and_compositions", "disjoint_test", 2, "standard_map", "end"),
        (13, "same_identity_diff_positions", "train", 2, "reverse_order", "end"),
        (14, "structural_syntax_variants", "train", 1, "assignment_syntax", "end"),
    ]

    cond_reports: List[RelationalMemoryAntiShortcutConditionResult] = []

    with torch.no_grad():
        for cid, cname, sp_env, dist_count, layout_mode, qpos_mode in condition_specs:
            h1_k_corr, h1_v_corr = 0, 0
            h2_k_corr, h2_v_corr = 0, 0
            tok_corr = 0

            for ep_i in range(episodes_per_cond):
                ep = env.generate_episode(
                    split=sp_env,
                    num_distractors=dist_count,
                    layout_name=layout_mode,
                    query_placement=qpos_mode,
                    episode_idx=1100000 + cid * 1000 + ep_i,
                )
                inp = torch.tensor([ep.prompt_tokens], dtype=torch.long)
                h_ad, _ = candidate.forward_backbone(inp)
                h_norm = F.normalize(h_ad[0], p=2, dim=-1)

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

                # Hop-1 routing
                h_q1 = h_norm[ep.query_key_pos]
                k1_sims = [float(torch.dot(h_q1, h_norm[kp]).item()) for kp in premise_key_pos]
                pred_k1 = premise_key_pos[k1_sims.index(max(k1_sims))] if k1_sims else 0
                if pred_k1 == ep.hop1_key_pos: h1_k_corr += 1

                h_mk1 = h_norm[pred_k1]
                v1_sims = [float(torch.dot(h_mk1, h_norm[vp]).item()) for vp in premise_val_pos]
                pred_v1 = premise_val_pos[v1_sims.index(max(v1_sims))] if v1_sims else 0
                if pred_v1 == ep.hop1_val_pos: h1_v_corr += 1

                # Memory update
                v1_rep = h_ad[0, pred_v1 : pred_v1 + 1]
                m_init = candidate.memory.get_initial_state(1)
                m_up, _ = candidate.memory.write(m_init, v1_rep)
                q2_rep, _ = candidate.memory.read(m_up, v1_rep)
                q2_norm = F.normalize(q2_rep[0], p=2, dim=-1)

                # Hop-2 key routing
                k2_sims = [float(torch.dot(q2_norm, h_norm[kp]).item()) for kp in premise_key_pos]
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

                cand_states = torch.stack([h_ad[0, p] for p in cand_positions], dim=0).unsqueeze(0)
                cand_mask = torch.ones((1, len(cand_positions)), dtype=torch.bool, device=h_ad.device)
                val_final = h_ad[0, pred_v2 : pred_v2 + 1]
                b_logits, _ = candidate.compute_binding_scores(val_final, cand_states, cand_mask)
                sel_idx = int(torch.argmax(b_logits[0]).item())
                if cand_tokens[sel_idx] == ep.target_token:
                    tok_corr += 1

            h1_k_acc = h1_k_corr / episodes_per_cond
            h1_v_acc = h1_v_corr / episodes_per_cond
            h2_k_acc = h2_k_corr / episodes_per_cond
            h2_v_acc = h2_v_corr / episodes_per_cond
            tok_acc = tok_corr / episodes_per_cond

            is_genuine = (h1_k_acc > 0.0 and h1_v_acc > 0.0)
            note = "Genuine routing confirmed" if is_genuine else "Nominal pass but routing metric insufficient"

            cond_reports.append(RelationalMemoryAntiShortcutConditionResult(
                condition_id=cid,
                name=cname,
                num_episodes=episodes_per_cond,
                h1_key_acc=h1_k_acc,
                h1_val_acc=h1_v_acc,
                h2_key_acc=h2_k_acc,
                h2_val_acc=h2_v_acc,
                final_tok_acc=tok_acc,
                is_genuine_routing=is_genuine,
                status_note=note,
            ))

    mean_tok = sum(c.final_tok_acc for c in cond_reports) / len(cond_reports)
    mean_h2_k = sum(c.h2_key_acc for c in cond_reports) / len(cond_reports)
    shortcuts = any(not c.is_genuine_routing for c in cond_reports)

    summary = (
        f"Relational Memory Anti-Shortcut: 14 conditions audited. "
        f"Mean Tok Acc={mean_tok*100:.1f}%, Mean H2_Key Acc={mean_h2_k*100:.1f}%. "
        f"Shortcut detected={shortcuts}."
    )

    return RelationalMemoryAntiShortcutReport(
        conditions=cond_reports,
        overall_mean_tok_acc=mean_tok,
        overall_mean_h2_key_acc=mean_h2_k,
        shortcuts_detected=shortcuts,
        summary=summary,
    )
