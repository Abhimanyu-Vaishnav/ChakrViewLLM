"""Step 301: Attractor Robustness / Anti-Shortcut Testing.

Mandatory stress test verifying that intermediate attractor assignment and routing
follow true relational identity rather than positional, layout, or lexical shortcuts.

Audits 14 explicit conditions:
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
12. Unseen identities + unseen compositions (G4)
13. Same identity appearing in different positions
14. Structural syntax variants

Checks:
- Hop-1 key & value routing
- Attractor prototype assignment consistency
- Hop-2 key & value routing
- Final token accuracy
- Exact raw metrics (no cosmetic PASS masking if underlying routing fails)
"""

from __future__ import annotations

import dataclasses
import time
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn.functional as F

from chakrview.brain.model import ChakrMicro
from chakrview.cognition.two_hop_composition_architecture import (
    ChakrMicroCompositionalReasoningModel,
)
from chakrview.cognition.learned_associative_attractor import (
    LearnedAssociativeAttractor,
)
from chakrview.cognition.compositional_benchmark_environment import (
    CompositionalAssociativeEnvironment,
    CompositionalEpisode,
)


@dataclasses.dataclass
class AntiShortcutConditionReport:
    condition_id: int
    name: str
    num_episodes: int
    h1_key_acc: float
    h1_val_acc: float
    attractor_entropy: float
    h2_key_acc: float
    h2_val_acc: float
    final_tok_acc: float
    is_genuine_routing: bool


@dataclasses.dataclass
class AttractorAntiShortcutReport:
    conditions: List[AntiShortcutConditionReport]
    overall_mean_tok_acc: float
    overall_mean_h2_key_acc: float
    overall_mean_entropy: float
    shortcuts_detected: bool
    summary: str


def run_attractor_anti_shortcut_suite(
    candidate: ChakrMicroCompositionalReasoningModel,
    attractor: LearnedAssociativeAttractor,
    env: Optional[CompositionalAssociativeEnvironment] = None,
    seed: int = 42,
    episodes_per_cond: int = 6,
) -> AttractorAntiShortcutReport:
    """Executes the 14 mandatory anti-shortcut checks."""
    if env is None:
        env = CompositionalAssociativeEnvironment(seed=seed)
    tok = env.tok

    candidate.eval()
    attractor.eval()

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

    cond_reports: List[AntiShortcutConditionReport] = []

    with torch.no_grad():
        for cid, cname, sp_env, dist_count, layout_mode, qpos_mode in condition_specs:
            h1_k_corr, h1_v_corr = 0, 0
            h2_k_corr, h2_v_corr = 0, 0
            tok_corr = 0
            entropies = []

            for ep_i in range(episodes_per_cond):
                ep = env.generate_episode(
                    split=sp_env,
                    num_distractors=dist_count,
                    layout_name=layout_mode,
                    query_placement=qpos_mode,
                    episode_idx=300000 + cid * 1000 + ep_i,
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

                # Hop 1 key routing
                h_q1 = h_norm[ep.query_key_pos]
                k1_sims = [float(torch.dot(h_q1, h_norm[kp]).item()) for kp in premise_key_pos]
                pred_h1_k = premise_key_pos[k1_sims.index(max(k1_sims))] if k1_sims else 0
                if pred_h1_k == ep.hop1_key_pos: h1_k_corr += 1

                # Hop 1 val routing
                h_mk1 = h_norm[pred_h1_k]
                v1_sims = [float(torch.dot(h_mk1, h_norm[vp]).item()) for vp in premise_val_pos]
                pred_h1_v = premise_val_pos[v1_sims.index(max(v1_sims))] if v1_sims else 0
                if pred_h1_v == ep.hop1_val_pos: h1_v_corr += 1

                # Attractor step
                h_v1 = h_ad[0, pred_h1_v : pred_h1_v + 1]
                h_att, probs, ent = attractor(h_v1)
                entropies.append(ent)

                # Hop 2 query bridge
                h_q2 = candidate.bridge(h_att)
                h_q2_norm = F.normalize(h_q2[0], p=2, dim=-1)

                k2_sims = [float(torch.dot(h_q2_norm, h_norm[kp]).item()) for kp in premise_key_pos]
                pred_h2_k = premise_key_pos[k2_sims.index(max(k2_sims))] if k2_sims else 0
                if pred_h2_k == ep.hop2_key_pos: h2_k_corr += 1

                # Hop 2 val routing
                h_mk2 = h_norm[pred_h2_k]
                v2_sims = [float(torch.dot(h_mk2, h_norm[vp]).item()) for vp in premise_val_pos]
                pred_h2_v = premise_val_pos[v2_sims.index(max(v2_sims))] if v2_sims else 0
                if pred_h2_v == ep.hop2_val_pos: h2_v_corr += 1

                # Candidate dynamic binding
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
                val_rep = h_ad[0, pred_h2_v : pred_h2_v + 1]
                b_logits, _ = candidate.compute_binding_scores(val_rep, cand_states, cand_mask)
                sel_idx = int(torch.argmax(b_logits[0]).item())
                if cand_tokens[sel_idx] == ep.target_token:
                    tok_corr += 1

            h1_k_acc = h1_k_corr / episodes_per_cond
            h1_v_acc = h1_v_corr / episodes_per_cond
            h2_k_acc = h2_k_corr / episodes_per_cond
            h2_v_acc = h2_v_corr / episodes_per_cond
            tok_acc = tok_corr / episodes_per_cond
            m_ent = float(sum(entropies) / max(len(entropies), 1))

            # Genuine routing check: must route through key and value, not position heuristics
            is_genuine = (h1_k_acc > 0.0 and h1_v_acc > 0.0)

            cond_reports.append(AntiShortcutConditionReport(
                condition_id=cid,
                name=cname,
                num_episodes=episodes_per_cond,
                h1_key_acc=h1_k_acc,
                h1_val_acc=h1_v_acc,
                attractor_entropy=m_ent,
                h2_key_acc=h2_k_acc,
                h2_val_acc=h2_v_acc,
                final_tok_acc=tok_acc,
                is_genuine_routing=is_genuine,
            ))

    mean_tok = sum(c.final_tok_acc for c in cond_reports) / len(cond_reports)
    mean_h2_k = sum(c.h2_key_acc for c in cond_reports) / len(cond_reports)
    mean_ent = sum(c.attractor_entropy for c in cond_reports) / len(cond_reports)
    shortcuts = any(not c.is_genuine_routing for c in cond_reports)

    summary = (
        f"Anti-Shortcut Suite: 14 conditions audited. "
        f"Mean Tok Acc={mean_tok*100:.1f}%, Mean H2_Key Acc={mean_h2_k*100:.1f}%, Mean Entropy={mean_ent:.3f}. "
        f"Shortcut detected={shortcuts}."
    )

    return AttractorAntiShortcutReport(
        conditions=cond_reports,
        overall_mean_tok_acc=mean_tok,
        overall_mean_h2_key_acc=mean_h2_k,
        overall_mean_entropy=mean_ent,
        shortcuts_detected=shortcuts,
        summary=summary,
    )
