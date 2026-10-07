"""Step 325: Distractor-Robust State Transition Suite.

Tests whether the recurrent state transition maintains relational state purity under
increasing distractor pair scales (0, 1, 2, 3, 5 distractors):
Measures:
- Hop-1 key routing accuracy
- Hop-1 value routing accuracy
- Recurrent state norm and purity
- Hop-2 key routing accuracy
- Hop-2 value routing accuracy
- Final token accuracy
"""

from __future__ import annotations

import dataclasses
import time
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn.functional as F

from chakrview.brain.model import ChakrMicro
from chakrview.cognition.state_attention_integration import (
    StateAttentionIntegrationModel,
)
from chakrview.cognition.compositional_benchmark_environment import (
    CompositionalAssociativeEnvironment,
    CompositionalEpisode,
)


@dataclasses.dataclass
class DistractorScaleStateMetrics:
    num_distractors: int
    num_episodes: int
    h1_key_acc: float
    h1_val_acc: float
    state_purity: float
    h2_key_acc: float
    h2_val_acc: float
    final_tok_acc: float


@dataclasses.dataclass
class DistractorRobustStateReport:
    scale_metrics: Dict[int, DistractorScaleStateMetrics]
    mean_state_purity: float
    distractor_resilient: bool
    summary: str


def evaluate_distractor_robust_state(
    model: StateAttentionIntegrationModel,
    env: Optional[CompositionalAssociativeEnvironment] = None,
    seed: int = 42,
    episodes_per_scale: int = 8,
) -> DistractorRobustStateReport:
    """Evaluates recurrent state transition under 0, 1, 2, 3, 5 distractors."""
    if env is None:
        env = CompositionalAssociativeEnvironment(seed=seed)
    tok = env.tok

    model.eval()
    distractor_counts = [0, 1, 2, 3, 5]
    results: Dict[int, DistractorScaleStateMetrics] = {}

    with torch.no_grad():
        for d_cnt in distractor_counts:
            h1_k_corr, h1_v_corr = 0, 0
            h2_k_corr, h2_v_corr = 0, 0
            tok_corr = 0
            purities = []

            for ep_i in range(episodes_per_scale):
                ep = env.generate_episode("disjoint_test", num_distractors=d_cnt, episode_idx=1600000 + d_cnt * 1000 + ep_i)
                inp = torch.tensor([ep.prompt_tokens], dtype=torch.long)
                h_ad, _ = model.forward_backbone(inp)
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

                # Hop-1 routing
                h_q1 = h_norm[ep.query_key_pos]
                k1_sims = [float(torch.dot(h_q1, h_norm[kp]).item()) for kp in premise_key_pos]
                pred_k1 = premise_key_pos[k1_sims.index(max(k1_sims))] if k1_sims else 0
                if pred_k1 == ep.hop1_key_pos: h1_k_corr += 1

                h_mk1 = h_norm[pred_k1]
                v1_sims = [float(torch.dot(h_mk1, h_norm[vp]).item()) for vp in premise_val_pos]
                pred_v1 = premise_val_pos[v1_sims.index(max(v1_sims))] if v1_sims else 0
                if pred_v1 == ep.hop1_val_pos: h1_v_corr += 1

                v1_rep = h_ad[0, pred_v1 : pred_v1 + 1]
                s0 = model.transition_core.get_initial_state(1)
                q2, _ = model.compute_second_hop_query(v1_rep, s0)
                q2_norm = F.normalize(q2[0], p=2, dim=-1)

                # Hop-2 key routing
                k2_sims = [float(torch.dot(q2_norm, h_norm[kp]).item()) for kp in premise_key_pos]
                pred_k2 = premise_key_pos[k2_sims.index(max(k2_sims))] if k2_sims else 0
                if pred_k2 == ep.hop2_key_pos: h2_k_corr += 1

                decoy_sims = [float(torch.dot(q2_norm, h_norm[kp]).item()) for kp in premise_key_pos if kp != ep.hop2_key_pos]
                avg_decoy = sum(decoy_sims) / max(len(decoy_sims), 1) if decoy_sims else 0.0
                purities.append(max(k2_sims) - avg_decoy if k2_sims else 0.0)

                # Hop-2 val routing
                h_mk2 = h_norm[pred_k2]
                v2_sims = [float(torch.dot(h_mk2, h_norm[vp]).item()) for vp in premise_val_pos]
                pred_v2 = premise_val_pos[v2_sims.index(max(v2_sims))] if v2_sims else 0
                if pred_v2 == ep.hop2_val_pos: h2_v_corr += 1

                val_final = h_ad[0, pred_v2 : pred_v2 + 1]
                b_logits, _ = model.binding(val_final, cand_states, cand_mask)
                sel_idx = int(torch.argmax(b_logits[0]).item())
                if cand_tokens[sel_idx] == ep.target_token:
                    tok_corr += 1

            results[d_cnt] = DistractorScaleStateMetrics(
                num_distractors=d_cnt,
                num_episodes=episodes_per_scale,
                h1_key_acc=h1_k_corr / episodes_per_scale,
                h1_val_acc=h1_v_corr / episodes_per_scale,
                state_purity=float(sum(purities) / max(len(purities), 1)),
                h2_key_acc=h2_k_corr / episodes_per_scale,
                h2_val_acc=h2_v_corr / episodes_per_scale,
                final_tok_acc=tok_corr / episodes_per_scale,
            )

    m_pur = sum(m.state_purity for m in results.values()) / len(results)
    high_d_tok = results[5].final_tok_acc
    is_resilient = (high_d_tok >= 0.25)

    summary = (
        f"Distractor-Robust State Study: Tested distractors in [0, 1, 2, 3, 5]. "
        f"Accuracies: " + ", ".join([f"{d}d: {results[d].final_tok_acc*100:.1f}%" for d in distractor_counts]) +
        f". Mean State Purity={m_pur:.3f}. Resilient (>25% at 5d): {is_resilient}."
    )

    return DistractorRobustStateReport(
        scale_metrics=results,
        mean_state_purity=m_pur,
        distractor_resilient=is_resilient,
        summary=summary,
    )
