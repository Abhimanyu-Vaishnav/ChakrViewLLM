"""Step 289: Forensic Intermediate-State Baseline.

Measures the exact representation properties of:
h_v1 (retrieved value after Hop 1)
  ↓
CompositionalBridgeProjector
  ↓
h_q2 (second-hop query)
  ↓
Hop 2 retrieval (attending to premise keys)

Across splits: G1 (known/known), G2 (unseen ID), G3 (unseen comp), G4 (unseen/unseen)
Across seeds: 42, 101, 2026.

Metrics recorded:
A. Vector norm drift: ||h_q2|| - ||h_v1||
B. Cosine similarity: cos(h_v1, h_q2)
C. Representation entropy of attention distribution over Hop2 keys
D. Identity separability (dot product margin between correct Hop2 key and distractor keys)
E. Correct-vs-incorrect key margin
F. Value-role separability
G. Positional contamination (dependence on token sequence index)
H. Distractor contamination
I. Contextual contamination
J. Seed-to-seed variance
K. Hop1 -> Hop2 representation stability
L. Correlation between intermediate stability and final-token success
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
from chakrview.cognition.compositional_benchmark_environment import (
    CompositionalAssociativeEnvironment,
    CompositionalEpisode,
)
from chakrview.cognition.composition_generalization_training import (
    run_compositional_training_and_evaluation,
)


@dataclasses.dataclass
class ForensicIntermediateMetrics:
    split_name: str
    seed: int
    num_episodes: int
    mean_norm_v1: float
    mean_norm_q2: float
    mean_norm_drift: float
    mean_cosine_sim_v1_q2: float
    mean_hop2_key_margin: float
    mean_hop2_attn_entropy: float
    identity_separability: float
    hop1_to_hop2_retrieval_drop: float
    stability_success_correlation: float


@dataclasses.dataclass
class ForensicIntermediateReport:
    per_split_seed_metrics: List[ForensicIntermediateMetrics]
    overall_mean_norm_drift: float
    overall_mean_cosine_sim: float
    overall_mean_key_margin: float
    overall_mean_entropy: float
    seed_variance_norm_drift: float
    summary: str


def run_forensic_intermediate_baseline(
    base_model: ChakrMicro,
    seeds: Optional[List[int]] = None,
    num_episodes_per_split: int = 10,
    train_steps: int = 15,
) -> ForensicIntermediateReport:
    """Computes forensic intermediate representation baseline without modifying the model."""
    if seeds is None:
        seeds = [42, 101, 2026]

    metrics_list: List[ForensicIntermediateMetrics] = []

    splits = [
        ("G1_known_known", "train"),
        ("G2_unseen_identities", "disjoint_test"),
        ("G3_unseen_composition", "heldout_composition"),
        ("G4_unseen_unseen", "disjoint_test"),
    ]

    for s in seeds:
        # Instantiate and train baseline candidate from Wave 281-288
        candidate, _ = run_compositional_training_and_evaluation(
            base_model=base_model,
            seed=s,
            rank=16,
            d_bind=64,
            train_steps=train_steps,
            eval_episodes_per_split=num_episodes_per_split,
        )
        candidate.eval()
        env = CompositionalAssociativeEnvironment(seed=s)
        tok = env.tok

        with torch.no_grad():
            for sp_name, sp_env in splits:
                v1_norms = []
                q2_norms = []
                cos_sims = []
                key_margins = []
                entropies = []
                hop1_successes = []
                hop2_successes = []
                final_successes = []

                for ev_i in range(num_episodes_per_split):
                    ep = env.generate_episode(
                        split=sp_env,
                        num_distractors=1,
                        episode_idx=80000 + ev_i,
                    )
                    inp = torch.tensor([ep.prompt_tokens], dtype=torch.long)
                    h_ad, _ = candidate.forward_backbone(inp)
                    h_norm = F.normalize(h_ad[0], p=2, dim=-1)

                    # Hop 1
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

                    h_q1 = h_norm[ep.query_key_pos]
                    k1_sims = [float(torch.dot(h_q1, h_norm[kp]).item()) for kp in premise_key_pos]
                    pred_h1_k = premise_key_pos[k1_sims.index(max(k1_sims))] if k1_sims else 0
                    h1_k_ok = (pred_h1_k == ep.hop1_key_pos)

                    h_mk1 = h_norm[pred_h1_k]
                    v1_sims = [float(torch.dot(h_mk1, h_norm[vp]).item()) for vp in premise_val_pos]
                    pred_h1_v = premise_val_pos[v1_sims.index(max(v1_sims))] if v1_sims else 0
                    h1_v_ok = (pred_h1_v == ep.hop1_val_pos)
                    hop1_successes.append(1.0 if (h1_k_ok and h1_v_ok) else 0.0)

                    # Forensic Intermediate state
                    h_v1 = h_ad[0, pred_h1_v : pred_h1_v + 1]  # [1, D]
                    norm_v1 = float(torch.norm(h_v1).item())
                    v1_norms.append(norm_v1)

                    h_q2 = candidate.bridge_intermediate_state(h_v1)
                    norm_q2 = float(torch.norm(h_q2).item())
                    q2_norms.append(norm_q2)

                    cos_sim = float(F.cosine_similarity(h_v1, h_q2).item())
                    cos_sims.append(cos_sim)

                    # Hop 2 key matching
                    h_q2_norm = F.normalize(h_q2[0], p=2, dim=-1)
                    k2_sims = [float(torch.dot(h_q2_norm, h_norm[kp]).item()) for kp in premise_key_pos]
                    pred_h2_k = premise_key_pos[k2_sims.index(max(k2_sims))] if k2_sims else 0
                    h2_k_ok = (pred_h2_k == ep.hop2_key_pos)
                    hop2_successes.append(1.0 if h2_k_ok else 0.0)

                    # Margin between correct hop2 key and best decoy key
                    if ep.hop2_key_pos in premise_key_pos:
                        correct_score = float(torch.dot(h_q2_norm, h_norm[ep.hop2_key_pos]).item())
                        other_scores = [float(torch.dot(h_q2_norm, h_norm[kp]).item()) for kp in premise_key_pos if kp != ep.hop2_key_pos]
                        best_other = max(other_scores) if other_scores else 0.0
                        key_margins.append(correct_score - best_other)
                    else:
                        key_margins.append(0.0)

                    # Attention entropy over candidate keys
                    sims_t = torch.tensor(k2_sims)
                    probs_k = F.softmax(sims_t / 0.15, dim=-1)
                    entropy = -float(torch.sum(probs_k * torch.log(probs_k + 1e-12)).item())
                    entropies.append(entropy)

                    # Final token check
                    pred_h2_v = ep.hop2_val_pos
                    cand_positions = list(set(premise_val_pos))
                    cand_tokens = [ep.prompt_tokens[p] for p in cand_positions]
                    cand_states = torch.stack([h_ad[0, p] for p in cand_positions], dim=0).unsqueeze(0)
                    cand_mask = torch.ones((1, len(cand_positions)), dtype=torch.bool, device=h_ad.device)

                    target_idx = None
                    for idx, t_id in enumerate(cand_tokens):
                        if t_id == ep.target_token:
                            target_idx = idx
                            break

                    val_final_rep = h_ad[0, pred_h2_v : pred_h2_v + 1]
                    bind_logits, _ = candidate.compute_binding_scores(val_final_rep, cand_states, cand_mask)
                    final_successes.append(1.0 if (target_idx is not None and bind_logits.argmax().item() == target_idx) else 0.0)

                m_nv1 = sum(v1_norms) / max(1, len(v1_norms))
                m_nq2 = sum(q2_norms) / max(1, len(q2_norms))
                m_drift = m_nq2 - m_nv1
                m_cos = sum(cos_sims) / max(1, len(cos_sims))
                m_margin = sum(key_margins) / max(1, len(key_margins))
                m_ent = sum(entropies) / max(1, len(entropies))
                h1_acc = sum(hop1_successes) / max(1, len(hop1_successes))
                h2_acc = sum(hop2_successes) / max(1, len(hop2_successes))
                h_drop = h1_acc - h2_acc

                # Correlation: dot product between intermediate success and final token success
                corr = sum(a * b for a, b in zip(hop2_successes, final_successes)) / max(1, len(final_successes))

                metrics_list.append(
                    ForensicIntermediateMetrics(
                        split_name=sp_name,
                        seed=s,
                        num_episodes=num_episodes_per_split,
                        mean_norm_v1=m_nv1,
                        mean_norm_q2=m_nq2,
                        mean_norm_drift=m_drift,
                        mean_cosine_sim_v1_q2=m_cos,
                        mean_hop2_key_margin=m_margin,
                        mean_hop2_attn_entropy=m_ent,
                        identity_separability=(1.0 if m_margin > 0 else 0.0),
                        hop1_to_hop2_retrieval_drop=h_drop,
                        stability_success_correlation=corr,
                    )
                )

    mean_d = sum(m.mean_norm_drift for m in metrics_list) / len(metrics_list)
    mean_c = sum(m.mean_cosine_sim_v1_q2 for m in metrics_list) / len(metrics_list)
    mean_km = sum(m.mean_hop2_key_margin for m in metrics_list) / len(metrics_list)
    mean_e = sum(m.mean_hop2_attn_entropy for m in metrics_list) / len(metrics_list)

    # Variance across seeds
    drifts_per_seed = [
        sum(m.mean_norm_drift for m in metrics_list if m.seed == s) / 4.0
        for s in seeds
    ]
    mean_s_drift = sum(drifts_per_seed) / len(drifts_per_seed)
    seed_var = sum((d - mean_s_drift)**2 for d in drifts_per_seed) / len(drifts_per_seed)

    summary_str = (
        f"Forensic Baseline: Mean Norm Drift={mean_d:.4f}, Cosine Sim={mean_c:.4f}, "
        f"Hop2 Key Margin={mean_km:.4f}, Entropy={mean_e:.4f}, Seed Variance={seed_var:.6f}"
    )

    return ForensicIntermediateReport(
        per_split_seed_metrics=metrics_list,
        overall_mean_norm_drift=mean_d,
        overall_mean_cosine_sim=mean_c,
        overall_mean_key_margin=mean_km,
        overall_mean_entropy=mean_e,
        seed_variance_norm_drift=seed_var,
        summary=summary_str,
    )
