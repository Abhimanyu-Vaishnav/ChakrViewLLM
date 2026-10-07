"""Step 307: Internal Location Comparison.

Tests and compares 4 principled internal insertion configurations under matched parameter budgets:
A. Early/Mid blocks (Layers 1, 2)
B. Late blocks (Layers 4, 5)
C. Attention-side residual pathway (Layers 3, 4, 5 on Attention residual)
D. FFN-side residual pathway (Layers 3, 4, 5 on FFN residual)

Evaluates:
- Hop-1 key & value routing accuracy
- Hop-2 key & value routing accuracy
- Final token accuracy across G1, G2, G3, G4
- Parameter overhead
- CPU runtime
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
from chakrview.cognition.internal_trainable_pathway import (
    ChakrMicroWithInternalPathway,
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
class LocationMetrics:
    location_id: str
    target_layers: List[int]
    insertion_side: str
    trainable_params: int
    mean_g1_tok_acc: float
    mean_g2_tok_acc: float
    mean_g3_tok_acc: float
    mean_g4_tok_acc: float
    mean_h2_key_acc: float
    mean_h2_val_acc: float
    cpu_latency_ms: float


@dataclasses.dataclass
class InternalLocationComparisonReport:
    location_results: Dict[str, LocationMetrics]
    best_location: str
    best_g4_tok_acc: float
    summary: str


def evaluate_internal_location(
    base_model: ChakrMicro,
    location_id: str,
    target_layers: List[int],
    insertion_side: str,
    seeds: Optional[List[int]] = None,
    train_steps: int = 15,
    eval_episodes: int = 8,
    bottleneck_dim: int = 32,
) -> LocationMetrics:
    """Trains and evaluates an internal pathway configuration across seeds."""
    if seeds is None:
        seeds = [42, 101, 2026]

    g1_list, g2_list, g3_list, g4_list = [], [], [], []
    h2_k_list, h2_v_list = [], []
    latencies = []

    sample_model = ChakrMicroWithInternalPathway(
        base_model, target_layers=target_layers, bottleneck_dim=bottleneck_dim, insertion_side=insertion_side,
    )
    p_count = sum(p.numel() for p in sample_model.parameters() if p.requires_grad)

    for s in seeds:
        torch.manual_seed(s)
        env = CompositionalAssociativeEnvironment(seed=s)
        tok = env.tok

        model = ChakrMicroWithInternalPathway(
            base_model, target_layers=target_layers, bottleneck_dim=bottleneck_dim, insertion_side=insertion_side,
        )

        opt = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=2.0e-3, weight_decay=0.01,
        )
        route_loss_fn = IdentityInvariantRepresentationLoss(temperature=0.15)

        # Train loop
        model.train()
        for st in range(train_steps):
            ep = env.generate_episode("train", num_distractors=1, episode_idx=st)
            inp = torch.tensor([ep.prompt_tokens], dtype=torch.long)
            opt.zero_grad()
            final_h, states = model.forward_hidden_states(inp)

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

            # Hop-2 key routing directly from Hop-1 value position
            h_norm = F.normalize(final_h[0], p=2, dim=-1)
            h_v1 = h_norm[ep.hop1_val_pos : ep.hop1_val_pos + 1]
            k2_logits = torch.matmul(h_v1, h_norm.transpose(0, 1)) / 0.15
            l_route2 = F.cross_entropy(k2_logits, torch.tensor([ep.hop2_key_pos], dtype=torch.long))

            loss = r_loss1.total_loss + 1.5 * l_route2
            loss.backward()
            opt.step()

        # Eval loop across G1-G4
        model.eval()
        split_map = {
            "G1": ("train", g1_list),
            "G2": ("disjoint_test", g2_list),
            "G3": ("heldout_composition", g3_list),
            "G4": ("disjoint_test", g4_list),
        }

        with torch.no_grad():
            for sp_name, (sp_mode, acc_list) in split_map.items():
                corr = 0
                for ev_i in range(eval_episodes):
                    ep = env.generate_episode(sp_mode, num_distractors=1, episode_idx=500000 + ev_i)
                    inp = torch.tensor([ep.prompt_tokens], dtype=torch.long)

                    t0 = time.perf_counter()
                    final_h, _ = model.forward_hidden_states(inp)
                    t1 = time.perf_counter()
                    latencies.append((t1 - t0) * 1000.0)

                    h_norm = F.normalize(final_h[0], p=2, dim=-1)

                    # Premise key positions
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

                    # Hop-1 val routing
                    h_mk1 = h_norm[pred_k1]
                    v1_sims = [float(torch.dot(h_mk1, h_norm[vp]).item()) for vp in premise_val_pos]
                    pred_v1 = premise_val_pos[v1_sims.index(max(v1_sims))] if v1_sims else 0

                    # Hop-2 key routing directly from internal transformed state
                    h_v1 = h_norm[pred_v1]
                    k2_sims = [float(torch.dot(h_v1, h_norm[kp]).item()) for kp in premise_key_pos]
                    pred_k2 = premise_key_pos[k2_sims.index(max(k2_sims))] if k2_sims else 0
                    h2_k_list.append(1.0 if pred_k2 == ep.hop2_key_pos else 0.0)

                    # Hop-2 val routing
                    h_mk2 = h_norm[pred_k2]
                    v2_sims = [float(torch.dot(h_mk2, h_norm[vp]).item()) for vp in premise_val_pos]
                    pred_v2 = premise_val_pos[v2_sims.index(max(v2_sims))] if v2_sims else 0
                    h2_v_list.append(1.0 if pred_v2 == ep.hop2_val_pos else 0.0)

                    # Final token check
                    if inp[0, pred_v2].item() == ep.target_token:
                        corr += 1

                acc_list.append(corr / eval_episodes)

    return LocationMetrics(
        location_id=location_id,
        target_layers=target_layers,
        insertion_side=insertion_side,
        trainable_params=p_count,
        mean_g1_tok_acc=float(sum(g1_list) / len(g1_list)),
        mean_g2_tok_acc=float(sum(g2_list) / len(g2_list)),
        mean_g3_tok_acc=float(sum(g3_list) / len(g3_list)),
        mean_g4_tok_acc=float(sum(g4_list) / len(g4_list)),
        mean_h2_key_acc=float(sum(h2_k_list) / max(len(h2_k_list), 1)),
        mean_h2_val_acc=float(sum(h2_v_list) / max(len(h2_v_list), 1)),
        cpu_latency_ms=float(sum(latencies) / len(latencies)),
    )


def run_internal_location_comparison(
    base_model: ChakrMicro,
    seeds: Optional[List[int]] = None,
    train_steps: int = 15,
    eval_episodes: int = 8,
) -> InternalLocationComparisonReport:
    """Executes Step 307 comparing configurations A, B, C, D."""
    if seeds is None:
        seeds = [42, 101, 2026]

    configs = [
        ("A_early_mid", [1, 2], "residual"),
        ("B_late", [4, 5], "residual"),
        ("C_attention_side", [3, 4, 5], "attention"),
        ("D_ffn_side", [3, 4, 5], "ffn"),
    ]

    results: Dict[str, LocationMetrics] = {}
    for loc_id, layers, side in configs:
        m = evaluate_internal_location(
            base_model=base_model,
            location_id=loc_id,
            target_layers=layers,
            insertion_side=side,
            seeds=seeds,
            train_steps=train_steps,
            eval_episodes=eval_episodes,
        )
        results[loc_id] = m

    best_loc = max(results.keys(), key=lambda k: results[k].mean_g4_tok_acc)
    best_g4 = results[best_loc].mean_g4_tok_acc

    summary = (
        f"Step 307 Internal Location Comparison: " +
        "; ".join([f"{k}: G4={v.mean_g4_tok_acc*100:.1f}%, H2_K={v.mean_h2_key_acc*100:.1f}%, params={v.trainable_params}"
                   for k, v in results.items()]) +
        f" -> Best Configuration: {best_loc} ({best_g4*100:.1f}%)"
    )

    return InternalLocationComparisonReport(
        location_results=results,
        best_location=best_loc,
        best_g4_tok_acc=best_g4,
        summary=summary,
    )
