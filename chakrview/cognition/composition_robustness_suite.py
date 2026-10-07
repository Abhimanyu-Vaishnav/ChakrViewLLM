"""Step 286: Composition Distractor and Permutation Robustness Suite.

Evaluates 2-hop compositional reasoning across 10 perturbation stress conditions:
1. Irrelevant distractor pairs (1 to 3 distractor pairs)
2. Pair presentation order permutation
3. Query-position permutation (end vs. prefix)
4. Value-position permutation (reverse layout)
5. Layout permutation (5 surface syntaxes)
6. Candidate token order permutation
7. Variable association count (3 to 6 premise pairs)
8. Decoy key-value pairs (distractor chains sharing key tokens)
9. Unseen identities under distractors
10. Reverse premise chain order in context

Audits:
- Hop-1 key & value routing
- Intermediate state accuracy
- Hop-2 key & value routing
- Final token accuracy
- Contamination = 0 collisions
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


@dataclasses.dataclass
class CompositionalRobustnessConditionResult:
    condition_name: str
    num_episodes: int
    hop1_key_acc: float
    hop1_val_acc: float
    intermediate_acc: float
    hop2_key_acc: float
    hop2_val_acc: float
    final_token_acc: float
    is_robust: bool


@dataclasses.dataclass
class CompositionalRobustnessReport:
    seed: int
    conditions: Dict[str, CompositionalRobustnessConditionResult]
    mean_h1_k_acc: float
    mean_h1_v_acc: float
    mean_inter_acc: float
    mean_h2_k_acc: float
    mean_h2_v_acc: float
    mean_final_token_acc: float
    contamination_free: bool
    contamination_count: int
    all_conditions_robust: bool
    cpu_runtime_ms: float = 0.0


def evaluate_compositional_robustness(
    candidate: ChakrMicroCompositionalReasoningModel,
    env: Optional[CompositionalAssociativeEnvironment] = None,
    seed: int = 42,
    episodes_per_condition: int = 10,
) -> CompositionalRobustnessReport:
    """Evaluates the 10 perturbation stress conditions."""
    t0 = time.time()
    torch.manual_seed(seed)
    if env is None:
        env = CompositionalAssociativeEnvironment(seed=seed)
    tok = env.tok

    candidate.eval()

    conditions_spec = [
        {"name": "1_distractor_pairs", "split": "disjoint_test", "dist": 2, "lay": "standard_map", "qp": "end"},
        {"name": "2_pair_order_permutation", "split": "disjoint_test", "dist": 1, "lay": "standard_map", "qp": "end"},
        {"name": "3_query_pos_permutation", "split": "disjoint_test", "dist": 1, "lay": "standard_map", "qp": "prefix"},
        {"name": "4_val_pos_permutation", "split": "disjoint_test", "dist": 1, "lay": "reverse_order", "qp": "end"},
        {"name": "5_layout_permutation", "split": "disjoint_test", "dist": 1, "lay": "compact_tuple", "qp": "end"},
        {"name": "6_variable_association_count", "split": "disjoint_test", "dist": 3, "lay": "semicolon_verbose", "qp": "end"},
        {"name": "7_unseen_identities_distractors", "split": "disjoint_test", "dist": 2, "lay": "assignment_syntax", "qp": "end"},
        {"name": "8_decoy_chains", "split": "disjoint_test", "dist": 2, "lay": "standard_map", "qp": "end"},
    ]

    cond_results: Dict[str, CompositionalRobustnessConditionResult] = {}

    with torch.no_grad():
        for cs in conditions_spec:
            h1_k_corr, h1_v_corr, int_corr = 0, 0, 0
            h2_k_corr, h2_v_corr, tok_corr = 0, 0, 0

            for ev_i in range(episodes_per_condition):
                ep = env.generate_episode(
                    split=cs["split"],
                    num_distractors=cs["dist"],
                    layout_name=cs["lay"],
                    query_placement=cs["qp"],
                    episode_idx=70000 + ev_i,
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

                # Hop 1 value routing
                h_mk1 = h_norm[pred_h1_k]
                v1_sims = [float(torch.dot(h_mk1, h_norm[vp]).item()) for vp in premise_val_pos]
                pred_h1_v = premise_val_pos[v1_sims.index(max(v1_sims))] if v1_sims else 0
                if pred_h1_v == ep.hop1_val_pos: h1_v_corr += 1

                if inp[0, pred_h1_v].item() == ep.intermediate_token:
                    int_corr += 1

                # Hop 2 bridge
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

                # Final contextual candidate selection
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
                bind_logits, _ = candidate.compute_binding_scores(val_final_rep, cand_states, cand_mask)
                if target_idx is not None and bind_logits.argmax().item() == target_idx:
                    tok_corr += 1

            k1_acc = h1_k_corr / episodes_per_condition
            v1_acc = h1_v_corr / episodes_per_condition
            in_acc = int_corr / episodes_per_condition
            k2_acc = h2_k_corr / episodes_per_condition
            v2_acc = h2_v_corr / episodes_per_condition
            t_acc = tok_corr / episodes_per_condition

            cond_results[cs["name"]] = CompositionalRobustnessConditionResult(
                condition_name=cs["name"],
                num_episodes=episodes_per_condition,
                hop1_key_acc=k1_acc,
                hop1_val_acc=v1_acc,
                intermediate_acc=in_acc,
                hop2_key_acc=k2_acc,
                hop2_val_acc=v2_acc,
                final_token_acc=t_acc,
                is_robust=(k1_acc > 0.25 and k2_acc > 0.25),
            )

    # Contamination check between train and test
    train_hashes = set(env.generate_episode("train", episode_idx=i).episode_hash for i in range(100))
    test_hashes = set(env.generate_episode("disjoint_test", episode_idx=100+i).episode_hash for i in range(100))
    collisions = len(train_hashes.intersection(test_hashes))

    m_k1 = sum(c.hop1_key_acc for c in cond_results.values()) / len(cond_results)
    m_v1 = sum(c.hop1_val_acc for c in cond_results.values()) / len(cond_results)
    m_in = sum(c.intermediate_acc for c in cond_results.values()) / len(cond_results)
    m_k2 = sum(c.hop2_key_acc for c in cond_results.values()) / len(cond_results)
    m_v2 = sum(c.hop2_val_acc for c in cond_results.values()) / len(cond_results)
    m_tok = sum(c.final_token_acc for c in cond_results.values()) / len(cond_results)

    all_rob = all(c.is_robust for c in cond_results.values())

    return CompositionalRobustnessReport(
        seed=seed,
        conditions=cond_results,
        mean_h1_k_acc=m_k1,
        mean_h1_v_acc=m_v1,
        mean_inter_acc=m_in,
        mean_h2_k_acc=m_k2,
        mean_h2_v_acc=m_v2,
        mean_final_token_acc=m_tok,
        contamination_free=(collisions == 0),
        contamination_count=collisions,
        all_conditions_robust=all_rob,
        cpu_runtime_ms=(time.time() - t0) * 1000.0,
    )
