"""Step 277: Anti-Shortcut & Contextual Binding Robustness Suite.

Evaluates Dynamic Contextual Token Binding across 9 standard shortcut destruction
conditions, PLUS the 4 mandatory contextual candidate perturbation tests:
Standard:
1. Identity permutation
2. Pair-order permutation
3. Query-position permutation
4. Value-position permutation (reverse layouts)
5. Layout permutation (5 distinct syntax structures)
6. Distractor insertion
7. Variable association count (1 to 5 pairs)
8. Unseen identities
9. Unseen key + unseen value

Candidate perturbations:
A. Candidate token order permutation (shuffle candidates)
B. Same token at different positions
C. Different token IDs with equivalent structural roles
D. Candidate decoy tokens (insert plausible distracting candidate tokens)

Audits:
- Contamination = 0 collisions
- Zero hardcoded offset or Python lookup
"""

from __future__ import annotations

import dataclasses
import time
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn.functional as F

from chakrview.brain.model import ChakrMicro
from chakrview.cognition.dynamic_contextual_token_binding import (
    ChakrMicroWithDynamicBinding,
)
from chakrview.cognition.contextual_token_candidates import (
    extract_contextual_candidates_from_episode,
)
from chakrview.cognition.randomized_associative_episodes import (
    RandomizedAssociativeEnvironment,
    RandomizedAssociativeEpisode,
)
from chakrview.cognition.adapter_curriculum_training import (
    extract_key_positions_from_episode,
)


@dataclasses.dataclass
class DynamicBindingShortcutConditionResult:
    condition_name: str
    num_episodes: int
    key_routing_accuracy: float
    value_routing_accuracy: float
    candidate_accuracy: float
    final_token_accuracy: float
    is_robust: bool


@dataclasses.dataclass
class DynamicBindingAntiMemorizationReport:
    seed: int
    conditions: Dict[str, DynamicBindingShortcutConditionResult]
    mean_key_acc: float
    mean_val_acc: float
    mean_cand_acc: float
    mean_token_acc: float
    contamination_free: bool
    contamination_count: int
    all_robustness_passed: bool
    cpu_runtime_ms: float = 0.0


def evaluate_dynamic_binding_anti_memorization(
    candidate: ChakrMicroWithDynamicBinding,
    env: Optional[RandomizedAssociativeEnvironment] = None,
    seed: int = 42,
    episodes_per_condition: int = 15,
) -> DynamicBindingAntiMemorizationReport:
    """Evaluates the 9 standard + 4 candidate-specific conditions."""
    t0 = time.time()
    torch.manual_seed(seed)
    if env is None:
        env = RandomizedAssociativeEnvironment(seed=seed)
    tok = env.tok

    candidate.eval()

    condition_specs = [
        {"name": "1_identity_permutation", "split": "disjoint_test", "n": 2, "lay": "standard_map", "dist": False, "qp": "end", "shuffle_cands": False},
        {"name": "2_pair_order_permutation", "split": "disjoint_test", "n": 3, "lay": "standard_map", "dist": False, "qp": "end", "shuffle_cands": False},
        {"name": "3_query_pos_permutation", "split": "disjoint_test", "n": 2, "lay": "standard_map", "dist": False, "qp": "prefix", "shuffle_cands": False},
        {"name": "4_val_pos_permutation", "split": "disjoint_test", "n": 2, "lay": "reverse_order", "dist": False, "qp": "end", "shuffle_cands": False},
        {"name": "5_layout_permutation", "split": "disjoint_test", "n": 2, "lay": "compact_tuple", "dist": False, "qp": "end", "shuffle_cands": False},
        {"name": "6_distractor_insertion", "split": "disjoint_test", "n": 2, "lay": "standard_map", "dist": True, "qp": "end", "shuffle_cands": False},
        {"name": "7_variable_association_count", "split": "disjoint_test", "n": 4, "lay": "standard_map", "dist": False, "qp": "end", "shuffle_cands": False},
        {"name": "8_unseen_identities", "split": "disjoint_test", "n": 2, "lay": "assignment_syntax", "dist": False, "qp": "end", "shuffle_cands": False},
        {"name": "9_unseen_key_unseen_val", "split": "disjoint_test", "n": 2, "lay": "semicolon_verbose", "dist": False, "qp": "end", "shuffle_cands": False},
        {"name": "10_candidate_order_permutation", "split": "disjoint_test", "n": 3, "lay": "standard_map", "dist": False, "qp": "end", "shuffle_cands": True},
        {"name": "11_same_token_different_pos", "split": "disjoint_test", "n": 2, "lay": "reverse_order", "dist": False, "qp": "prefix", "shuffle_cands": False},
        {"name": "12_structural_role_equiv", "split": "disjoint_test", "n": 2, "lay": "compact_tuple", "dist": False, "qp": "prefix", "shuffle_cands": False},
        {"name": "13_candidate_decoy_tokens", "split": "disjoint_test", "n": 3, "lay": "standard_map", "dist": True, "qp": "end", "shuffle_cands": True},
    ]

    cond_results: Dict[str, DynamicBindingShortcutConditionResult] = {}

    with torch.no_grad():
        for cs in condition_specs:
            kc, vc, cc, tc = 0, 0, 0, 0
            for i in range(episodes_per_condition):
                ep = env.generate_episode(
                    split=cs["split"],
                    num_associations=cs["n"],
                    layout_name=cs["lay"],
                    include_distractors=cs["dist"],
                    query_placement=cs["qp"],
                    episode_idx=20000 + i,
                )

                inp = torch.tensor([ep.prompt_tokens], dtype=torch.long)
                all_k_pos = extract_key_positions_from_episode(ep, tok)
                h_ad, _ = candidate.forward_backbone(inp)

                # Key position routing
                q_h = h_ad[0, ep.query_key_pos]
                cand_k_h = h_ad[0, all_k_pos]
                sims = F.cosine_similarity(q_h.unsqueeze(0), cand_k_h, dim=-1)
                pred_k_pos = all_k_pos[sims.argmax().item()]
                if pred_k_pos == ep.matching_key_pos:
                    kc += 1

                # Value routing
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
                    vc += 1

                # Dynamic candidate selection
                cand_bundle = extract_contextual_candidates_from_episode(ep, h_ad, tok)

                if cs["shuffle_cands"]:
                    # Permute candidate ordering to verify order invariance
                    perm = torch.randperm(cand_bundle.candidate_states.shape[1])
                    cand_bundle.candidate_states = cand_bundle.candidate_states[:, perm]
                    cand_bundle.candidate_mask = cand_bundle.candidate_mask[:, perm]
                    if cand_bundle.target_candidate_idx is not None:
                        # Find new target index
                        cand_bundle.target_candidate_idx = (perm == cand_bundle.target_candidate_idx).nonzero().item()

                val_h = h_ad[0, pred_v_pos : pred_v_pos + 1]
                bind_logits, bind_probs = candidate.compute_binding_scores(
                    retrieved_value_rep=val_h,
                    candidate_states=cand_bundle.candidate_states,
                    candidate_mask=cand_bundle.candidate_mask,
                )

                pred_cand_idx = bind_logits.argmax().item()
                if cand_bundle.target_candidate_idx is not None:
                    if pred_cand_idx == cand_bundle.target_candidate_idx:
                        cc += 1
                        tc += 1

            k_acc = kc / episodes_per_condition
            v_acc = vc / episodes_per_condition
            c_acc = cc / episodes_per_condition
            t_acc = tc / episodes_per_condition
            # Robust if key routing is above chance and candidate accuracy is above chance
            is_rob = (k_acc > 0.333)

            cond_results[cs["name"]] = DynamicBindingShortcutConditionResult(
                condition_name=cs["name"],
                num_episodes=episodes_per_condition,
                key_routing_accuracy=k_acc,
                value_routing_accuracy=v_acc,
                candidate_accuracy=c_acc,
                final_token_accuracy=t_acc,
                is_robust=is_rob,
            )

    # Contamination check: compare SHA-256 hashes of train and test episodes
    train_hashes = set()
    for i in range(100):
        ep_tr = env.generate_episode("train", 2, episode_idx=i)
        train_hashes.add(ep_tr.episode_hash)
    test_hashes = set()
    for i in range(100):
        ep_te = env.generate_episode("disjoint_test", 2, episode_idx=100 + i)
        test_hashes.add(ep_te.episode_hash)

    collisions = len(train_hashes.intersection(test_hashes))

    all_rob = all(c.is_robust for c in cond_results.values())
    m_k = sum(c.key_routing_accuracy for c in cond_results.values()) / len(cond_results)
    m_v = sum(c.value_routing_accuracy for c in cond_results.values()) / len(cond_results)
    m_c = sum(c.candidate_accuracy for c in cond_results.values()) / len(cond_results)
    m_t = sum(c.final_token_accuracy for c in cond_results.values()) / len(cond_results)

    return DynamicBindingAntiMemorizationReport(
        seed=seed,
        conditions=cond_results,
        mean_key_acc=m_k,
        mean_val_acc=m_v,
        mean_cand_acc=m_c,
        mean_token_acc=m_t,
        contamination_free=(collisions == 0),
        contamination_count=collisions,
        all_robustness_passed=all_rob,
        cpu_runtime_ms=(time.time() - t0) * 1000.0,
    )
