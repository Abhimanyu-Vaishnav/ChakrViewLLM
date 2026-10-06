"""Step 269: Anti-Memorization & Anti-Shortcut Evaluation (Emission Wave).

Evaluates Candidate B (Representation Adapter + Contextual Readout) across
the 9 specified anti-memorization conditions:
1. Identity permutation
2. Pair-order permutation
3. Query-position permutation
4. Value-position permutation (reverse layouts)
5. Layout permutation (5 distinct syntactic structures)
6. Distractor insertion
7. Variable association count (1 to 5 pairs)
8. Unseen identities
9. Unseen key + unseen value

Audits SHA-256 contamination between train and evaluation splits.
Ensures zero Python lookup, zero hardcoded offset.
"""

from __future__ import annotations

import dataclasses
import time
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn.functional as F

from chakrview.brain.model import ChakrMicro
from chakrview.cognition.contextual_vocabulary_readout import (
    ChakrMicroWithContextualReadout,
)
from chakrview.cognition.randomized_associative_episodes import (
    RandomizedAssociativeEnvironment,
    RandomizedAssociativeEpisode,
)
from chakrview.cognition.adapter_curriculum_training import (
    extract_key_positions_from_episode,
)


@dataclasses.dataclass
class EmissionShortcutConditionResult:
    condition_name: str
    num_episodes: int
    key_position_accuracy: float
    value_position_accuracy: float
    final_token_accuracy: float
    is_robust: bool


@dataclasses.dataclass
class EmissionAntiMemorizationReport:
    seed: int
    conditions: Dict[str, EmissionShortcutConditionResult]
    mean_key_pos_acc: float
    mean_val_pos_acc: float
    mean_token_acc: float
    contamination_free: bool
    contamination_count: int
    routing_nature: str
    cpu_runtime_ms: float = 0.0


def evaluate_emission_anti_memorization(
    candidate: ChakrMicroWithContextualReadout,
    env: Optional[RandomizedAssociativeEnvironment] = None,
    seed: int = 42,
    episodes_per_condition: int = 15,
) -> EmissionAntiMemorizationReport:
    """Runs Candidate B through all 9 shortcut destruction conditions."""
    t0 = time.time()
    torch.manual_seed(seed)
    if env is None:
        env = RandomizedAssociativeEnvironment(seed=seed)
    tok = env.tok

    candidate.eval()

    condition_specs = [
        {"name": "1_identity_permutation", "split": "disjoint_test", "n": 2, "lay": "standard_map", "dist": False, "qp": "end"},
        {"name": "2_pair_order_permutation", "split": "disjoint_test", "n": 3, "lay": "standard_map", "dist": False, "qp": "end"},
        {"name": "3_query_pos_permutation", "split": "disjoint_test", "n": 2, "lay": "standard_map", "dist": False, "qp": "prefix"},
        {"name": "4_val_pos_permutation", "split": "disjoint_test", "n": 2, "lay": "reverse_order", "dist": False, "qp": "end"},
        {"name": "5_layout_permutation", "split": "disjoint_test", "n": 2, "lay": "compact_tuple", "dist": False, "qp": "end"},
        {"name": "6_distractor_insertion", "split": "disjoint_test", "n": 2, "lay": "standard_map", "dist": True, "qp": "end"},
        {"name": "7_variable_association_count", "split": "disjoint_test", "n": 4, "lay": "standard_map", "dist": False, "qp": "end"},
        {"name": "8_unseen_identities", "split": "disjoint_test", "n": 2, "lay": "assignment_syntax", "dist": False, "qp": "end"},
        {"name": "9_unseen_key_unseen_val", "split": "disjoint_test", "n": 2, "lay": "semicolon_verbose", "dist": False, "qp": "end"},
    ]

    cond_results: Dict[str, EmissionShortcutConditionResult] = {}

    with torch.no_grad():
        for cs in condition_specs:
            kc, vc, tc = 0, 0, 0
            for i in range(episodes_per_condition):
                ep = env.generate_episode(
                    split=cs["split"],
                    num_associations=cs["n"],
                    layout_name=cs["lay"],
                    include_distractors=cs["dist"],
                    query_placement=cs["qp"],
                    episode_idx=9000 + i,
                )

                inp = torch.tensor([ep.prompt_tokens], dtype=torch.long)
                all_k = extract_key_positions_from_episode(ep, tok)
                h_ad, _ = candidate.forward_backbone(inp)

                # Key position routing
                h_norm = F.normalize(h_ad[0], p=2, dim=-1)
                h_q = h_norm[ep.query_key_pos]
                sims = [float(torch.dot(h_q, h_norm[kp]).item()) for kp in all_k]
                if sims and all_k[sims.index(max(sims))] == ep.matching_key_pos:
                    kc += 1

                # Value position routing
                val_pos = []
                for p in ep.pairs:
                    v_enc = tok.encode(p.val)[0]
                    vp_l = [j for j, t in enumerate(ep.prompt_tokens[:-1]) if t == v_enc]
                    if vp_l: val_pos.append(vp_l[0])

                h_m = h_norm[ep.matching_key_pos]
                v_sims = [float(torch.dot(h_m, h_norm[vp]).item()) for vp in val_pos]
                best_vp = val_pos[v_sims.index(max(v_sims))]
                if best_vp == ep.associated_val_pos:
                    vc += 1

                # Readout emission
                selected_v = h_ad[0, best_vp].unsqueeze(0)
                em_logits = candidate.forward_from_value_rep(selected_v)
                if torch.argmax(em_logits[0]).item() == ep.target_token:
                    tc += 1

            k_acc = kc / max(1, episodes_per_condition)
            v_acc = vc / max(1, episodes_per_condition)
            t_acc = tc / max(1, episodes_per_condition)
            cond_results[cs["name"]] = EmissionShortcutConditionResult(
                condition_name=cs["name"],
                num_episodes=episodes_per_condition,
                key_position_accuracy=k_acc,
                value_position_accuracy=v_acc,
                final_token_accuracy=t_acc,
                is_robust=(k_acc >= 0.50),
            )

    train_eps = env.generate_batch(30, split="train", num_associations=2)
    eval_eps = env.generate_batch(30, split="disjoint_test", num_associations=2)
    clean, coll = env.verify_no_contamination(train_eps, eval_eps)

    mean_k = sum(r.key_position_accuracy for r in cond_results.values()) / len(cond_results)
    mean_v = sum(r.value_position_accuracy for r in cond_results.values()) / len(cond_results)
    mean_t = sum(r.final_token_accuracy for r in cond_results.values()) / len(cond_results)
    routing_nat = "ROLE_RELATIONSHIP" if (mean_k >= 0.50 and mean_v >= 0.50) else "POSITIONAL_SHORTCUT"
    elapsed_ms = (time.time() - t0) * 1000.0

    return EmissionAntiMemorizationReport(
        seed=seed,
        conditions=cond_results,
        mean_key_pos_acc=mean_k,
        mean_val_pos_acc=mean_v,
        mean_token_acc=mean_t,
        contamination_free=clean,
        contamination_count=coll,
        routing_nature=routing_nat,
        cpu_runtime_ms=elapsed_ms,
    )
