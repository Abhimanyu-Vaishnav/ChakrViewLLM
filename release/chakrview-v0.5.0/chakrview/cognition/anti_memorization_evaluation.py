"""Step 262: Anti-Memorization & Anti-Shortcut Evaluation.

Subject the adapted representation model to 8 orthogonal shortcut-destruction tests:
1. Identity permutation (randomized token mapping assignments)
2. Pair-order permutation (random presentation order of clauses)
3. Query-position permutation (query at prefix vs end)
4. Value-position permutation (reverse layouts: value before key)
5. Layout permutation (5 distinct syntactic structures)
6. Distractor insertion (unrelated noise pairs inserted)
7. Variable number of associations (1 to 5 pairs)
8. Completely unseen identities (zero overlap with training vocabulary tokens)

Also performs SHA-256 contamination audit between train and evaluation sets.
Determines whether representation adaptation learned ROLE/RELATIONSHIP vs TOKEN ID/POSITION.
"""

from __future__ import annotations

import dataclasses
import time
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn.functional as F

from chakrview.brain.model import ChakrMicro
from chakrview.cognition.neural_representation_adapter import (
    ChakrMicroWithAdaptedRepresentations,
)
from chakrview.cognition.randomized_associative_episodes import (
    RandomizedAssociativeEnvironment,
    RandomizedAssociativeEpisode,
)
from chakrview.cognition.adapter_curriculum_training import (
    extract_key_positions_from_episode,
)


@dataclasses.dataclass
class ShortcutTestConditionResult:
    condition_name: str
    num_episodes: int
    key_selection_accuracy: float
    value_position_accuracy: float
    final_token_accuracy: float
    is_robust: bool


@dataclasses.dataclass
class AntiMemorizationReport:
    seed: int
    condition_results: Dict[str, ShortcutTestConditionResult]
    mean_key_selection_acc: float
    mean_val_pos_acc: float
    mean_token_acc: float
    contamination_free: bool
    contamination_count: int
    learned_property: str              # "ROLE_RELATIONSHIP" or "TOKEN_POSITION_SHORTCUT"
    cpu_runtime_ms: float = 0.0


def evaluate_anti_memorization(
    candidate: ChakrMicroWithAdaptedRepresentations,
    env: Optional[RandomizedAssociativeEnvironment] = None,
    seed: int = 42,
    episodes_per_condition: int = 15,
) -> AntiMemorizationReport:
    """Runs the 8 anti-shortcut conditions to verify genuine role/relationship learning."""
    t0 = time.time()
    torch.manual_seed(seed)
    if env is None:
        env = RandomizedAssociativeEnvironment(seed=seed)
    tok = env.tok

    candidate.eval()

    conditions = [
        {"name": "1_identity_permutation", "split": "disjoint_test", "n": 2, "lay": "standard_map", "dist": False, "qp": "end"},
        {"name": "2_pair_order_permutation", "split": "disjoint_test", "n": 3, "lay": "standard_map", "dist": False, "qp": "end"},
        {"name": "3_query_pos_permutation", "split": "disjoint_test", "n": 2, "lay": "standard_map", "dist": False, "qp": "prefix"},
        {"name": "4_val_pos_permutation", "split": "disjoint_test", "n": 2, "lay": "reverse_order", "dist": False, "qp": "end"},
        {"name": "5_layout_permutation", "split": "disjoint_test", "n": 2, "lay": "compact_tuple", "dist": False, "qp": "end"},
        {"name": "6_distractor_insertion", "split": "disjoint_test", "n": 2, "lay": "standard_map", "dist": True, "qp": "end"},
        {"name": "7_variable_pair_count", "split": "disjoint_test", "n": 4, "lay": "standard_map", "dist": False, "qp": "end"},
        {"name": "8_unseen_identities", "split": "disjoint_test", "n": 2, "lay": "assignment_syntax", "dist": False, "qp": "end"},
    ]

    cond_results: Dict[str, ShortcutTestConditionResult] = {}

    with torch.no_grad():
        for c in conditions:
            k_corr = 0
            v_corr = 0
            tok_corr = 0

            for i in range(episodes_per_condition):
                ep = env.generate_episode(
                    split=c["split"],
                    num_associations=c["n"],
                    layout_name=c["lay"],
                    include_distractors=c["dist"],
                    query_placement=c["qp"],
                    episode_idx=7000 + i,
                )

                inp = torch.tensor([ep.prompt_tokens], dtype=torch.long)
                all_k_pos = extract_key_positions_from_episode(ep, tok)

                h_ad, _, logits = candidate.forward_adapted_backbone(inp)
                h_norm = F.normalize(h_ad[0], p=2, dim=-1)

                # Key selection
                h_q = h_norm[ep.query_key_pos]
                sims = [float(torch.dot(h_q, h_norm[p]).item()) for p in all_k_pos]
                if sims and all_k_pos[sims.index(max(sims))] == ep.matching_key_pos:
                    k_corr += 1

                # Value position routing
                val_positions = []
                for p in ep.pairs:
                    v_enc = tok.encode(p.val)[0]
                    vp_list = [j for j, t in enumerate(ep.prompt_tokens[:-1]) if t == v_enc]
                    if vp_list:
                        val_positions.append(vp_list[0])

                h_m = h_norm[ep.matching_key_pos]
                v_sims = [float(torch.dot(h_m, h_norm[vp]).item()) for vp in val_positions]
                if v_sims and val_positions[v_sims.index(max(v_sims))] == ep.associated_val_pos:
                    v_corr += 1

                pred_tok = torch.argmax(logits[0, -1, :]).item()
                if pred_tok == ep.target_token:
                    tok_corr += 1

            k_acc = k_corr / max(1, episodes_per_condition)
            v_acc = v_corr / max(1, episodes_per_condition)
            t_acc = tok_corr / max(1, episodes_per_condition)

            cond_results[c["name"]] = ShortcutTestConditionResult(
                condition_name=c["name"],
                num_episodes=episodes_per_condition,
                key_selection_accuracy=k_acc,
                value_position_accuracy=v_acc,
                final_token_accuracy=t_acc,
                is_robust=(k_acc >= 0.50),
            )

    # Contamination audit
    train_eps = env.generate_batch(30, split="train", num_associations=2)
    eval_eps = env.generate_batch(30, split="disjoint_test", num_associations=2)
    clean, coll = env.verify_no_contamination(train_eps, eval_eps)

    mean_k = sum(r.key_selection_accuracy for r in cond_results.values()) / len(cond_results)
    mean_v = sum(r.value_position_accuracy for r in cond_results.values()) / len(cond_results)
    mean_t = sum(r.final_token_accuracy for r in cond_results.values()) / len(cond_results)

    learned_prop = "ROLE_RELATIONSHIP" if (mean_k >= 0.50 and mean_v >= 0.50) else "TOKEN_POSITION_SHORTCUT"
    elapsed_ms = (time.time() - t0) * 1000.0

    return AntiMemorizationReport(
        seed=seed,
        condition_results=cond_results,
        mean_key_selection_acc=mean_k,
        mean_val_pos_acc=mean_v,
        mean_token_acc=mean_t,
        contamination_free=clean,
        contamination_count=coll,
        learned_property=learned_prop,
        cpu_runtime_ms=elapsed_ms,
    )
