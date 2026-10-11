"""Step 260: Representation Generalization Test.

Diagnostic evaluation measuring representation alignment and separation across 4 splits:
1. Known key / Known value
2. Known key / Unseen value
3. Unseen key / Known value
4. Unseen key / Unseen value

For each episode measures:
1. query -> correct key similarity
2. query -> incorrect key similarity
3. key-selection accuracy (via cosine similarity)
4. selected key representation norm / stats
5. selected key -> correct value-position score
6. selected key -> incorrect value-position score
7. value-position accuracy
8. final token accuracy

Determines empirically:
Does the representation adapter improve unseen query -> unseen key
without simply memorizing identities?
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
class SplitRepresentationMetrics:
    split_name: str
    num_episodes: int
    mean_correct_key_sim: float
    mean_incorrect_key_sim: float
    key_separation_margin: float
    key_selection_accuracy: float
    mean_correct_val_sim: float
    mean_incorrect_val_sim: float
    val_separation_margin: float
    val_position_accuracy: float
    final_token_accuracy: float
    mean_target_rank: float


@dataclasses.dataclass
class RepresentationGeneralizationReport:
    seed: int
    adapter_type: str
    splits: Dict[str, SplitRepresentationMetrics]
    unseen_unseen_key_acc: float
    unseen_unseen_val_acc: float
    unseen_unseen_tok_acc: float
    is_generalization_successful: bool
    summary: str
    cpu_runtime_ms: float = 0.0


def evaluate_representation_generalization(
    candidate: ChakrMicroWithAdaptedRepresentations,
    env: Optional[RandomizedAssociativeEnvironment] = None,
    seed: int = 42,
    num_episodes_per_split: int = 20,
) -> RepresentationGeneralizationReport:
    """Evaluates representation geometry across the four orthogonal splits."""
    t0 = time.time()
    torch.manual_seed(seed)
    if env is None:
        env = RandomizedAssociativeEnvironment(seed=seed)
    tok = env.tok

    candidate.eval()

    splits_to_test = [
        ("known_known", "train"),
        ("known_unseen", "known_unseen"),
        ("unseen_known", "unseen_known"),
        ("unseen_unseen", "disjoint_test"),
    ]

    split_results: Dict[str, SplitRepresentationMetrics] = {}

    with torch.no_grad():
        for sp_key, sp_env in splits_to_test:
            corr_k_sims = []
            incorr_k_sims = []
            k_correct = 0

            corr_v_sims = []
            incorr_v_sims = []
            v_correct = 0

            tok_correct = 0
            ranks = []

            for i in range(num_episodes_per_split):
                ep = env.generate_episode(
                    split=sp_env,
                    num_associations=2,
                    layout_name="standard_map",
                    include_distractors=False,
                    episode_idx=6000 + i,
                )

                inp = torch.tensor([ep.prompt_tokens], dtype=torch.long)
                all_k_pos = extract_key_positions_from_episode(ep, tok)

                h_ad, _, logits = candidate.forward_adapted_backbone(inp)
                h_norm = F.normalize(h_ad[0], p=2, dim=-1)

                # 1. Query-Key matching
                h_q = h_norm[ep.query_key_pos]
                h_m = h_norm[ep.matching_key_pos]

                pos_k_sim = float(torch.dot(h_q, h_m).item())
                corr_k_sims.append(pos_k_sim)

                neg_k_sims = []
                for kp in all_k_pos:
                    if kp != ep.matching_key_pos:
                        neg_k_sims.append(float(torch.dot(h_q, h_norm[kp]).item()))

                mean_neg_k = (sum(neg_k_sims) / len(neg_k_sims)) if neg_k_sims else 0.0
                incorr_k_sims.append(mean_neg_k)

                all_sims = [float(torch.dot(h_q, h_norm[p]).item()) for p in all_k_pos]
                if all_sims and all_k_pos[all_sims.index(max(all_sims))] == ep.matching_key_pos:
                    k_correct += 1

                # 2. Key->Value routing from matching key
                h_v = h_norm[ep.associated_val_pos]
                pos_v_sim = float(torch.dot(h_m, h_v).item())
                corr_v_sims.append(pos_v_sim)

                # Incorrect value positions across candidate value tokens
                # All values in prompt
                val_positions = []
                for p in ep.pairs:
                    v_enc = tok.encode(p.val)[0]
                    vp_list = [j for j, t in enumerate(ep.prompt_tokens[:-1]) if t == v_enc]
                    if vp_list:
                        val_positions.append(vp_list[0])

                neg_v_sims = []
                for vp in val_positions:
                    if vp != ep.associated_val_pos:
                        neg_v_sims.append(float(torch.dot(h_m, h_norm[vp]).item()))
                mean_neg_v = (sum(neg_v_sims) / len(neg_v_sims)) if neg_v_sims else 0.0
                incorr_v_sims.append(mean_neg_v)

                all_v_sims = [float(torch.dot(h_m, h_norm[p]).item()) for p in val_positions]
                if all_v_sims and val_positions[all_v_sims.index(max(all_v_sims))] == ep.associated_val_pos:
                    v_correct += 1

                # 3. Final token output
                pred_tok = torch.argmax(logits[0, -1, :]).item()
                if pred_tok == ep.target_token:
                    tok_correct += 1

                last_l = logits[0, -1, :]
                rk = int((last_l > last_l[ep.target_token]).sum().item()) + 1
                ranks.append(rk)

            mean_corr_k = sum(corr_k_sims) / max(1, len(corr_k_sims))
            mean_incorr_k = sum(incorr_k_sims) / max(1, len(incorr_k_sims))
            margin_k = mean_corr_k - mean_incorr_k

            mean_corr_v = sum(corr_v_sims) / max(1, len(corr_v_sims))
            mean_incorr_v = sum(incorr_v_sims) / max(1, len(incorr_v_sims))
            margin_v = mean_corr_v - mean_incorr_v

            split_results[sp_key] = SplitRepresentationMetrics(
                split_name=sp_key,
                num_episodes=num_episodes_per_split,
                mean_correct_key_sim=mean_corr_k,
                mean_incorrect_key_sim=mean_incorr_k,
                key_separation_margin=margin_k,
                key_selection_accuracy=k_correct / max(1, num_episodes_per_split),
                mean_correct_val_sim=mean_corr_v,
                mean_incorrect_val_sim=mean_incorr_v,
                val_separation_margin=margin_v,
                val_position_accuracy=v_correct / max(1, num_episodes_per_split),
                final_token_accuracy=tok_correct / max(1, num_episodes_per_split),
                mean_target_rank=sum(ranks) / max(1, len(ranks)),
            )

    uu = split_results["unseen_unseen"]
    is_success = (uu.key_selection_accuracy >= 0.50 and uu.val_position_accuracy >= 0.50 and uu.final_token_accuracy >= 0.50)

    summary = (
        f"Step 260 Representation Generalization: "
        f"UU Key Acc={uu.key_selection_accuracy:.2%}, Key Margin=+{uu.key_separation_margin:.4f}; "
        f"UU Val Acc={uu.val_position_accuracy:.2%}, Val Margin=+{uu.val_separation_margin:.4f}; "
        f"UU Token Acc={uu.final_token_accuracy:.2%}. Success={is_success}."
    )

    elapsed_ms = (time.time() - t0) * 1000.0

    return RepresentationGeneralizationReport(
        seed=seed,
        adapter_type=candidate.adapter_type,
        splits=split_results,
        unseen_unseen_key_acc=uu.key_selection_accuracy,
        unseen_unseen_val_acc=uu.val_position_accuracy,
        unseen_unseen_tok_acc=uu.final_token_accuracy,
        is_generalization_successful=is_success,
        summary=summary,
        cpu_runtime_ms=elapsed_ms,
    )
