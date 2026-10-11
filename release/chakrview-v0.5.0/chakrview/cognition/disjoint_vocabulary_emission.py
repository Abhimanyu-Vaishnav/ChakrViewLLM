"""Step 267: Disjoint Vocabulary Emission Evaluation.

Evaluates contextual vocabulary emission across the four orthogonal splits:
A. known -> known (in-distribution)
B. known -> unseen (unseen value)
C. unseen -> known (unseen key)
D. unseen -> unseen (primary I3 capability gate)

For each condition reports:
- Key-position accuracy
- Value-position accuracy
- Final token accuracy
- Target token probability
- Target token rank
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
class DisjointEmissionConditionMetrics:
    condition_name: str
    num_episodes: int
    key_position_accuracy: float
    value_position_accuracy: float
    final_token_accuracy: float
    mean_target_prob: float
    mean_target_rank: float


@dataclasses.dataclass
class DisjointVocabularyEmissionReport:
    seed: int
    conditions: Dict[str, DisjointEmissionConditionMetrics]
    unseen_unseen_key_acc: float
    unseen_unseen_val_acc: float
    unseen_unseen_tok_acc: float
    is_i3_gate_satisfied: bool
    summary: str
    cpu_runtime_ms: float = 0.0


def evaluate_disjoint_vocabulary_emission(
    candidate: ChakrMicroWithContextualReadout,
    env: Optional[RandomizedAssociativeEnvironment] = None,
    seed: int = 42,
    num_episodes_per_condition: int = 20,
) -> DisjointVocabularyEmissionReport:
    """Evaluates the 4 disjoint emission splits."""
    t0 = time.time()
    torch.manual_seed(seed)
    if env is None:
        env = RandomizedAssociativeEnvironment(seed=seed)
    tok = env.tok

    candidate.eval()

    splits = [
        ("A_known_known", "train"),
        ("B_known_unseen", "known_unseen"),
        ("C_unseen_known", "unseen_known"),
        ("D_unseen_unseen", "disjoint_test"),
    ]

    cond_results: Dict[str, DisjointEmissionConditionMetrics] = {}

    with torch.no_grad():
        for cond_name, sp_env in splits:
            k_corr, v_corr, t_corr = 0, 0, 0
            probs, ranks = [], []

            for i in range(num_episodes_per_condition):
                ep = env.generate_episode(
                    split=sp_env,
                    num_associations=2,
                    layout_name="standard_map",
                    include_distractors=False,
                    episode_idx=8000 + i,
                )

                inp = torch.tensor([ep.prompt_tokens], dtype=torch.long)
                all_k_pos = extract_key_positions_from_episode(ep, tok)
                adapted_hidden, _ = candidate.forward_backbone(inp)

                # Key selection
                h_norm = F.normalize(adapted_hidden[0], p=2, dim=-1)
                h_q = h_norm[ep.query_key_pos]
                k_sims = [float(torch.dot(h_q, h_norm[kp]).item()) for kp in all_k_pos]
                if k_sims and all_k_pos[k_sims.index(max(k_sims))] == ep.matching_key_pos:
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
                best_vp = val_positions[v_sims.index(max(v_sims))]
                if best_vp == ep.associated_val_pos:
                    v_corr += 1

                # Forward through Contextual Vocabulary Readout
                selected_v_state = adapted_hidden[0, best_vp].unsqueeze(0)
                em_logits = candidate.forward_from_value_rep(selected_v_state)
                tgt_tok = ep.target_token

                pred_tok = torch.argmax(em_logits[0]).item()
                if pred_tok == tgt_tok:
                    t_corr += 1

                em_probs = F.softmax(em_logits[0], dim=-1)
                probs.append(float(em_probs[tgt_tok].item()))
                rk = int((em_logits[0] > em_logits[0, tgt_tok]).sum().item()) + 1
                ranks.append(rk)

            cond_results[cond_name] = DisjointEmissionConditionMetrics(
                condition_name=cond_name,
                num_episodes=num_episodes_per_condition,
                key_position_accuracy=k_corr / max(1, num_episodes_per_condition),
                value_position_accuracy=v_corr / max(1, num_episodes_per_condition),
                final_token_accuracy=t_corr / max(1, num_episodes_per_condition),
                mean_target_prob=sum(probs) / max(1, len(probs)),
                mean_target_rank=sum(ranks) / max(1, len(ranks)),
            )

    uu = cond_results["D_unseen_unseen"]
    i3_pass = (uu.final_token_accuracy >= 0.50 and uu.key_position_accuracy > 0.33 and uu.value_position_accuracy > 0.33)

    summary = (
        f"Step 267 Disjoint Emission: UU Key Acc={uu.key_position_accuracy:.2%}, "
        f"UU Val Acc={uu.value_position_accuracy:.2%}, "
        f"UU Token Acc={uu.final_token_accuracy:.2%}, "
        f"Mean Target Prob={uu.mean_target_prob:.4f}, Mean Target Rank={uu.mean_target_rank:.1f}. "
        f"I3 Gate Satisfied={i3_pass}."
    )

    elapsed_ms = (time.time() - t0) * 1000.0

    return DisjointVocabularyEmissionReport(
        seed=seed,
        conditions=cond_results,
        unseen_unseen_key_acc=uu.key_position_accuracy,
        unseen_unseen_val_acc=uu.value_position_accuracy,
        unseen_unseen_tok_acc=uu.final_token_accuracy,
        is_i3_gate_satisfied=i3_pass,
        summary=summary,
        cpu_runtime_ms=elapsed_ms,
    )
