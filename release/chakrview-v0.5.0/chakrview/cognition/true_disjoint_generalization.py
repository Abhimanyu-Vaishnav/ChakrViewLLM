"""Step 276: True Disjoint Generalization Evaluation.

Evaluates Dynamic Contextual Token Binding across four orthogonal conditions:
A. known key / known value (in-distribution)
B. known key / unseen value (unseen value identities)
C. unseen key / known value (unseen key identities)
D. unseen key / unseen value (PRIMARY I3 CAPABILITY GATE)

Reports:
- Key-position routing accuracy
- Value-position routing accuracy
- Candidate-token selection accuracy
- Final token accuracy
- Target probability & target rank
- Direct comparison with Wave 265-272 static readout (UU token = 8.33%)
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
class DisjointBindingConditionMetrics:
    condition_name: str
    num_episodes: int
    key_routing_accuracy: float
    value_routing_accuracy: float
    candidate_token_accuracy: float
    final_token_accuracy: float
    mean_target_prob: float
    mean_target_rank: float


@dataclasses.dataclass
class DisjointDynamicBindingReport:
    seed: int
    conditions: Dict[str, DisjointBindingConditionMetrics]
    unseen_unseen_key_acc: float
    unseen_unseen_val_acc: float
    unseen_unseen_cand_acc: float
    unseen_unseen_tok_acc: float
    static_readout_baseline_tok_acc: float  # 0.0833
    beats_static_readout: bool
    is_i3_gate_satisfied: bool
    summary: str
    cpu_runtime_ms: float = 0.0


def evaluate_true_disjoint_binding(
    candidate: ChakrMicroWithDynamicBinding,
    env: Optional[RandomizedAssociativeEnvironment] = None,
    seed: int = 42,
    num_episodes_per_condition: int = 20,
) -> DisjointDynamicBindingReport:
    """Evaluates the 4 disjoint binding splits A, B, C, D."""
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

    cond_results: Dict[str, DisjointBindingConditionMetrics] = {}

    with torch.no_grad():
        for cond_name, sp_env in splits:
            k_corr, v_corr, c_corr, t_corr = 0, 0, 0, 0
            probs, ranks = [], []

            for i in range(num_episodes_per_condition):
                ep = env.generate_episode(
                    split=sp_env,
                    num_associations=2,
                    layout_name="standard_map",
                    include_distractors=False,
                    episode_idx=15000 + i,
                )

                inp = torch.tensor([ep.prompt_tokens], dtype=torch.long)
                all_k_pos = extract_key_positions_from_episode(ep, tok)
                h_ad, _ = candidate.forward_backbone(inp)

                # Key routing
                q_h = h_ad[0, ep.query_key_pos]
                cand_k_h = h_ad[0, all_k_pos]
                sims = F.cosine_similarity(q_h.unsqueeze(0), cand_k_h, dim=-1)
                pred_k_pos = all_k_pos[sims.argmax().item()]
                if pred_k_pos == ep.matching_key_pos:
                    k_corr += 1

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
                    v_corr += 1

                # Dynamic candidate selection
                cand_bundle = extract_contextual_candidates_from_episode(ep, h_ad, tok)
                val_h = h_ad[0, pred_v_pos : pred_v_pos + 1]

                bind_logits, bind_probs = candidate.compute_binding_scores(
                    retrieved_value_rep=val_h,
                    candidate_states=cand_bundle.candidate_states,
                    candidate_mask=cand_bundle.candidate_mask,
                )

                pred_cand_idx = bind_logits.argmax().item()
                if cand_bundle.target_candidate_idx is not None:
                    if pred_cand_idx == cand_bundle.target_candidate_idx:
                        c_corr += 1
                        t_corr += 1
                    t_prob = bind_probs[0, cand_bundle.target_candidate_idx].item()
                    probs.append(t_prob)
                    sorted_idxs = torch.argsort(bind_logits[0], descending=True)
                    t_rank = (sorted_idxs == cand_bundle.target_candidate_idx).nonzero().item() + 1
                    ranks.append(t_rank)
                else:
                    probs.append(0.0)
                    ranks.append(len(cand_bundle.candidate_positions))

            cond_results[cond_name] = DisjointBindingConditionMetrics(
                condition_name=cond_name,
                num_episodes=num_episodes_per_condition,
                key_routing_accuracy=k_corr / num_episodes_per_condition,
                value_routing_accuracy=v_corr / num_episodes_per_condition,
                candidate_token_accuracy=c_corr / num_episodes_per_condition,
                final_token_accuracy=t_corr / num_episodes_per_condition,
                mean_target_prob=sum(probs) / max(1, len(probs)),
                mean_target_rank=sum(ranks) / max(1, len(ranks)),
            )

    uu_m = cond_results["D_unseen_unseen"]
    uu_key = uu_m.key_routing_accuracy
    uu_val = uu_m.value_routing_accuracy
    uu_cand = uu_m.candidate_token_accuracy
    uu_tok = uu_m.final_token_accuracy

    static_baseline = 0.0833
    beats_static = uu_tok > static_baseline
    i3_gate = (uu_tok >= 0.50 and uu_key > 0.333 and uu_val > 0.333 and uu_cand > 0.333)

    return DisjointDynamicBindingReport(
        seed=seed,
        conditions=cond_results,
        unseen_unseen_key_acc=uu_key,
        unseen_unseen_val_acc=uu_val,
        unseen_unseen_cand_acc=uu_cand,
        unseen_unseen_tok_acc=uu_tok,
        static_readout_baseline_tok_acc=static_baseline,
        beats_static_readout=beats_static,
        is_i3_gate_satisfied=i3_gate,
        summary=f"UU Key: {uu_key*100:.1f}%, UU Val: {uu_val*100:.1f}%, UU Cand: {uu_cand*100:.1f}%, UU Tok: {uu_tok*100:.1f}% (Static Readout: {static_baseline*100:.2f}%)",
        cpu_runtime_ms=(time.time() - t0) * 1000.0,
    )
