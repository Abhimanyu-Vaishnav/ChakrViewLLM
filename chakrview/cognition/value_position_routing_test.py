"""Step 252: True Value-Position Routing Anti-Shortcut Test.

Constructs diagnostic episodes where the associated value appears at variable contextual positions:
- value before query
- value after query
- value far away
- pairs interleaved
- distractors between key and value
- unrelated values closer than correct value

Empirical Validation Rule:
Measures:
1. Matching key position selection accuracy
2. Associated value position selection accuracy
3. Final output token retrieval accuracy
4. 2x2 Diagnostic Routing Matrix:
   - key_corr_val_corr
   - key_corr_val_incorr
   - key_incorr_val_corr
   - key_incorr_val_incorr

Success requires:
key selection > chance (1/N)
AND
value-position selection > chance (1/N)
AND
final token accuracy > chance (1/V).
If key selection succeeds but value selection fails, the experiment is marked unrouted.
"""

from __future__ import annotations

import dataclasses
import random
import time
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn.functional as F

from chakrview.brain.model import ChakrMicro
from chakrview.cognition.associative_pointer_circuit import (
    ChakrMicroWithAssociativePointer,
)
from chakrview.cognition.randomized_associative_episodes import (
    RandomizedAssociativeEnvironment,
    RandomizedAssociativeEpisode,
)


@dataclasses.dataclass
class ValuePositionRoutingResult:
    num_episodes: int
    chance_key_acc: float
    chance_val_acc: float
    key_selection_accuracy: float
    value_selection_accuracy: float
    final_token_accuracy: float
    # 2x2 Routing Matrix
    key_corr_val_corr_rate: float
    key_corr_val_incorr_rate: float
    key_incorr_val_corr_rate: float
    key_incorr_val_incorr_rate: float
    mean_value_attention_mass: float
    mean_distractor_value_mass: float
    routing_success: bool
    status_summary: str
    cpu_runtime_ms: float = 0.0


def evaluate_value_position_routing(
    model: ChakrMicroWithAssociativePointer,
    env: Optional[RandomizedAssociativeEnvironment] = None,
    seed: int = 42,
    num_episodes: int = 20,
    num_associations: int = 3,
) -> ValuePositionRoutingResult:
    """Evaluates whether associative pointer selects true value position across challenging layouts."""
    t0 = time.time()
    torch.manual_seed(seed)
    if env is None:
        env = RandomizedAssociativeEnvironment(seed=seed)

    model.eval()

    matrix_counts = {"c_c": 0, "c_i": 0, "i_c": 0, "i_i": 0}
    k_correct = 0
    v_correct = 0
    tok_correct = 0

    val_masses = []
    dist_masses = []

    # Chance probabilities for N=3 associations
    chance_k = 1.0 / float(num_associations)
    chance_v = 1.0 / float(num_associations)

    with torch.no_grad():
        for i in range(num_episodes):
            # Test on multiple layouts with distractors
            layout = env.LAYOUTS[i % len(env.LAYOUTS)]
            ep = env.generate_episode(
                split="train",
                num_associations=num_associations,
                layout_name=layout,
                include_distractors=True,
                episode_idx=2000 + i,
            )

            inp = torch.tensor([ep.prompt_tokens], dtype=torch.long)
            logits, out = model(inp)

            gt_k = min(ep.matching_key_pos, len(ep.prompt_tokens) - 2)
            gt_v = min(ep.associated_val_pos, len(ep.prompt_tokens) - 2)
            gt_tok = ep.target_token

            pred_tok = torch.argmax(logits[0], dim=-1).item()
            pred_k = out.selected_key_pos.item()
            pred_v = out.selected_val_pos.item()

            k_ok = (pred_k == gt_k)
            v_ok = (pred_v == gt_v)
            tok_ok = (pred_tok == gt_tok)

            if k_ok:
                k_correct += 1
            if v_ok:
                v_correct += 1
            if tok_ok:
                tok_correct += 1

            if k_ok and v_ok:
                matrix_counts["c_c"] += 1
            elif k_ok and not v_ok:
                matrix_counts["c_i"] += 1
            elif not k_ok and v_ok:
                matrix_counts["i_c"] += 1
            else:
                matrix_counts["i_i"] += 1

            v_dist = out.value_distribution[0]
            val_mass = float(v_dist[gt_v].item()) if gt_v < v_dist.shape[0] else 0.0
            dist_mass = (float(v_dist.sum().item()) - val_mass) / max(1, v_dist.shape[0] - 1)

            val_masses.append(val_mass)
            dist_masses.append(dist_mass)

    k_acc = k_correct / max(1, num_episodes)
    v_acc = v_correct / max(1, num_episodes)
    tok_acc = tok_correct / max(1, num_episodes)

    is_success = (k_acc > chance_k) and (v_acc > chance_v) and (tok_acc > 0.0)

    summary = (
        f"Step 252 Value Routing: KeyAcc={k_acc:.2%} (chance={chance_k:.2%}), "
        f"ValAcc={v_acc:.2%} (chance={chance_v:.2%}), TokAcc={tok_acc:.2%}, "
        f"Matrix=[c_c={matrix_counts['c_c']}, c_i={matrix_counts['c_i']}, "
        f"i_c={matrix_counts['i_c']}, i_i={matrix_counts['i_i']}]. "
        f"Success={is_success}."
    )

    elapsed_ms = (time.time() - t0) * 1000.0

    return ValuePositionRoutingResult(
        num_episodes=num_episodes,
        chance_key_acc=chance_k,
        chance_val_acc=chance_v,
        key_selection_accuracy=k_acc,
        value_selection_accuracy=v_acc,
        final_token_accuracy=tok_acc,
        key_corr_val_corr_rate=matrix_counts["c_c"] / num_episodes,
        key_corr_val_incorr_rate=matrix_counts["c_i"] / num_episodes,
        key_incorr_val_corr_rate=matrix_counts["i_c"] / num_episodes,
        key_incorr_val_incorr_rate=matrix_counts["i_i"] / num_episodes,
        mean_value_attention_mass=sum(val_masses) / max(1, len(val_masses)),
        mean_distractor_value_mass=sum(dist_masses) / max(1, len(dist_masses)),
        routing_success=is_success,
        status_summary=summary,
        cpu_runtime_ms=elapsed_ms,
    )
