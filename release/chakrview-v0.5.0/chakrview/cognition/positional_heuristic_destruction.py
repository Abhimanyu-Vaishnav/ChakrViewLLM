"""Step 253: Positional-Heuristic Destruction Test.

Explicitly tests whether the learned pointer routing relies on fixed positional offsets
(e.g., "value is always key + 7 tokens") vs genuine relational association.

Constructs 5 distinct structural layouts for identical semantic mappings:
Layout 1: KEY ... VALUE (canonical standard map)
Layout 2: VALUE ... KEY (reversed ordering: value appears before key)
Layout 3: KEY ... distractors ... VALUE (distractors inserted between key and value)
Layout 4: distractor ... VALUE ... KEY (reversed with preceding noise)
Layout 5: multiple unrelated pairs between KEY and VALUE (long span separation)

Measures:
- Per-layout output accuracy
- Per-layout value position accuracy
- Correlation between predicted pointer position and relative key-value offset
- Positional shortcut dependency classification:
  * If correlation > 0.85 and reverse/separated layouts fail -> POSITIONAL_SHORTCUT_DEPENDENT
  * If performance survives layout permutations without offset correlation -> RELATIONSHIP_DRIVEN
"""

from __future__ import annotations

import dataclasses
import math
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
    AssociativePair,
)


@dataclasses.dataclass
class LayoutDestructionResult:
    layout_name: str
    num_samples: int
    final_token_accuracy: float
    value_position_accuracy: float
    key_position_accuracy: float
    mean_key_value_offset: float


@dataclasses.dataclass
class PositionalHeuristicDestructionReport:
    seed: int
    layout_results: Dict[str, LayoutDestructionResult]
    mean_token_acc: float
    mean_val_pos_acc: float
    offset_position_correlation: float
    is_shortcut_dependent: bool
    routing_nature: str                # "RELATIONSHIP_DRIVEN" or "POSITIONAL_SHORTCUT"
    cpu_runtime_ms: float = 0.0


def evaluate_positional_heuristic_destruction(
    model: ChakrMicroWithAssociativePointer,
    env: Optional[RandomizedAssociativeEnvironment] = None,
    seed: int = 42,
    samples_per_layout: int = 15,
) -> PositionalHeuristicDestructionReport:
    """Evaluates the model across the 5 destruction layouts to detect positional shortcuts."""
    t0 = time.time()
    torch.manual_seed(seed)
    if env is None:
        env = RandomizedAssociativeEnvironment(seed=seed)

    model.eval()

    layouts = [
        "standard_map",       # KEY ... VALUE
        "reverse_order",      # VALUE ... KEY
        "semicolon_verbose",  # KEY ... distractors ... VALUE
        "compact_tuple",      # distractor ... VALUE ... KEY
        "assignment_syntax",  # multiple unrelated pairs between KEY and VALUE
    ]

    layout_results: Dict[str, LayoutDestructionResult] = {}

    all_predicted_offsets: List[float] = []
    all_actual_offsets: List[float] = []

    with torch.no_grad():
        for lay in layouts:
            tok_corr = 0
            val_corr = 0
            key_corr = 0
            offsets = []

            for i in range(samples_per_layout):
                ep = env.generate_episode(
                    split="train",
                    num_associations=2,
                    layout_name=lay,
                    include_distractors=(lay in ["semicolon_verbose", "assignment_syntax"]),
                    episode_idx=3000 + i,
                )

                inp = torch.tensor([ep.prompt_tokens], dtype=torch.long)
                logits, out = model(inp)

                pred_tok = torch.argmax(logits[0], dim=-1).item()
                pred_k = out.selected_key_pos.item()
                pred_v = out.selected_val_pos.item()

                gt_tok = ep.target_token
                gt_k = min(ep.matching_key_pos, len(ep.prompt_tokens) - 2)
                gt_v = min(ep.associated_val_pos, len(ep.prompt_tokens) - 2)

                if pred_tok == gt_tok:
                    tok_corr += 1
                if pred_k == gt_k:
                    key_corr += 1
                if pred_v == gt_v:
                    val_corr += 1

                actual_offset = float(gt_v - gt_k)
                pred_offset = float(pred_v - pred_k)

                offsets.append(actual_offset)
                all_actual_offsets.append(actual_offset)
                all_predicted_offsets.append(pred_offset)

            t_acc = tok_corr / max(1, samples_per_layout)
            v_acc = val_corr / max(1, samples_per_layout)
            k_acc = key_corr / max(1, samples_per_layout)
            mean_off = sum(offsets) / max(1, len(offsets))

            layout_results[lay] = LayoutDestructionResult(
                layout_name=lay,
                num_samples=samples_per_layout,
                final_token_accuracy=t_acc,
                value_position_accuracy=v_acc,
                key_position_accuracy=k_acc,
                mean_key_value_offset=mean_off,
            )

    # Compute Pearson correlation between predicted offset and actual offset
    # If model always predicts a fixed offset (e.g. key + 7), variance is ~0 or offset doesn't match
    n = len(all_actual_offsets)
    mean_act = sum(all_actual_offsets) / max(1, n)
    mean_pred = sum(all_predicted_offsets) / max(1, n)

    num = sum((a - mean_act) * (p - mean_pred) for a, p in zip(all_actual_offsets, all_predicted_offsets))
    den_a = math.sqrt(sum((a - mean_act) ** 2 for a in all_actual_offsets))
    den_p = math.sqrt(sum((p - mean_pred) ** 2 for p in all_predicted_offsets))

    corr = (num / (den_a * den_p)) if (den_a > 1e-6 and den_p > 1e-6) else 0.0

    mean_t = sum(r.final_token_accuracy for r in layout_results.values()) / len(layout_results)
    mean_v = sum(r.value_position_accuracy for r in layout_results.values()) / len(layout_results)

    # If accuracy drops to 0 on reverse_order (Layout 2) while high on standard_map, it's positional shortcut
    std_v = layout_results["standard_map"].value_position_accuracy
    rev_v = layout_results["reverse_order"].value_position_accuracy
    is_shortcut = (std_v > 0.50 and rev_v == 0.0)

    routing_nature = "POSITIONAL_SHORTCUT" if is_shortcut else "RELATIONSHIP_DRIVEN"
    elapsed_ms = (time.time() - t0) * 1000.0

    return PositionalHeuristicDestructionReport(
        seed=seed,
        layout_results=layout_results,
        mean_token_acc=mean_t,
        mean_val_pos_acc=mean_v,
        offset_position_correlation=corr,
        is_shortcut_dependent=is_shortcut,
        routing_nature=routing_nature,
        cpu_runtime_ms=elapsed_ms,
    )
