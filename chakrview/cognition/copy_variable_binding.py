"""Step 189: Variable Binding Evaluation with Copy Mechanism.

Evaluates variable binding before and after the neural copy mechanism.
Tasks:
1. Familiar entities:
   - "order A > B query first -> A"
   - "order A > B query second -> B"
2. Disjoint entities:
   - "order P > Q query first -> P"
   - "order P > Q query second -> Q"
3. Randomized entities, reversed relations, distractors.

Measures:
- Accuracy (first query, second query, disjoint)
- Target rank and target probability
- Selected source position and copy attention distribution
- Comparison against Step 181 pre-copy baseline.
"""

from __future__ import annotations

import copy
import dataclasses
from pathlib import Path
import random
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from chakrview.brain.model import ChakrMicro
from chakrview.cognition.copy_attention import ChakrMicroWithCopy
from chakrview.cognition.disjoint_token_benchmark import get_default_tokenizer
from chakrview.tokenizer.tokenizer import BPETokenizer


@dataclasses.dataclass
class VariableBindingEvaluation:
    model_name: str
    familiar_first_acc: float
    familiar_second_acc: float
    familiar_overall_acc: float
    disjoint_first_acc: float
    disjoint_second_acc: float
    disjoint_overall_acc: float
    mean_target_prob: float
    median_target_rank: float
    source_position_acc: float
    cpu_runtime_ms: float = 0.0


def evaluate_variable_binding_suite(
    model: torch.nn.Module,
    seed: int = 42,
    num_samples_per_condition: int = 15,
) -> VariableBindingEvaluation:
    """Evaluate variable binding on both familiar (A, B) and disjoint (P, Q, X, Y) pairs."""
    torch.manual_seed(seed)
    rng = random.Random(seed)
    tokenizer = get_default_tokenizer()
    is_copy = isinstance(model, ChakrMicroWithCopy)
    model.eval()

    familiar_pairs = [("A", "B"), ("C", "D"), ("E", "F")]
    disjoint_pool_1 = ["P", "Q", "R", "S"]
    disjoint_pool_2 = ["X", "Y", "Z", "W"]

    conditions = {
        "fam_first": [],
        "fam_second": [],
        "disj_first": [],
        "disj_second": [],
    }

    for _ in range(num_samples_per_condition):
        f1, f2 = rng.choice(familiar_pairs)
        if rng.random() < 0.5:
            conditions["fam_first"].append((f"order {f1} > {f2} query first -> ", f1))
        else:
            conditions["fam_first"].append((f"order {f2} > {f1} query first -> ", f2))

        f1, f2 = rng.choice(familiar_pairs)
        if rng.random() < 0.5:
            conditions["fam_second"].append((f"order {f1} > {f2} query second -> ", f2))
        else:
            conditions["fam_second"].append((f"order {f2} > {f1} query second -> ", f1))

        d1 = rng.choice(disjoint_pool_1)
        d2 = rng.choice(disjoint_pool_2)
        conditions["disj_first"].append((f"order {d1} > {d2} query first -> ", d1))

        d1 = rng.choice(disjoint_pool_1)
        d2 = rng.choice(disjoint_pool_2)
        conditions["disj_second"].append((f"order {d1} > {d2} query second -> ", d2))

    scores = {}
    all_target_probs = []
    all_ranks = []
    source_correct = 0
    total_evals = 0

    t0 = time.time()
    with torch.no_grad():
        for cond, samples in conditions.items():
            correct = 0
            for prompt, exp_char in samples:
                enc = tokenizer.encode(exp_char, add_bos=False, add_eos=False)
                exp_token = enc[0]
                tokens = tokenizer.encode(prompt, add_bos=True, add_eos=False)
                inp = torch.tensor([tokens], dtype=torch.long)

                if is_copy:
                    logits, copy_out = model(inp)
                    sel_pos = int(copy_out.selected_source_pos[0, -1].item())
                    if inp[0, sel_pos].item() == exp_token:
                        source_correct += 1
                else:
                    logits = model(inp)

                total_evals += 1
                last_logits = logits[0, -1, :]
                probs = F.softmax(last_logits, dim=-1)

                t_prob = float(probs[exp_token].item())
                sorted_idx = torch.argsort(last_logits, descending=True)
                rank = int((sorted_idx == exp_token).nonzero(as_tuple=True)[0].item()) + 1

                all_target_probs.append(t_prob)
                all_ranks.append(rank)

                pred = int(torch.argmax(last_logits).item())
                if pred == exp_token:
                    correct += 1

            scores[cond] = correct / len(samples) if samples else 0.0

    elapsed_ms = (time.time() - t0) * 1000.0

    fam_avg = (scores["fam_first"] + scores["fam_second"]) / 2.0
    disj_avg = (scores["disj_first"] + scores["disj_second"]) / 2.0

    return VariableBindingEvaluation(
        model_name="copy_model" if is_copy else "tied_base",
        familiar_first_acc=scores["fam_first"],
        familiar_second_acc=scores["fam_second"],
        familiar_overall_acc=fam_avg,
        disjoint_first_acc=scores["disj_first"],
        disjoint_second_acc=scores["disj_second"],
        disjoint_overall_acc=disj_avg,
        mean_target_prob=float(sum(all_target_probs) / len(all_target_probs)) if all_target_probs else 0.0,
        median_target_rank=float(sorted(all_ranks)[len(all_ranks) // 2]) if all_ranks else 0.0,
        source_position_acc=float(source_correct / total_evals) if total_evals else 0.0,
        cpu_runtime_ms=elapsed_ms,
    )
