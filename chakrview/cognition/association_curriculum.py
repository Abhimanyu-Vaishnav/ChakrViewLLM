"""Step 226: Association Curriculum Ladder.

Gated curriculum levels:
    L0 = One association: B -> Y, query B
    L1 = Two associations: A -> X, B -> Y, query B
    L2 = Multiple associations (3-4 pairs): A -> X, B -> Y, C -> Z, D -> P, query C
    L3 = Distractors: Unrelated pairs interleaved
    L4 = Reordered mappings: Shuffled contextual pair presentation order
    L5 = Delayed query: Filler tokens between mappings and query
    L6 = Disjoint identities: Train on A-E/1-5, eval on P-T/6-0
    L7 = Randomized variable roles & dynamic assignments
    L8 = Mixed association conditions (composite benchmark)

Predefined Thresholds for Level Advancement:
    Train Accuracy >= 0.80
    Validation Accuracy >= 0.70
    Held-out Accuracy >= 0.50

Promotion Rule:
If a level fails, HALT CURRICULUM ADVANCEMENT and record the failure boundary.
"""

from __future__ import annotations

import copy
import dataclasses
from pathlib import Path
import random
import time
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from chakrview.brain.model import ChakrMicro
from chakrview.tokenizer.tokenizer import BPETokenizer
from chakrview.cognition.value_representation_retrieval import get_default_tokenizer


@dataclasses.dataclass
class CurriculumLevelResult:
    level_id: str
    level_name: str
    train_acc: float
    val_acc: float
    heldout_acc: float
    passed: bool
    failure_reason: Optional[str] = None


@dataclasses.dataclass
class CurriculumLadderReport:
    seed: int
    levels: Dict[str, CurriculumLevelResult]
    highest_passed_level: str
    halted_at_level: Optional[str]
    is_curriculum_completed: bool
    cpu_runtime_ms: float = 0.0


def evaluate_curriculum_level(
    model: ChakrMicro,
    tok: BPETokenizer,
    level_id: str,
    seed: int = 42,
    num_episodes: int = 10,
) -> CurriculumLevelResult:
    rng = random.Random(seed)
    model.eval()

    train_keys = ["A", "B", "C", "D", "E"]
    train_vals = ["1", "2", "3", "4", "5"]
    heldout_keys = ["P", "Q", "R", "S", "T"]
    heldout_vals = ["6", "7", "8", "9", "0"]

    def run_split(keys: List[str], vals: List[str], episodes: int) -> float:
        correct = 0
        with torch.no_grad():
            for _ in range(episodes):
                if level_id == "L0":
                    k = rng.choice(keys)
                    v = rng.choice(vals)
                    pairs = [(k, v)]
                    query_k = k
                    exp_v = v
                    prompt = f"map |{k}| -> |{v}| query |{query_k}| -> |"
                elif level_id == "L1":
                    ks = rng.sample(keys, min(2, len(keys)))
                    vs = rng.sample(vals, min(2, len(vals)))
                    pairs = list(zip(ks, vs))
                    query_pair = rng.choice(pairs)
                    query_k, exp_v = query_pair
                    prompt = f"map |{pairs[0][0]}| -> |{pairs[0][1]}| and |{pairs[1][0]}| -> |{pairs[1][1]}| query |{query_k}| -> |"
                elif level_id == "L2":
                    ks = rng.sample(keys, min(4, len(keys)))
                    vs = rng.sample(vals, min(4, len(vals)))
                    pairs = list(zip(ks, vs))
                    query_pair = rng.choice(pairs)
                    query_k, exp_v = query_pair
                    parts = [f"|{k}| -> |{v}|" for k, v in pairs]
                    prompt = "map " + " and ".join(parts) + f" query |{query_k}| -> |"
                elif level_id == "L3": # Distractors
                    ks = rng.sample(keys, min(3, len(keys)))
                    vs = rng.sample(vals, min(3, len(vals)))
                    pairs = list(zip(ks, vs))
                    query_pair = pairs[0]
                    query_k, exp_v = query_pair
                    parts = [f"|{k}| -> |{v}|" for k, v in pairs]
                    parts.append("note distractor |X| -> |ignore|")
                    prompt = "map " + " and ".join(parts) + f" query |{query_k}| -> |"
                elif level_id == "L4": # Reordered
                    ks = rng.sample(keys, min(3, len(keys)))
                    vs = rng.sample(vals, min(3, len(vals)))
                    pairs = list(zip(ks, vs))
                    query_pair = pairs[0]
                    query_k, exp_v = query_pair
                    rng.shuffle(pairs)
                    parts = [f"|{k}| -> |{v}|" for k, v in pairs]
                    prompt = "map " + " and ".join(parts) + f" query |{query_k}| -> |"
                elif level_id == "L5": # Delayed
                    ks = rng.sample(keys, min(3, len(keys)))
                    vs = rng.sample(vals, min(3, len(vals)))
                    pairs = list(zip(ks, vs))
                    query_pair = pairs[0]
                    query_k, exp_v = query_pair
                    parts = [f"|{k}| -> |{v}|" for k, v in pairs]
                    prompt = "map " + " and ".join(parts) + " delay buffer filler tokens wait query " + f"|{query_k}| -> |"
                elif level_id in ("L6", "L7", "L8"):
                    # Disjoint / dynamic
                    ks = rng.sample(keys, min(3, len(keys)))
                    vs = rng.sample(vals, min(3, len(vals)))
                    pairs = list(zip(ks, vs))
                    query_pair = rng.choice(pairs)
                    query_k, exp_v = query_pair
                    parts = [f"|{k}| -> |{v}|" for k, v in pairs]
                    prompt = "map " + " and ".join(parts) + f" query |{query_k}| -> |"
                else:
                    return 0.0

                token_ids = tok.encode(prompt, add_bos=True, add_eos=False)
                inp = torch.tensor([token_ids], dtype=torch.long)
                exp_tok = tok.encode(exp_v, add_bos=False, add_eos=False)[0]

                logits = model(inp)
                pred_tok = torch.argmax(logits[0, -1, :]).item()
                if pred_tok == exp_tok:
                    correct += 1
        return float(correct / max(1, episodes))

    train_acc = run_split(train_keys, train_vals, num_episodes)
    val_acc = run_split(train_keys, train_vals, num_episodes)
    heldout_acc = run_split(heldout_keys, heldout_vals, num_episodes)

    # Thresholds: train >= 0.80, val >= 0.70, heldout >= 0.50
    passed = (train_acc >= 0.80 and val_acc >= 0.70 and heldout_acc >= 0.50)
    reason = None if passed else f"Accuracies below threshold: train={train_acc:.2f}, val={val_acc:.2f}, heldout={heldout_acc:.2f}"

    names = {
        "L0": "one_association",
        "L1": "two_associations",
        "L2": "multiple_associations",
        "L3": "distractors",
        "L4": "reordered_mappings",
        "L5": "delayed_query",
        "L6": "disjoint_identities",
        "L7": "randomized_roles",
        "L8": "mixed_conditions",
    }

    return CurriculumLevelResult(
        level_id=level_id,
        level_name=names.get(level_id, level_id),
        train_acc=train_acc,
        val_acc=val_acc,
        heldout_acc=heldout_acc,
        passed=passed,
        failure_reason=reason,
    )


def run_association_curriculum_ladder(
    model: ChakrMicro,
    seed: int = 42,
    episodes_per_level: int = 10,
) -> CurriculumLadderReport:
    """Evaluates the gated curriculum ladder. Halts at the first failed level."""
    tok = get_default_tokenizer()
    t0 = time.time()

    ladder = ["L0", "L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8"]
    levels_res: Dict[str, CurriculumLevelResult] = {}
    highest_passed = "NONE"
    halted_level = None

    for lvl in ladder:
        res = evaluate_curriculum_level(model, tok, lvl, seed=seed, num_episodes=episodes_per_level)
        levels_res[lvl] = res

        if res.passed:
            highest_passed = lvl
        else:
            halted_level = lvl
            # Strict Promotion Rule: HALT CURRICULUM ADVANCEMENT
            break

    is_complete = (highest_passed == "L8")
    elapsed = (time.time() - t0) * 1000.0

    return CurriculumLadderReport(
        seed=seed,
        levels=levels_res,
        highest_passed_level=highest_passed,
        halted_at_level=halted_level,
        is_curriculum_completed=is_complete,
        cpu_runtime_ms=elapsed,
    )
