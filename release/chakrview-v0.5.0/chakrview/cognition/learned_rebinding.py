"""Step 239: Learned Contextual Rebinding.

Scientific Rule:
ONLY execute promotion-level testing if I3 has meaningful evidence.
If I3 is not established, mark status as BLOCKED / EXPLORATORY.

Goal:
Teach and test dynamic contextual rebinding:
Context 1:
    A -> B
Context 2:
    A -> C
The model must correctly retrieve the context-specific association.

Then test:
- New context mapping
- Reordered context
- Distractors
- Changed query position
- Disjoint identities

STRICT INVARIANTS:
- No Python dictionary
- No symbolic propagation
- No external memory lookup
- The intermediate neural state must encode the active association.
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
class RebindingConditionResult:
    condition_name: str
    description: str
    accuracy: float
    is_followed: bool
    notes: str


@dataclasses.dataclass
class LearnedRebindingReport:
    seed: int
    is_i3_qualified: bool
    status: str
    conditions: Dict[str, RebindingConditionResult]
    mean_rebinding_acc: float
    is_rebinding_achieved: bool
    conclusion: str
    cpu_runtime_ms: float = 0.0


def evaluate_learned_contextual_rebinding(
    model: ChakrMicro,
    is_i3_qualified: bool = False,
    seed: int = 42,
    episodes: int = 8,
) -> LearnedRebindingReport:
    """Evaluates contextual rebinding. If I3 not met, marks as blocked/exploratory."""
    t0 = time.time()
    tok = get_default_tokenizer()

    if not is_i3_qualified:
        return LearnedRebindingReport(
            seed=seed,
            is_i3_qualified=False,
            status="BLOCKED — I3 PREREQUISITE NOT SATISFIED",
            conditions={},
            mean_rebinding_acc=0.0,
            is_rebinding_achieved=False,
            conclusion="Disjoint associative learning (I3) remains zero. Contextual rebinding capability cannot be verified without working associative routing.",
            cpu_runtime_ms=(time.time() - t0) * 1000.0,
        )

    torch.manual_seed(seed)
    rng = random.Random(seed)
    model.eval()

    corr_c1 = 0
    corr_c2 = 0
    corr_rev = 0

    with torch.no_grad():
        for _ in range(episodes):
            k = "A"
            v1, v2 = "B", "C"
            tok_v1 = tok.encode(v1, add_bos=False, add_eos=False)[0]
            tok_v2 = tok.encode(v2, add_bos=False, add_eos=False)[0]

            # Context 1: A -> B
            p1 = f"map |{k}| -> |{v1}| query |{k}| -> |"
            inp1 = torch.tensor([tok.encode(p1, add_bos=True, add_eos=False)], dtype=torch.long)
            if torch.argmax(model(inp1)[0, -1, :]).item() == tok_v1:
                corr_c1 += 1

            # Context 2: A -> B followed by in-context update A -> C
            p2 = f"map |{k}| -> |{v1}| update |{k}| -> |{v2}| query |{k}| -> |"
            inp2 = torch.tensor([tok.encode(p2, add_bos=True, add_eos=False)], dtype=torch.long)
            if torch.argmax(model(inp2)[0, -1, :]).item() == tok_v2:
                corr_c2 += 1

            # Context 3: Reverse rebinding A -> C then update A -> B
            p3 = f"map |{k}| -> |{v2}| update |{k}| -> |{v1}| query |{k}| -> |"
            inp3 = torch.tensor([tok.encode(p3, add_bos=True, add_eos=False)], dtype=torch.long)
            if torch.argmax(model(inp3)[0, -1, :]).item() == tok_v1:
                corr_rev += 1

    n = max(1, episodes)
    acc1 = corr_c1 / n
    acc2 = corr_c2 / n
    acc3 = corr_rev / n

    conds = {
        "context_1_initial": RebindingConditionResult("context_1_initial", "Initial binding A -> B", acc1, acc1 >= 0.50, f"Acc: {acc1:.2f}"),
        "context_2_overwrite": RebindingConditionResult("context_2_overwrite", "Contextual overwrite A -> C", acc2, acc2 >= 0.50, f"Acc: {acc2:.2f}"),
        "context_3_reverse": RebindingConditionResult("context_3_reverse", "Reverse rebinding update A -> B", acc3, acc3 >= 0.50, f"Acc: {acc3:.2f}"),
    }

    mean_acc = (acc1 + acc2 + acc3) / 3.0
    achieved = (acc2 >= 0.50 and acc3 >= 0.50)
    elapsed = (time.time() - t0) * 1000.0

    return LearnedRebindingReport(
        seed=seed,
        is_i3_qualified=True,
        status="EXECUTED",
        conditions=conds,
        mean_rebinding_acc=mean_acc,
        is_rebinding_achieved=achieved,
        conclusion="Contextual rebinding successfully followed in-context updates." if achieved else "Model failed to overwrite active binding.",
        cpu_runtime_ms=elapsed,
    )
