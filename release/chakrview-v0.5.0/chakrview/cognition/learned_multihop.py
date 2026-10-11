"""Step 231: Rebinding + Multi-Hop Learning.

Scientific Rule:
If the candidate cannot pass I3 (disjoint associative transfer), classify this step as
EXPLORATORY / BLOCKED rather than manufacturing ungrounded I4/I5 claims.

Tests:
1. Contextual Rebinding:
   B -> Y, query B -> Y
   then: B -> X, query B -> X
   then reverse: B -> Y
   Measures whether the candidate follows the latest contextual overwrite.

2. Neural Multi-Hop Learning:
   2-hop: A -> B, B -> C, query A -> target C
   3-hop: A -> B -> C -> D, query A -> target D
   4-hop: A -> B -> C -> D -> E, query A -> target E

Required neural process:
   A -> retrieve B rep -> B rep becomes next query -> retrieve C rep.
No symbolic traversal, no Python dictionary lookup, no precomputed intermediate answers.
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
from chakrview.cognition.representation_multihop import (
    evaluate_representation_multihop,
    RepresentationMultiHopReport,
)


@dataclasses.dataclass
class RebindingEvaluationResult:
    original_mapping_acc: float
    overwritten_mapping_acc: float
    reverse_rebinding_acc: float
    contextual_overwrite_ratio: float
    is_rebinding_followed: bool


@dataclasses.dataclass
class LearnedMultiHopReport:
    seed: int
    is_i3_qualified: bool
    status: str
    rebinding_result: Optional[RebindingEvaluationResult]
    multihop_report: Optional[RepresentationMultiHopReport]
    conclusion: str
    cpu_runtime_ms: float = 0.0


def evaluate_learned_rebinding(
    model: ChakrMicro,
    seed: int = 42,
    num_episodes: int = 10,
) -> RebindingEvaluationResult:
    """Evaluates whether model follows in-context rebinding updates."""
    torch.manual_seed(seed)
    rng = random.Random(seed)
    tok = get_default_tokenizer()
    model.eval()

    corr_orig = 0
    corr_over = 0
    corr_rev = 0

    with torch.no_grad():
        for _ in range(num_episodes):
            k = "B"
            v1, v2 = "Y", "X"
            tok_v1 = tok.encode(v1, add_bos=False, add_eos=False)[0]
            tok_v2 = tok.encode(v2, add_bos=False, add_eos=False)[0]

            # 1. Original
            p1 = f"map |{k}| -> |{v1}| query |{k}| -> |"
            t1 = tok.encode(p1, add_bos=True, add_eos=False)
            l1 = model(torch.tensor([t1], dtype=torch.long))
            if torch.argmax(l1[0, -1, :]).item() == tok_v1:
                corr_orig += 1

            # 2. Overwrite: |B| -> |Y| and later |B| -> |X|
            p2 = f"map |{k}| -> |{v1}| update |{k}| -> |{v2}| query |{k}| -> |"
            t2 = tok.encode(p2, add_bos=True, add_eos=False)
            l2 = model(torch.tensor([t2], dtype=torch.long))
            if torch.argmax(l2[0, -1, :]).item() == tok_v2:
                corr_over += 1

            # 3. Reverse
            p3 = f"map |{k}| -> |{v2}| update |{k}| -> |{v1}| query |{k}| -> |"
            t3 = tok.encode(p3, add_bos=True, add_eos=False)
            l3 = model(torch.tensor([t3], dtype=torch.long))
            if torch.argmax(l3[0, -1, :]).item() == tok_v1:
                corr_rev += 1

    n = max(1, num_episodes)
    acc_orig = corr_orig / n
    acc_over = corr_over / n
    acc_rev = corr_rev / n
    followed = (acc_over >= 0.50 and acc_rev >= 0.50)

    return RebindingEvaluationResult(
        original_mapping_acc=acc_orig,
        overwritten_mapping_acc=acc_over,
        reverse_rebinding_acc=acc_rev,
        contextual_overwrite_ratio=acc_over,
        is_rebinding_followed=followed,
    )


def evaluate_learned_multihop(
    candidate: ChakrMicro,
    is_i3_qualified: bool = False,
    seed: int = 42,
) -> LearnedMultiHopReport:
    """Evaluates rebinding and multi-hop. If I3 is not qualified, flags as exploratory/blocked."""
    t0 = time.time()

    if not is_i3_qualified:
        # Exploratory diagnostic run
        rebinding_res = evaluate_learned_rebinding(candidate, seed=seed, num_episodes=6)
        multihop_res = evaluate_representation_multihop(candidate, seed=seed, num_eval_chains=3)

        return LearnedMultiHopReport(
            seed=seed,
            is_i3_qualified=False,
            status="EXPLORATORY / BLOCKED — I3 PRECONDITION NOT SATISFIED",
            rebinding_result=rebinding_res,
            multihop_report=multihop_res,
            conclusion="Disjoint associative retrieval remains unproven. Multi-hop and variable binding cannot be promoted without satisfying I3 criteria.",
            cpu_runtime_ms=(time.time() - t0) * 1000.0,
        )

    # If I3 were qualified:
    rebinding_res = evaluate_learned_rebinding(candidate, seed=seed, num_episodes=15)
    multihop_res = evaluate_representation_multihop(candidate, seed=seed, num_eval_chains=8)

    return LearnedMultiHopReport(
        seed=seed,
        is_i3_qualified=True,
        status="PROMOTION_EVALUATED",
        rebinding_result=rebinding_res,
        multihop_report=multihop_res,
        conclusion="Full I4/I5 pipeline evaluated under verified disjoint transfer.",
        cpu_runtime_ms=(time.time() - t0) * 1000.0,
    )
