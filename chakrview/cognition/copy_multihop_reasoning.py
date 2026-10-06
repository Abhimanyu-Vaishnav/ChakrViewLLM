"""Step 190: Multi-Hop Transitive Reasoning with Copy Mechanism.

Evaluates whether the copy mechanism enables transitive relational reasoning:
1-hop: A -> B, query A -> B
2-hop: A -> B, B -> C, query A -> C
3-hop: A -> B, B -> C, C -> D, query A -> D
4-hop: A -> B, B -> C, C -> D, D -> E, query A -> E

Uses disjoint entity symbols to prevent associative shortcut memorization.
Measures:
- Hop-by-hop accuracy (1-hop, 2-hop, 3-hop, 4-hop)
- Intermediate representation & attention routing
- Copy source position (does it copy the final target C, or intermediate B, or last token?)
- Target probability and target rank
- Identifies the exact failure boundary if 2-hop remains 0.0000.
"""

from __future__ import annotations

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
class HopResult:
    hop_num: int
    accuracy: float
    mean_target_prob: float
    mean_target_logit: float
    median_target_rank: float
    copied_target_pos_rate: float
    copied_last_token_rate: float
    copied_intermediate_token_rate: float


@dataclasses.dataclass
class MultiHopCopyEvaluation:
    model_name: str
    seed: int
    results_by_hop: Dict[int, HopResult]
    failure_boundary_hop: int
    cpu_runtime_ms: float = 0.0


def evaluate_multihop_reasoning(
    model: torch.nn.Module,
    seed: int = 42,
    samples_per_hop: int = 15,
) -> MultiHopCopyEvaluation:
    """Evaluates 1-hop through 4-hop transitive reasoning with disjoint symbols."""
    torch.manual_seed(seed)
    rng = random.Random(seed)
    tokenizer = get_default_tokenizer()
    is_copy = isinstance(model, ChakrMicroWithCopy)
    model.eval()

    symbols_pool = ["P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]

    results_by_hop = {}
    failure_boundary = 1
    t0 = time.time()

    with torch.no_grad():
        for hops in [1, 2, 3, 4]:
            correct = 0
            target_probs = []
            target_logits = []
            ranks = []
            copied_target_cnt = 0
            copied_last_cnt = 0
            copied_inter_cnt = 0

            for _ in range(samples_per_hop):
                chain = rng.sample(symbols_pool, hops + 1)
                pairs = [(chain[i], chain[i + 1]) for i in range(hops)]
                parts = [f"|{u}| -> |{v}|" for u, v in pairs]
                prompt = "map " + " and ".join(parts) + f" query |{chain[0]}| -> |"
                expected_char = chain[-1]
                enc_exp = tokenizer.encode(expected_char, add_bos=False, add_eos=False)
                expected_token = enc_exp[0]

                intermediate_tokens = [tokenizer.encode(c, add_bos=False, add_eos=False)[0] for c in chain[1:-1]]

                tokens = tokenizer.encode(prompt, add_bos=True, add_eos=False)
                inp = torch.tensor([tokens], dtype=torch.long)

                if is_copy:
                    logits, copy_out = model(inp)
                    sel_pos = int(copy_out.selected_source_pos[0, -1].item())
                    sel_tok = int(inp[0, sel_pos].item())
                    if sel_tok == expected_token:
                        copied_target_cnt += 1
                    if sel_tok in intermediate_tokens:
                        copied_inter_cnt += 1
                    if sel_pos == inp.shape[1] - 1 or sel_pos == inp.shape[1] - 2:
                        copied_last_cnt += 1
                else:
                    logits = model(inp)

                last_logits = logits[0, -1, :]
                probs = F.softmax(last_logits, dim=-1)

                t_prob = float(probs[expected_token].item())
                t_logit = float(last_logits[expected_token].item())
                sorted_idx = torch.argsort(last_logits, descending=True)
                rank = int((sorted_idx == expected_token).nonzero(as_tuple=True)[0].item()) + 1

                target_probs.append(t_prob)
                target_logits.append(t_logit)
                ranks.append(rank)

                pred = int(torch.argmax(last_logits).item())
                if pred == expected_token:
                    correct += 1

            n = samples_per_hop
            acc = float(correct / n) if n else 0.0
            if acc > 0.0 and failure_boundary <= hops:
                failure_boundary = hops + 1

            results_by_hop[hops] = HopResult(
                hop_num=hops,
                accuracy=acc,
                mean_target_prob=float(sum(target_probs) / n) if n else 0.0,
                mean_target_logit=float(sum(target_logits) / n) if n else 0.0,
                median_target_rank=float(sorted(ranks)[len(ranks) // 2]) if ranks else 0.0,
                copied_target_pos_rate=float(copied_target_cnt / n) if n else 0.0,
                copied_last_token_rate=float(copied_last_cnt / n) if n else 0.0,
                copied_intermediate_token_rate=float(copied_inter_cnt / n) if n else 0.0,
            )

    elapsed_ms = (time.time() - t0) * 1000.0

    return MultiHopCopyEvaluation(
        model_name="copy_model" if is_copy else "tied_base",
        seed=seed,
        results_by_hop=results_by_hop,
        failure_boundary_hop=failure_boundary,
        cpu_runtime_ms=elapsed_ms,
    )
