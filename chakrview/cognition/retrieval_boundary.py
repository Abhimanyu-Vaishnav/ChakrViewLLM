"""Step 218: Three-Stage Failure Boundary.

Explicitly separates:
STAGE A: Association formation (pair state encoding & contrastive separation >= 0.50)
STAGE B: Value representation retrieval (positive cosine margin & rank above chance)
STAGE C: Vocabulary / token output (token prediction accuracy >= 0.50)

Evaluates the exact same prompt instances across all three stages simultaneously:
- known-known
- known-unseen
- unseen-known
- unseen-unseen

Provides explicit diagnosis of where the capability breaks down.
Does NOT infer Stage B success from Stage C.
Does NOT infer Stage A success from probes alone.
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
from chakrview.tokenizer.serialization import load_tokenizer_artifacts
from chakrview.cognition.value_representation_retrieval import (
    extract_contextual_representations,
    compute_representation_retrieval_metrics,
    get_default_tokenizer,
)


@dataclasses.dataclass
class StageBoundaryResult:
    condition: str
    stage_a_association_score: float   # Contrastive separation between bound pair and negatives
    stage_a_passed: bool              # threshold >= 0.50
    stage_b_value_margin: float       # Cosine margin between retrieved rep and distractors
    stage_b_value_rank: float         # Rank among candidate values (1 to N)
    stage_b_passed: bool              # margin > 0.0 and rank < 2.0 (for N=3)
    stage_c_final_token_acc: float    # Exact token match accuracy
    stage_c_passed: bool              # accuracy >= 0.50
    first_failure_stage: str          # "STAGE_A", "STAGE_B", "STAGE_C", or "NONE"


@dataclasses.dataclass
class Step218BoundaryReport:
    seed: int
    stages_table: Dict[str, StageBoundaryResult]
    overall_failure_boundary: str
    cpu_runtime_ms: float = 0.0


def evaluate_three_stage_retrieval_boundary(
    model: ChakrMicro,
    seed: int = 42,
    samples_per_condition: int = 15,
) -> Step218BoundaryReport:
    """Evaluates Stage A, Stage B, and Stage C on identical prompt instances."""
    torch.manual_seed(seed)
    rng = random.Random(seed)
    tok = get_default_tokenizer()
    model.eval()

    known_keys = ["A", "B", "C", "D", "E"]
    known_vals = ["1", "2", "3", "4", "5"]
    unseen_keys = ["P", "Q", "R", "S", "T"]
    unseen_vals = ["6", "7", "8", "9", "0"]

    conditions_map = {
        "known_known": (known_keys, known_vals),
        "known_unseen": (known_keys, unseen_vals),
        "unseen_known": (unseen_keys, known_vals),
        "unseen_unseen": (unseen_keys, unseen_vals),
    }

    t0 = time.time()
    table = {}
    failure_stages = []

    cos = nn.CosineSimilarity(dim=0)

    for cond_name, (k_pool, v_pool) in conditions_map.items():
        stage_a_scores = []
        stage_b_margins = []
        stage_b_ranks = []
        stage_c_correct = 0
        total_eval = 0

        for _ in range(samples_per_condition):
            sampled_k = rng.sample(k_pool, 3)
            sampled_v = rng.sample(v_pool, 3)
            pairs = list(zip(sampled_k, sampled_v))
            rng.shuffle(pairs)

            query_pair = pairs[0]
            query_k, exp_v = query_pair
            exp_tok = tok.encode(exp_v, add_bos=False, add_eos=False)[0]

            parts = [f"|{k}| -> |{v}|" for k, v in pairs]
            prompt = "map " + " and ".join(parts) + f" query |{query_k}| -> |"
            token_ids = tok.encode(prompt, add_bos=True, add_eos=False)
            inp = torch.tensor([token_ids], dtype=torch.long)

            # 1. Forward pass
            with torch.no_grad():
                x = model.embedding(inp)
                for i, layer in enumerate(model.layers):
                    x = layer(x, attention_mask=None, kv_cache=None, layer_idx=i)
                hidden = model.final_norm(x)[0]
                logits = model.lm_head(hidden.unsqueeze(0))[0]

            extracted = extract_contextual_representations(
                model=model,
                tokenizer=tok,
                prompt=prompt,
                pairs=pairs,
                query_key=query_k,
            )

            v_reps = extracted["v_reps"]
            k_reps = extracted["k_reps"]
            if exp_v not in v_reps or query_k not in k_reps or len(v_reps) < 2:
                continue

            total_eval += 1

            # STAGE A: Association formation
            # Target pair representation vs negative mismatched pairs
            # Positive: k_query * v_target
            # Negative: k_query * v_other
            pos_assoc = k_reps[query_k] * v_reps[exp_v]
            neg_assoc_sims = []
            for vk, vt in v_reps.items():
                if vk != exp_v:
                    neg_assoc = k_reps[query_k] * vt
                    neg_assoc_sims.append(float(cos(pos_assoc, neg_assoc).item()))
            
            # Association separation metric (1.0 - mean negative cosine similarity)
            # When pairs are well distinguished, cosine between distinct association vectors is lower
            mean_neg_sim = float(sum(neg_assoc_sims) / len(neg_assoc_sims)) if neg_assoc_sims else 0.0
            stage_a_score = max(0.0, 1.0 - abs(mean_neg_sim))
            stage_a_scores.append(stage_a_score)

            # STAGE B: Value representation retrieval
            retrieved_rep = extracted["terminal_query"]
            sim_corr, sim_incorr, margin, rank = compute_representation_retrieval_metrics(
                retrieved_rep=retrieved_rep,
                correct_value_rep=v_reps[exp_v],
                candidate_value_reps=v_reps,
                correct_val_key=exp_v,
            )
            stage_b_margins.append(margin)
            stage_b_ranks.append(float(rank))

            # STAGE C: Vocabulary / Token output
            last_logits = logits[-1]
            pred_tok = torch.argmax(last_logits).item()
            if pred_tok == exp_tok:
                stage_c_correct += 1

        n = total_eval if total_eval > 0 else 1
        avg_a = float(sum(stage_a_scores) / len(stage_a_scores)) if stage_a_scores else 0.0
        avg_b_margin = float(sum(stage_b_margins) / len(stage_b_margins)) if stage_b_margins else 0.0
        avg_b_rank = float(sum(stage_b_ranks) / len(stage_b_ranks)) if stage_b_ranks else 3.0
        c_acc = float(stage_c_correct / n)

        # Explicit thresholds:
        pass_a = avg_a >= 0.50
        pass_b = (avg_b_margin > 0.0) and (avg_b_rank < 2.0)
        pass_c = c_acc >= 0.50

        if not pass_a:
            first_fail = "STAGE_A"
        elif not pass_b:
            first_fail = "STAGE_B"
        elif not pass_c:
            first_fail = "STAGE_C"
        else:
            first_fail = "NONE"

        failure_stages.append(first_fail)

        table[cond_name] = StageBoundaryResult(
            condition=cond_name,
            stage_a_association_score=avg_a,
            stage_a_passed=pass_a,
            stage_b_value_margin=avg_b_margin,
            stage_b_value_rank=avg_b_rank,
            stage_b_passed=pass_b,
            stage_c_final_token_acc=c_acc,
            stage_c_passed=pass_c,
            first_failure_stage=first_fail,
        )

    # Determine boundary across unseen-unseen condition specifically
    unseen_res = table.get("unseen_unseen")
    overall_boundary = unseen_res.first_failure_stage if unseen_res else "UNKNOWN"

    cpu_ms = (time.time() - t0) * 1000.0

    return Step218BoundaryReport(
        seed=seed,
        stages_table=table,
        overall_failure_boundary=overall_boundary,
        cpu_runtime_ms=cpu_ms,
    )
