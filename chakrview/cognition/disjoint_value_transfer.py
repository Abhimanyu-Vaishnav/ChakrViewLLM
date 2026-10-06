"""Step 219: Disjoint Value Representation Transfer.

Train/evaluate using disjoint identity sets:
Training:
    keys A-E
    values 1-5
Evaluation:
    keys P-T
    values 6-0

Mappings randomized independently per episode.
No fixed mapping.

Conditions:
1. known-key / known-value
2. known-key / unseen-value
3. unseen-key / known-value
4. unseen-key / unseen-value

Primary Question:
Can the association mechanism retrieve a VALUE REPRESENTATION for an unseen value
even when final vocabulary output fails?

Multi-seed replication: Seeds 42, 101, 2026.
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
class DisjointSplitMetrics:
    condition: str
    association_score: float
    value_rep_cosine: float
    value_rep_margin: float
    value_rep_rank: float
    final_token_prob: float
    final_token_rank: float
    final_token_acc: float
    sample_count: int


@dataclasses.dataclass
class DisjointTransferReport:
    seed: int
    splits: Dict[str, DisjointSplitMetrics]
    is_value_representation_transferred: bool
    is_token_retrieval_transferred: bool
    next_investigation_target: str # "VOCABULARY_PROJECTION" or "ASSOCIATIVE_ROUTING"
    cpu_runtime_ms: float = 0.0


def evaluate_disjoint_value_transfer(
    model: ChakrMicro,
    seed: int = 42,
    samples_per_split: int = 15,
) -> DisjointTransferReport:
    """Evaluates representation retrieval vs token retrieval across disjoint entity sets."""
    torch.manual_seed(seed)
    rng = random.Random(seed)
    tok = get_default_tokenizer()
    model.eval()

    # Disjoint entity pools
    train_keys = ["A", "B", "C", "D", "E"]
    train_vals = ["1", "2", "3", "4", "5"]
    test_keys = ["P", "Q", "R", "S", "T"]
    test_vals = ["6", "7", "8", "9", "0"]

    conditions_map = {
        "known_known": (train_keys, train_vals),
        "known_unseen": (train_keys, test_vals),
        "unseen_known": (test_keys, train_vals),
        "unseen_unseen": (test_keys, test_vals),
    }

    t0 = time.time()
    splits_res = {}
    cos = nn.CosineSimilarity(dim=0)

    for cond_name, (k_pool, v_pool) in conditions_map.items():
        assoc_scores = []
        val_cosines = []
        val_margins = []
        val_ranks = []
        token_probs = []
        token_ranks = []
        token_correct = 0
        total_eval = 0

        for _ in range(samples_per_split):
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

            # Association formation score
            pos_pair = k_reps[query_k] * v_reps[exp_v]
            neg_sims = [float(cos(pos_pair, k_reps[query_k] * vt).item()) for vk, vt in v_reps.items() if vk != exp_v]
            mean_neg = float(sum(neg_sims) / len(neg_sims)) if neg_sims else 0.0
            assoc_score = max(0.0, 1.0 - abs(mean_neg))
            assoc_scores.append(assoc_score)

            # Value representation retrieval
            retrieved_rep = extracted["terminal_query"]
            sim_corr, sim_incorr, margin, rank = compute_representation_retrieval_metrics(
                retrieved_rep=retrieved_rep,
                correct_value_rep=v_reps[exp_v],
                candidate_value_reps=v_reps,
                correct_val_key=exp_v,
            )
            val_cosines.append(sim_corr)
            val_margins.append(margin)
            val_ranks.append(float(rank))

            # Final token prediction
            last_logits = logits[-1]
            p = F.softmax(last_logits, dim=-1)
            t_prob = float(p[exp_tok].item())
            sorted_idx = torch.argsort(last_logits, descending=True)
            t_rank = int((sorted_idx == exp_tok).nonzero(as_tuple=True)[0].item()) + 1

            token_probs.append(t_prob)
            token_ranks.append(float(t_rank))

            pred_tok = torch.argmax(last_logits).item()
            if pred_tok == exp_tok:
                token_correct += 1

        n = total_eval if total_eval > 0 else 1
        avg_assoc = float(sum(assoc_scores) / len(assoc_scores)) if assoc_scores else 0.0
        avg_val_cos = float(sum(val_cosines) / len(val_cosines)) if val_cosines else 0.0
        avg_val_margin = float(sum(val_margins) / len(val_margins)) if val_margins else 0.0
        avg_val_rank = float(sum(val_ranks) / len(val_ranks)) if val_ranks else 3.0
        avg_t_prob = float(sum(token_probs) / len(token_probs)) if token_probs else 0.0
        avg_t_rank = float(sum(token_ranks) / len(token_ranks)) if token_ranks else 1000.0
        t_acc = float(token_correct / n)

        splits_res[cond_name] = DisjointSplitMetrics(
            condition=cond_name,
            association_score=avg_assoc,
            value_rep_cosine=avg_val_cos,
            value_rep_margin=avg_val_margin,
            value_rep_rank=avg_val_rank,
            final_token_prob=avg_t_prob,
            final_token_rank=avg_t_rank,
            final_token_acc=t_acc,
            sample_count=n,
        )

    unseen_res = splits_res.get("unseen_unseen")
    # Rep retrieval success requires rank < 2.0 (out of 3) and margin > 0.0
    val_rep_success = bool(unseen_res and unseen_res.value_rep_margin > 0.0 and unseen_res.value_rep_rank < 2.0)
    token_success = bool(unseen_res and unseen_res.final_token_acc >= 0.50)

    # Next investigation target decision logic:
    if val_rep_success and not token_success:
        next_target = "VOCABULARY_PROJECTION"
    else:
        next_target = "ASSOCIATIVE_ROUTING"

    cpu_ms = (time.time() - t0) * 1000.0

    return DisjointTransferReport(
        seed=seed,
        splits=splits_res,
        is_value_representation_transferred=val_rep_success,
        is_token_retrieval_transferred=token_success,
        next_investigation_target=next_target,
        cpu_runtime_ms=cpu_ms,
    )
