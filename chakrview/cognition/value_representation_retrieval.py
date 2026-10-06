"""Step 217: Value Representation Retrieval.

Scientific question:
When the model receives:
    map |B| -> |Y| query |B| -> |
does its internal representation retrieve the representation corresponding to Y?

Distinguishes:
1. Query representation (at query token position)
2. Association representation (contextual state encoding relationship)
3. Expected value representation (contextual hidden state at target value token position)
4. Retrieved value representation (internal neural retrieved vector)

Measures:
A. cosine(retrieved, correct value)
B. cosine(retrieved, incorrect values)
C. correct-value representation rank
D. contrastive margin (cosine(retrieved, correct) - max(cosine(retrieved, incorrect)))
E. representation retrieval accuracy (rank == 1)

Tested across 4 splits:
- known key / known value
- known key / unseen value
- unseen key / known value
- unseen key / unseen value

Strict constraints:
- Does NOT use token ID equality as the retrieval mechanism.
- Expected value representation is obtained from ChakrView's own contextual representation
  under the identical controlled protocol.
- Independent of vocabulary decoding / token generation.
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


def get_default_tokenizer() -> BPETokenizer:
    tok_dir = Path("data/experiments/vocab_4096")
    tok, _ = load_tokenizer_artifacts(tok_dir)
    return tok


@dataclasses.dataclass
class RepresentationRetrievalSplitResult:
    split_name: str
    cosine_correct: float
    cosine_incorrect: float
    contrastive_margin: float
    representation_rank: float
    representation_retrieval_acc: float
    sample_count: int


@dataclasses.dataclass
class Step217RetrievalReport:
    seed: int
    splits: Dict[str, RepresentationRetrievalSplitResult]
    is_representation_retrieval_successful: bool
    overall_mean_margin: float
    cpu_runtime_ms: float = 0.0


def extract_contextual_representations(
    model: ChakrMicro,
    tokenizer: BPETokenizer,
    prompt: str,
    pairs: List[Tuple[str, str]],
    query_key: str,
) -> Dict[str, Any]:
    """Extracts query, key, association, and value representations from ChakrMicro forward pass.
    
    Returns:
        h_query: contextual representation at the final query prompt token
        h_keys: list of representations for each key token
        h_vals: list of representations for each value token
        k_reps_dict: dict of key -> tensor
        v_reps_dict: dict of val -> tensor
        h_assoc: pooled association states
    """
    tokens = tokenizer.encode(prompt, add_bos=True, add_eos=False)
    inp = torch.tensor([tokens], dtype=torch.long)

    with torch.no_grad():
        x = model.embedding(inp)
        for i, layer in enumerate(model.layers):
            x = layer(x, attention_mask=None, kv_cache=None, layer_idx=i)
        hidden = model.final_norm(x)[0]  # [T, d_model]

    # Map positions of keys and values in context
    k_reps = {}
    v_reps = {}
    assoc_reps = {}

    for k, v in pairs:
        k_enc = tokenizer.encode(f"|{k}|", add_bos=False, add_eos=False)
        v_enc = tokenizer.encode(f"|{v}|", add_bos=False, add_eos=False)
        k_pos, v_pos = -1, -1

        for idx in range(len(tokens) - len(k_enc) + 1):
            if tokens[idx:idx+len(k_enc)] == k_enc:
                k_pos = idx + 1
                break
        for idx in range(len(tokens) - len(v_enc) + 1):
            if tokens[idx:idx+len(v_enc)] == v_enc:
                v_pos = idx + 1
                break

        if k_pos != -1:
            k_reps[k] = hidden[k_pos].detach()
        if v_pos != -1:
            v_reps[v] = hidden[v_pos].detach()
        if k_pos != -1 and v_pos != -1:
            # Pair interaction representation (k * v)
            assoc_reps[(k, v)] = (hidden[k_pos] * hidden[v_pos]).detach()

    # Query token position: the last occurrence of |query_k|
    q_enc = tokenizer.encode(f"|{query_key}|", add_bos=False, add_eos=False)
    q_pos = -1
    for idx in reversed(range(len(tokens) - len(q_enc) + 1)):
        if tokens[idx:idx+len(q_enc)] == q_enc:
            q_pos = idx + 1
            break

    # Terminal query prompt state (the state at the final prompt position '|')
    # This is the exact state that projects to LM head
    h_query_terminal = hidden[-1].detach()
    h_query_key = hidden[q_pos].detach() if q_pos != -1 else hidden[-1].detach()

    return {
        "terminal_query": h_query_terminal,
        "query_key": h_query_key,
        "k_reps": k_reps,
        "v_reps": v_reps,
        "assoc_reps": assoc_reps,
        "hidden": hidden,
    }


def compute_representation_retrieval_metrics(
    retrieved_rep: torch.Tensor,
    correct_value_rep: torch.Tensor,
    candidate_value_reps: Dict[str, torch.Tensor],
    correct_val_key: str,
) -> Tuple[float, float, float, int]:
    """Computes cosine similarity, contrastive margin, and representation rank."""
    cos = nn.CosineSimilarity(dim=0)
    
    sim_correct = float(cos(retrieved_rep, correct_value_rep).item())
    
    sim_others = []
    all_sims = [(sim_correct, correct_val_key)]
    
    for v_key, v_tensor in candidate_value_reps.items():
        if v_key != correct_val_key:
            s = float(cos(retrieved_rep, v_tensor).item())
            sim_others.append(s)
            all_sims.append((s, v_key))

    avg_incorrect = float(sum(sim_others) / len(sim_others)) if sim_others else 0.0
    max_incorrect = max(sim_others) if sim_others else 0.0
    margin = sim_correct - max_incorrect

    # Rank (1-indexed, descending by similarity)
    all_sims.sort(key=lambda x: x[0], reverse=True)
    rank = 1
    for idx, (s, vk) in enumerate(all_sims):
        if vk == correct_val_key:
            rank = idx + 1
            break

    return sim_correct, avg_incorrect, margin, rank


def evaluate_value_representation_retrieval(
    model: ChakrMicro,
    seed: int = 42,
    samples_per_split: int = 15,
) -> Step217RetrievalReport:
    """Evaluates representation retrieval across 4 conditions without vocabulary decoding."""
    torch.manual_seed(seed)
    rng = random.Random(seed)
    tok = get_default_tokenizer()
    model.eval()

    known_keys = ["A", "B", "C", "D", "E"]
    known_vals = ["1", "2", "3", "4", "5"]
    unseen_keys = ["P", "Q", "R", "S", "T"]
    unseen_vals = ["6", "7", "8", "9", "0"]

    splits_config = {
        "known_known": (known_keys, known_vals),
        "known_unseen": (known_keys, unseen_vals),
        "unseen_known": (unseen_keys, known_vals),
        "unseen_unseen": (unseen_keys, unseen_vals),
    }

    t0 = time.time()
    results = {}
    margins = []

    for split_name, (k_pool, v_pool) in splits_config.items():
        cos_corr_list = []
        cos_incorr_list = []
        margin_list = []
        rank_list = []
        acc_list = []

        for _ in range(samples_per_split):
            sampled_k = rng.sample(k_pool, 3)
            sampled_v = rng.sample(v_pool, 3)
            pairs = list(zip(sampled_k, sampled_v))
            rng.shuffle(pairs)

            query_pair = pairs[0]
            query_k, exp_v = query_pair

            parts = [f"|{k}| -> |{v}|" for k, v in pairs]
            prompt = "map " + " and ".join(parts) + f" query |{query_k}| -> |"

            extracted = extract_contextual_representations(
                model=model,
                tokenizer=tok,
                prompt=prompt,
                pairs=pairs,
                query_key=query_k,
            )

            v_reps = extracted["v_reps"]
            if exp_v not in v_reps or len(v_reps) < 2:
                continue

            # Retrieved value representation:
            # We examine the model's terminal query representation h_term
            # which is ChakrMicro's internal state directly prior to vocabulary projection.
            retrieved_rep = extracted["terminal_query"]
            correct_val_rep = v_reps[exp_v]

            sim_corr, sim_incorr, margin, rank = compute_representation_retrieval_metrics(
                retrieved_rep=retrieved_rep,
                correct_value_rep=correct_val_rep,
                candidate_value_reps=v_reps,
                correct_val_key=exp_v,
            )

            cos_corr_list.append(sim_corr)
            cos_incorr_list.append(sim_incorr)
            margin_list.append(margin)
            rank_list.append(rank)
            acc_list.append(1.0 if rank == 1 else 0.0)

        n = len(cos_corr_list)
        if n > 0:
            avg_corr = float(sum(cos_corr_list) / n)
            avg_incorr = float(sum(cos_incorr_list) / n)
            avg_margin = float(sum(margin_list) / n)
            avg_rank = float(sum(rank_list) / n)
            avg_acc = float(sum(acc_list) / n)
        else:
            avg_corr, avg_incorr, avg_margin, avg_rank, avg_acc = 0.0, 0.0, 0.0, 1.0, 0.0

        margins.append(avg_margin)
        results[split_name] = RepresentationRetrievalSplitResult(
            split_name=split_name,
            cosine_correct=avg_corr,
            cosine_incorrect=avg_incorr,
            contrastive_margin=avg_margin,
            representation_rank=avg_rank,
            representation_retrieval_acc=avg_acc,
            sample_count=n,
        )

    cpu_ms = (time.time() - t0) * 1000.0
    overall_mean_margin = float(sum(margins) / len(margins)) if margins else 0.0
    # Success threshold: unseen_unseen has rank significantly above chance (e.g. rank < 2.0 out of 3)
    # and positive contrastive margin
    unseen_res = results.get("unseen_unseen")
    is_success = bool(unseen_res and unseen_res.contrastive_margin > 0.0 and unseen_res.representation_rank < 2.0)

    return Step217RetrievalReport(
        seed=seed,
        splits=results,
        is_representation_retrieval_successful=is_success,
        overall_mean_margin=overall_mean_margin,
        cpu_runtime_ms=cpu_ms,
    )
