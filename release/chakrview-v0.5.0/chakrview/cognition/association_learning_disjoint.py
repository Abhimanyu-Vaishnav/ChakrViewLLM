"""Step 229: Disjoint Associative Generalization & Step 237: Disjoint Associative Learning Gate.

Implements:
- Step 229: evaluate_disjoint_associative_generalization()
- Step 237: run_disjoint_associative_learning_gate()
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
from chakrview.cognition.disjoint_value_transfer import (
    evaluate_disjoint_value_transfer,
    DisjointTransferReport,
    DisjointSplitMetrics,
)
from chakrview.cognition.value_representation_retrieval import get_default_tokenizer
from chakrview.cognition.association_generalization import (
    evaluate_contextual_association_generalization,
    ContextualGeneralizationSplitResult,
)
from chakrview.cognition.association_learning_minimal import (
    evaluate_language_loss,
    compute_param_delta,
)
from chakrview.cognition.neural_language_learning import ControlledNeuralLanguageTrainer


# ---------------------------------------------------------------------------
# Step 229 Types & Functions
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class DisjointGeneralizationSeedResult:
    seed: int
    train_loss: float
    splits: Dict[str, DisjointSplitMetrics]
    unseen_unseen_assoc_score: float
    unseen_unseen_value_margin: float
    unseen_unseen_token_acc: float
    is_seed_i3_qualified: bool


@dataclasses.dataclass
class DisjointGeneralizationReport:
    seed_results: Dict[int, DisjointGeneralizationSeedResult]
    all_seeds_qualified: bool
    i3_promotion_eligible: bool
    summary_verdict: str
    cpu_runtime_ms: float = 0.0


def train_and_eval_disjoint_candidate(
    base_model: ChakrMicro,
    seed: int = 42,
    epochs: int = 10,
    steps_per_epoch: int = 6,
    lr: float = 3e-4,
) -> DisjointGeneralizationSeedResult:
    """Trains an isolated candidate on Keys A-E / Values 1-5, then tests on P-T / 6-0 (Step 229)."""
    torch.manual_seed(seed)
    rng = random.Random(seed)
    tok = get_default_tokenizer()

    candidate = copy.deepcopy(base_model)
    candidate.train()
    for p in candidate.parameters():
        p.requires_grad = True

    opt = torch.optim.AdamW(candidate.parameters(), lr=lr, weight_decay=1e-4)

    train_keys = ["A", "B", "C", "D", "E"]
    train_vals = ["1", "2", "3", "4", "5"]

    total_loss = 0.0
    step_count = 0

    for _ in range(epochs):
        for _ in range(steps_per_epoch):
            ks = rng.sample(train_keys, 3)
            vs = rng.sample(train_vals, 3)
            pairs = list(zip(ks, vs))
            rng.shuffle(pairs)

            query_pair = rng.choice(pairs)
            query_k, exp_v = query_pair
            exp_tok = tok.encode(exp_v, add_bos=False, add_eos=False)[0]

            prompt = "map " + " and ".join([f"|{k}| -> |{v}|" for k, v in pairs]) + f" query |{query_k}| -> |"
            token_ids = tok.encode(prompt, add_bos=True, add_eos=False)
            inp = torch.tensor([token_ids], dtype=torch.long)
            target = torch.tensor([exp_tok], dtype=torch.long)

            opt.zero_grad()
            logits = candidate(inp)
            loss = F.cross_entropy(logits[0, -1, :].unsqueeze(0), target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(candidate.parameters(), 1.0)
            opt.step()

            total_loss += float(loss.item())
            step_count += 1

    candidate.eval()

    report_trans = evaluate_disjoint_value_transfer(candidate, seed=seed, samples_per_split=10)
    unseen_split = report_trans.splits.get("unseen_unseen")

    u_assoc = unseen_split.association_score if unseen_split else 0.0
    u_val_margin = unseen_split.value_rep_margin if unseen_split else 0.0
    u_tok_acc = unseen_split.final_token_acc if unseen_split else 0.0

    is_qualified = (u_tok_acc >= 0.50 and u_val_margin > 0.0)

    return DisjointGeneralizationSeedResult(
        seed=seed,
        train_loss=total_loss / max(1, step_count),
        splits=report_trans.splits,
        unseen_unseen_assoc_score=u_assoc,
        unseen_unseen_value_margin=u_val_margin,
        unseen_unseen_token_acc=u_tok_acc,
        is_seed_i3_qualified=is_qualified,
    )


def evaluate_disjoint_associative_generalization(
    base_model: ChakrMicro,
    seeds: Optional[List[int]] = None,
) -> DisjointGeneralizationReport:
    """Evaluates multi-seed disjoint generalization across seeds 42, 101, 2026 (Step 229)."""
    if seeds is None:
        seeds = [42, 101, 2026]

    t0 = time.time()
    seed_res = {}
    for s in seeds:
        seed_res[s] = train_and_eval_disjoint_candidate(base_model, seed=s)

    all_qual = all(r.is_seed_i3_qualified for r in seed_res.values())
    elapsed = (time.time() - t0) * 1000.0

    verdict = "I3_ELIGIBLE" if all_qual else "I3_PROMOTION_DENIED_ZERO_DISJOINT_RETRIEVAL"

    return DisjointGeneralizationReport(
        seed_results=seed_res,
        all_seeds_qualified=all_qual,
        i3_promotion_eligible=all_qual,
        summary_verdict=verdict,
        cpu_runtime_ms=elapsed,
    )


# ---------------------------------------------------------------------------
# Step 237 Types & Functions
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class DisjointSeedEvaluationResult:
    seed: int
    train_loss: float
    train_acc: float
    unseen_unseen_acc: float
    unseen_unseen_value_margin: float
    unseen_unseen_value_rank: float
    unseen_unseen_assoc_score: float
    language_loss_before: float
    language_loss_after: float
    is_seed_i3_qualified: bool
    splits: Dict[str, ContextualGeneralizationSplitResult]


@dataclasses.dataclass
class DisjointAssociativeLearningGateReport:
    seed_evaluations: Dict[int, DisjointSeedEvaluationResult]
    is_i3_candidate_achieved: bool
    mean_unseen_unseen_acc: float
    summary_verdict: str
    cpu_runtime_ms: float = 0.0


def train_and_eval_disjoint_learning_candidate(
    base_model: ChakrMicro,
    seed: int = 42,
    epochs: int = 12,
    steps_per_epoch: int = 6,
    lr: float = 5e-4,
) -> DisjointSeedEvaluationResult:
    """Trains an isolated candidate on randomized mappings and evaluates disjoint splits (Step 237)."""
    torch.manual_seed(seed)
    rng = random.Random(seed)
    tok = get_default_tokenizer()
    lang_trainer = ControlledNeuralLanguageTrainer(tokenizer=tok)

    cand = copy.deepcopy(base_model)
    cand.train()
    for p in cand.parameters():
        p.requires_grad = True

    opt = torch.optim.AdamW(cand.parameters(), lr=lr, weight_decay=1e-4)

    train_keys = ["A", "B", "C", "D", "E"]
    train_vals = ["1", "2", "3", "4", "5"]

    lang_before = evaluate_language_loss(base_model, lang_trainer)

    total_loss = 0.0
    step_count = 0

    for _ in range(epochs):
        for _ in range(steps_per_epoch):
            num_pairs = rng.choice([2, 3])
            k_sample = rng.sample(train_keys, num_pairs)
            v_sample = rng.sample(train_vals, num_pairs)
            pairs = list(zip(k_sample, v_sample))
            rng.shuffle(pairs)

            query_pair = rng.choice(pairs)
            qk, ev = query_pair
            exp_tok = tok.encode(ev, add_bos=False, add_eos=False)[0]

            prompt = "map " + " and ".join([f"|{k}| -> |{v}|" for k, v in pairs]) + f" query |{qk}| -> |"
            token_ids = tok.encode(prompt, add_bos=True, add_eos=False)
            inp = torch.tensor([token_ids], dtype=torch.long)
            target = torch.tensor([exp_tok], dtype=torch.long)

            opt.zero_grad()
            logits = cand(inp)
            loss = F.cross_entropy(logits[0, -1, :].unsqueeze(0), target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(cand.parameters(), 1.0)
            opt.step()

            total_loss += float(loss.item())
            step_count += 1

    cand.eval()

    gen_report = evaluate_contextual_association_generalization(cand, seed=seed, samples_per_split=15)
    lang_after = evaluate_language_loss(cand, lang_trainer)

    kk = gen_report.splits["known_known"]
    uu = gen_report.splits["unseen_unseen"]

    is_qualified = (uu.token_accuracy >= 0.50 and uu.value_rep_margin > 0.0)

    return DisjointSeedEvaluationResult(
        seed=seed,
        train_loss=total_loss / max(1, step_count),
        train_acc=kk.token_accuracy,
        unseen_unseen_acc=uu.token_accuracy,
        unseen_unseen_value_margin=uu.value_rep_margin,
        unseen_unseen_value_rank=uu.value_rep_rank,
        unseen_unseen_assoc_score=uu.association_score,
        language_loss_before=lang_before,
        language_loss_after=lang_after,
        is_seed_i3_qualified=is_qualified,
        splits=gen_report.splits,
    )


def run_disjoint_associative_learning_gate(
    base_model: ChakrMicro,
    seeds: Optional[List[int]] = None,
) -> DisjointAssociativeLearningGateReport:
    """Evaluates multi-seed disjoint associative learning across seeds 42, 101, 2026 (Step 237)."""
    if seeds is None:
        seeds = [42, 101, 2026]

    t0 = time.time()
    seed_res = {}
    uu_accs = []

    for s in seeds:
        res = train_and_eval_disjoint_learning_candidate(base_model, seed=s)
        seed_res[s] = res
        uu_accs.append(res.unseen_unseen_acc)

    all_qual = all(r.is_seed_i3_qualified for r in seed_res.values())
    mean_uu = float(sum(uu_accs) / len(uu_accs)) if uu_accs else 0.0

    verdict = "I3_CANDIDATE_ACHIEVED" if all_qual else "I3_DENIED_DISJOINT_RETRIEVAL_ZERO"
    elapsed = (time.time() - t0) * 1000.0

    return DisjointAssociativeLearningGateReport(
        seed_evaluations=seed_res,
        is_i3_candidate_achieved=all_qual,
        mean_unseen_unseen_acc=mean_uu,
        summary_verdict=verdict,
        cpu_runtime_ms=elapsed,
    )
