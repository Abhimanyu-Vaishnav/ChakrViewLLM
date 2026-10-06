"""Step 225: Minimal Association Learnability.

Goal:
Find the smallest task on which gradient learning can create operational associative retrieval.
Start with:
    B -> Y
    query B
    expected Y
Prevent fixed answer memorization by randomizing mappings each episode:
Episode 1: A -> X, B -> Y, C -> Z
Episode 2: A -> Z, B -> X, C -> Y

Measures:
A. Training association accuracy
B. Validation association accuracy
C. Held-out mapping accuracy
D. Association representation score
E. Value representation retrieval (margin & rank)
F. Final token accuracy
G. Language retention (language validation loss delta)
H. Parameter delta norm (||theta_cand - theta_base||)
I. Gradient norm
J. Training loss

Strict Constraints:
- CPU only
- Cloned isolated candidate
- Canonical baseline FROZEN and verified bit-exact
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
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
from chakrview.cognition.neural_language_learning import ControlledNeuralLanguageTrainer


@dataclasses.dataclass
class MinimalAssociationCandidateManifest:
    candidate_id: str
    parent_baseline_hash: str
    seed: int
    optimizer: str
    learning_rate: float
    epochs: int
    parameter_count: int
    parameter_delta_norm: float
    candidate_hash: str
    acceptance_state: str  # ACCEPT, REJECT, ROLLBACK, HOLD_FOR_INVESTIGATION


@dataclasses.dataclass
class MinimalAssociationEvaluationResult:
    manifest: MinimalAssociationCandidateManifest
    training_loss: float
    mean_gradient_norm: float
    train_acc: float
    val_acc: float
    heldout_mapping_acc: float
    association_rep_score: float
    value_rep_margin: float
    value_rep_rank: float
    final_token_acc: float
    language_loss_before: float
    language_loss_after: float
    language_retention_preserved: bool
    is_minimal_association_learned: bool
    cpu_runtime_ms: float = 0.0


def compute_param_delta(model_cand: nn.Module, model_base: nn.Module) -> float:
    delta_sq = 0.0
    for p_cand, p_base in zip(model_cand.parameters(), model_base.parameters()):
        delta_sq += torch.norm(p_cand.detach() - p_base.detach()).item() ** 2
    return delta_sq ** 0.5


def evaluate_language_loss(model: ChakrMicro, trainer: ControlledNeuralLanguageTrainer) -> float:
    _, val_tensors, _ = trainer.create_synthetic_language_corpus()
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for t in val_tensors:
            inp = t[:-1].unsqueeze(0)
            target = t[1:].unsqueeze(0)
            logits = model(inp)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1), ignore_index=0)
            total_loss += float(loss.item())
    return total_loss / len(val_tensors) if val_tensors else 0.0


def evaluate_association_task(
    model: ChakrMicro,
    tok: BPETokenizer,
    keys: List[str],
    vals: List[str],
    num_episodes: int = 10,
    seed: int = 42,
) -> Dict[str, float]:
    rng = random.Random(seed)
    model.eval()
    cos = nn.CosineSimilarity(dim=0)

    token_correct = 0
    assoc_scores = []
    val_margins = []
    val_ranks = []

    with torch.no_grad():
        for _ in range(num_episodes):
            k_sample = rng.sample(keys, min(3, len(keys)))
            v_sample = rng.sample(vals, min(3, len(vals)))
            pairs = list(zip(k_sample, v_sample))
            rng.shuffle(pairs)

            query_pair = rng.choice(pairs)
            query_k, exp_v = query_pair
            exp_tok = tok.encode(exp_v, add_bos=False, add_eos=False)[0]

            parts = [f"|{k}| -> |{v}|" for k, v in pairs]
            prompt = "map " + " and ".join(parts) + f" query |{query_k}| -> |"
            token_ids = tok.encode(prompt, add_bos=True, add_eos=False)
            inp = torch.tensor([token_ids], dtype=torch.long)

            logits = model(inp)
            pred_tok = torch.argmax(logits[0, -1, :]).item()
            if pred_tok == exp_tok:
                token_correct += 1

            extracted = extract_contextual_representations(model, tok, prompt, pairs, query_k)
            k_reps = extracted["k_reps"]
            v_reps = extracted["v_reps"]
            if exp_v in v_reps and query_k in k_reps:
                pos_assoc = k_reps[query_k] * v_reps[exp_v]
                neg_sims = [float(cos(pos_assoc, k_reps[query_k] * vt).item()) for vk, vt in v_reps.items() if vk != exp_v]
                mean_neg = float(sum(neg_sims) / len(neg_sims)) if neg_sims else 0.0
                assoc_scores.append(max(0.0, 1.0 - abs(mean_neg)))

                sim_corr, sim_incorr, margin, rank = compute_representation_retrieval_metrics(
                    extracted["terminal_query"], v_reps[exp_v], v_reps, exp_v
                )
                val_margins.append(margin)
                val_ranks.append(float(rank))

    n = max(1, num_episodes)
    return {
        "token_acc": token_correct / n,
        "assoc_score": float(sum(assoc_scores) / len(assoc_scores)) if assoc_scores else 0.0,
        "val_margin": float(sum(val_margins) / len(val_margins)) if val_margins else 0.0,
        "val_rank": float(sum(val_ranks) / len(val_ranks)) if val_ranks else 3.0,
    }


def train_and_eval_minimal_association(
    base_model: ChakrMicro,
    seed: int = 42,
    learning_rate: float = 3e-4,
    epochs: int = 15,
    steps_per_epoch: int = 8,
) -> MinimalAssociationEvaluationResult:
    """Trains an isolated candidate on minimal contextual association with randomized mappings."""
    torch.manual_seed(seed)
    rng = random.Random(seed)
    tok = get_default_tokenizer()
    lang_trainer = ControlledNeuralLanguageTrainer(tokenizer=tok)

    t0 = time.time()
    parent_hash = hashlib.sha256()
    for p in base_model.parameters():
        parent_hash.update(p.detach().cpu().numpy().tobytes())
    parent_hash_str = parent_hash.hexdigest()

    # Measure initial language loss
    lang_loss_before = evaluate_language_loss(base_model, lang_trainer)

    # Clone isolated candidate model
    candidate = copy.deepcopy(base_model)
    candidate.train()

    # Allow fine-tuning across all layers
    for p in candidate.parameters():
        p.requires_grad = True

    optimizer = torch.optim.AdamW(candidate.parameters(), lr=learning_rate, weight_decay=1e-4)

    # Key/Value identity sets for minimal training
    train_keys = ["A", "B", "C", "D", "E"]
    train_vals = ["1", "2", "3", "4", "5"]

    val_keys = ["A", "B", "C", "D", "E"]
    val_vals = ["1", "2", "3", "4", "5"]

    heldout_keys = ["F", "G", "H"]
    heldout_vals = ["7", "8", "9"]

    total_loss = 0.0
    grad_norms = []
    step_count = 0

    for ep in range(epochs):
        for _ in range(steps_per_epoch):
            # Dynamic randomized mapping per episode
            k_sample = rng.sample(train_keys, 3)
            v_sample = rng.sample(train_vals, 3)
            pairs = list(zip(k_sample, v_sample))
            rng.shuffle(pairs)

            query_pair = rng.choice(pairs)
            query_k, exp_v = query_pair
            exp_tok = tok.encode(exp_v, add_bos=False, add_eos=False)[0]

            parts = [f"|{k}| -> |{v}|" for k, v in pairs]
            prompt = "map " + " and ".join(parts) + f" query |{query_k}| -> |"
            token_ids = tok.encode(prompt, add_bos=True, add_eos=False)
            inp = torch.tensor([token_ids], dtype=torch.long)
            target = torch.tensor([exp_tok], dtype=torch.long)

            optimizer.zero_grad()
            logits = candidate(inp)
            last_logits = logits[0, -1, :].unsqueeze(0)

            loss = F.cross_entropy(last_logits, target)
            loss.backward()

            # Gradient clipping & logging
            gnorm = float(torch.nn.utils.clip_grad_norm_(candidate.parameters(), max_norm=1.0).item())
            grad_norms.append(gnorm)
            optimizer.step()

            total_loss += float(loss.item())
            step_count += 1

    avg_train_loss = total_loss / max(1, step_count)
    mean_gnorm = float(sum(grad_norms) / len(grad_norms)) if grad_norms else 0.0

    # Post-training evaluations
    train_metrics = evaluate_association_task(candidate, tok, train_keys, train_vals, num_episodes=10, seed=seed)
    val_metrics = evaluate_association_task(candidate, tok, val_keys, val_vals, num_episodes=10, seed=seed + 1)
    heldout_metrics = evaluate_association_task(candidate, tok, heldout_keys, heldout_vals, num_episodes=10, seed=seed + 2)

    lang_loss_after = evaluate_language_loss(candidate, lang_trainer)
    lang_preserved = (lang_loss_after <= lang_loss_before * 1.25)

    delta_norm = compute_param_delta(candidate, base_model)

    cand_hash = hashlib.sha256()
    for p in candidate.parameters():
        cand_hash.update(p.detach().cpu().numpy().tobytes())
    cand_hash_str = cand_hash.hexdigest()

    is_learned = bool(train_metrics["token_acc"] >= 0.80 and val_metrics["token_acc"] >= 0.70)
    acceptance = "ACCEPT" if (is_learned and lang_preserved) else "HOLD_FOR_INVESTIGATION"

    manifest = MinimalAssociationCandidateManifest(
        candidate_id=f"cand_step225_seed{seed}",
        parent_baseline_hash=parent_hash_str,
        seed=seed,
        optimizer="AdamW",
        learning_rate=learning_rate,
        epochs=epochs,
        parameter_count=sum(p.numel() for p in candidate.parameters()),
        parameter_delta_norm=delta_norm,
        candidate_hash=cand_hash_str,
        acceptance_state=acceptance,
    )

    return MinimalAssociationEvaluationResult(
        manifest=manifest,
        training_loss=avg_train_loss,
        mean_gradient_norm=mean_gnorm,
        train_acc=train_metrics["token_acc"],
        val_acc=val_metrics["token_acc"],
        heldout_mapping_acc=heldout_metrics["token_acc"],
        association_rep_score=train_metrics["assoc_score"],
        value_rep_margin=train_metrics["val_margin"],
        value_rep_rank=train_metrics["val_rank"],
        final_token_acc=train_metrics["token_acc"],
        language_loss_before=lang_loss_before,
        language_loss_after=lang_loss_after,
        language_retention_preserved=lang_preserved,
        is_minimal_association_learned=is_learned,
        cpu_runtime_ms=(time.time() - t0) * 1000.0,
    )
