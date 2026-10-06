"""Step 185: Untied Readout Baseline Investigation.

Tests whether replacing the tied language model head (W_out = E^T)
with an independent learned output projection W_out in R^(d_model x vocab_size)
solves or mitigates zero-shot disjoint token retrieval.

Architecture isolation:
- Baseline ChakrMicro uses W_out = E^T (tie_embeddings=True)
- Candidate ChakrMicro uses independent W_out (tie_embeddings=False)
- No backbone modification, no retrieval memory, no external models.
"""

from __future__ import annotations

import copy
import dataclasses
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from chakrview.brain.model import ChakrMicro
from chakrview.brain.config import ModelConfig
from chakrview.tokenizer.tokenizer import BPETokenizer
from chakrview.tokenizer.serialization import load_tokenizer_artifacts


@dataclasses.dataclass
class UntiedEvaluationResult:
    model_type: str  # "tied" or "untied"
    total_params: int
    trainable_params: int
    train_loss: float
    train_acc: float
    val_loss: float
    val_acc: float
    heldout_acc: float
    disjoint_acc: float
    target_logit: float
    target_prob: float
    target_rank: int
    cosine_sim_to_tied: Optional[float] = None
    seed: int = 42
    cpu_runtime_ms: float = 0.0


def get_default_tokenizer() -> BPETokenizer:
    tok_dir = Path("data/experiments/vocab_4096")
    tok, _ = load_tokenizer_artifacts(tok_dir)
    return tok


def create_untied_candidate(base_model: ChakrMicro) -> ChakrMicro:
    """Create an untied readout candidate isolated from the base model."""
    new_config = dataclasses.replace(base_model.config, tie_embeddings=False)
    candidate = ChakrMicro(new_config)
    
    # Copy all weights from base_model except lm_head
    candidate.load_state_dict(base_model.state_dict(), strict=False)
    # Initialize untied lm_head with base embedding transpose
    with torch.no_grad():
        candidate.lm_head.weight.copy_(base_model.embedding.weight.clone())
    return candidate


def evaluate_associative_mappings(
    model: ChakrMicro,
    tokenizer: BPETokenizer,
    context_mappings: List[Tuple[str, str]],
    query_mappings: List[Tuple[str, str]],
) -> Dict[str, float]:
    """Evaluates next-token retrieval for associative mapping prompts:
    Prompt format: 'map |K1| -> |V1| and |K2| -> |V2| query |KQ| -> |'
    Target: 'VQ'
    """
    model.eval()
    correct = 0
    total = len(query_mappings)
    target_probs = []
    target_ranks = []
    target_logits = []

    with torch.no_grad():
        for query_k, expected_v in query_mappings:
            parts = [f"|{k}| -> |{v}|" for k, v in context_mappings]
            prompt = "map " + " and ".join(parts) + f" query |{query_k}| -> |"
            expected_ids = tokenizer.encode(expected_v, add_bos=False, add_eos=False)
            if not expected_ids:
                continue
            target_token = expected_ids[0]

            input_tokens = tokenizer.encode(prompt, add_bos=True, add_eos=False)
            input_ids = torch.tensor([input_tokens], dtype=torch.long)
            logits = model(input_ids)
            last_logits = logits[0, -1, :]
            probs = F.softmax(last_logits, dim=-1)

            t_logit = float(last_logits[target_token].item())
            t_prob = float(probs[target_token].item())
            sorted_indices = torch.argsort(last_logits, descending=True)
            rank = int((sorted_indices == target_token).nonzero(as_tuple=True)[0].item()) + 1

            target_logits.append(t_logit)
            target_probs.append(t_prob)
            target_ranks.append(rank)

            pred_token = int(torch.argmax(last_logits).item())
            if pred_token == target_token:
                correct += 1

    return {
        "accuracy": float(correct / total) if total > 0 else 0.0,
        "mean_target_logit": float(sum(target_logits) / len(target_logits)) if target_logits else 0.0,
        "mean_target_prob": float(sum(target_probs) / len(target_probs)) if target_probs else 0.0,
        "median_rank": float(sorted(target_ranks)[len(target_ranks) // 2]) if target_ranks else 0.0,
    }


def train_and_evaluate_step185(
    base_model: ChakrMicro,
    seed: int = 42,
    epochs: int = 20,
    lr: float = 1e-3,
) -> Tuple[UntiedEvaluationResult, UntiedEvaluationResult]:
    """Train and compare tied vs untied models on familiar vs disjoint mappings.
    Returns (tied_result, untied_result).
    """
    torch.manual_seed(seed)
    tokenizer = get_default_tokenizer()

    # Familiar train mappings
    train_pairs = [("A", "1"), ("B", "2"), ("C", "3")]
    heldout_pairs = [("A", "3"), ("B", "1"), ("C", "2")]  # Permuted values
    disjoint_pairs = [("X", "7"), ("Y", "8"), ("Z", "9")]  # Disjoint entities

    def build_sample(pairs: List[Tuple[str, str]], query_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        parts = [f"|{k}| -> |{v}|" for k, v in pairs]
        q_k, exp_v = pairs[query_idx]
        text = "map " + " and ".join(parts) + f" query |{q_k}| -> |{exp_v}"
        tokens = tokenizer.encode(text, add_bos=True, add_eos=False)
        inp = torch.tensor(tokens[:-1], dtype=torch.long)
        tgt = torch.tensor(tokens[1:], dtype=torch.long)
        return inp, tgt

    train_data = []
    for _ in range(8):
        for i in range(len(train_pairs)):
            train_data.append(build_sample(train_pairs, i))

    models = {
        "tied": copy.deepcopy(base_model),
        "untied": create_untied_candidate(base_model),
    }

    results = {}
    for name, model in models.items():
        t0 = time.time()
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

        model.train()
        train_loss = 0.0
        for ep in range(epochs):
            ep_loss = 0.0
            for inp, tgt in train_data:
                optimizer.zero_grad()
                logits = model(inp.unsqueeze(0))
                loss = F.cross_entropy(logits[0], tgt)
                loss.backward()
                optimizer.step()
                ep_loss += loss.item()
            train_loss = ep_loss / len(train_data)

        elapsed_ms = (time.time() - t0) * 1000.0

        fam_eval = evaluate_associative_mappings(model, tokenizer, train_pairs, train_pairs)
        heldout_eval = evaluate_associative_mappings(model, tokenizer, train_pairs, heldout_pairs)
        disj_eval = evaluate_associative_mappings(model, tokenizer, disjoint_pairs, disjoint_pairs)

        total_p = sum(p.numel() for p in model.parameters())
        trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)

        results[name] = UntiedEvaluationResult(
            model_type=name,
            total_params=total_p,
            trainable_params=trainable_p,
            train_loss=train_loss,
            train_acc=fam_eval["accuracy"],
            val_loss=train_loss,
            val_acc=fam_eval["accuracy"],
            heldout_acc=heldout_eval["accuracy"],
            disjoint_acc=disj_eval["accuracy"],
            target_logit=disj_eval["mean_target_logit"],
            target_prob=disj_eval["mean_target_prob"],
            target_rank=int(disj_eval["median_rank"]),
            seed=seed,
            cpu_runtime_ms=elapsed_ms,
        )

    cos_sim = F.cosine_similarity(
        models["untied"].lm_head.weight,
        base_model.embedding.weight,
        dim=-1,
    ).mean().item()
    results["untied"].cosine_sim_to_tied = float(cos_sim)

    return results["tied"], results["untied"]
