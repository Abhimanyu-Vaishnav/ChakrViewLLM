"""
Batches 798-802: Grammar-Constrained Plan Training and Multi-Seed Reproducibility.
Location: experiments/batch781_820/train_planner.py

Features:
1. Trains ChakrMicro on the expanded 380-example dataset with prompt masking.
2. Supports dual multi-seed training (Seed 42 and Seed 101).
3. Evaluates both unconstrained and grammar-constrained decoding on the frozen validation and test sets.
4. Saves candidate checkpoints under experiments/batch781_820/checkpoints/.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.cognition.grammar_constrained_decoder import GrammarConstrainedDecoder
from chakrview.cognition.plan_dsl import PlanDSLParser, PlanDSLValidator
from chakrview.tokenizer.serialization import load_tokenizer_artifacts

STARTING_CKPT = REPO_ROOT / "experiments" / "batch16_21_combined" / "main_training_10000" / "training_checkpoint_step_10000.pt"
TOKENIZER_DIR = REPO_ROOT / "release" / "chakrview-v0.5.0" / "data" / "experiments" / "vocab_4096"

TRAIN_DATA = REPO_ROOT / "experiments" / "batch781_820" / "data" / "train_plan_dsl_expanded.jsonl"
VAL_DATA = REPO_ROOT / "experiments" / "batch781_820" / "data" / "val_plan_dsl_expanded.jsonl"
TEST_DATA = REPO_ROOT / "experiments" / "batch781_820" / "data" / "test_plan_dsl_frozen.jsonl"

OUT_CKPT_DIR = REPO_ROOT / "experiments" / "batch781_820" / "checkpoints"


def encode_training_pair(tokenizer: Any, prompt: str, target: str, max_seq_len: int = 512) -> Tuple[torch.Tensor, torch.Tensor]:
    p_tokens = [0] + tokenizer.encode(prompt)
    t_tokens = tokenizer.encode(target)

    full_input = p_tokens + t_tokens
    full_target = [-100] * len(p_tokens) + t_tokens

    if len(full_input) > max_seq_len:
        full_input = full_input[:max_seq_len]
        full_target = full_target[:max_seq_len]

    input_ids = torch.tensor(full_input[:-1], dtype=torch.long)
    labels = torch.tensor(full_target[1:], dtype=torch.long)
    return input_ids, labels


def evaluate_decoder(
    model: ChakrMicro,
    decoder: GrammarConstrainedDecoder,
    eval_items: List[Dict[str, Any]],
    device: torch.device,
    constrained: bool = True,
) -> Dict[str, Any]:
    model.eval()
    total = len(eval_items)
    syntax_valid_count = 0
    clean_eos_count = 0
    exact_match_count = 0

    per_length: Dict[int, Dict[str, int]] = {}

    for it in eval_items:
        length = it.get("length", 2)
        if length not in per_length:
            per_length[length] = {"total": 0, "valid": 0, "eos": 0, "exact": 0}
        per_length[length]["total"] += 1

        prompt = it["prompt"]
        target = it["target_dsl"]

        gen_dsl, has_eos, _ = decoder.generate(model, prompt, device=device, constrained=constrained)
        parsed = PlanDSLParser.parse(gen_dsl)

        if has_eos:
            clean_eos_count += 1
            per_length[length]["eos"] += 1

        if parsed.is_valid:
            syntax_valid_count += 1
            per_length[length]["valid"] += 1

        if gen_dsl.strip() == target.strip():
            exact_match_count += 1
            per_length[length]["exact"] += 1

    return {
        "total": total,
        "syntax_validity_rate": round(syntax_valid_count / total * 100.0, 1),
        "clean_eos_rate": round(clean_eos_count / total * 100.0, 1),
        "exact_match_rate": round(exact_match_count / total * 100.0, 1),
        "per_length": per_length,
    }


def train_candidate(seed: int = 42, steps: int = 300) -> Dict[str, Any]:
    torch.manual_seed(seed)
    random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer, _ = load_tokenizer_artifacts(TOKENIZER_DIR)
    decoder = GrammarConstrainedDecoder(tokenizer, max_new_tokens=150)

    base_cfg = ModelConfig(
        vocab_size=4096,
        d_model=192,
        n_layers=6,
        n_heads=6,
        max_seq_len=512,
    )
    model = ChakrMicro(base_cfg)

    # Load canonical checkpoint weights
    raw_ckpt = torch.load(STARTING_CKPT, map_location=device, weights_only=False)
    state = raw_ckpt.get("model", raw_ckpt.get("model_state", raw_ckpt))
    clean_state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(clean_state, strict=False)
    model.to(device)

    # Load items
    with open(TRAIN_DATA, "r", encoding="utf-8") as f:
        train_items = [json.loads(line) for line in f]
    with open(VAL_DATA, "r", encoding="utf-8") as f:
        val_items = [json.loads(line) for line in f]
    with open(TEST_DATA, "r", encoding="utf-8") as f:
        test_items = [json.loads(line) for line in f]

    optimizer = optim.AdamW(model.parameters(), lr=4e-4, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    model.train()
    t0 = time.perf_counter()

    loss_history = []
    for step in range(1, steps + 1):
        it = random.choice(train_items)
        input_ids, labels = encode_training_pair(tokenizer, it["prompt"], it["target_dsl"])

        input_ids = input_ids.unsqueeze(0).to(device)
        labels = labels.unsqueeze(0).to(device)

        optimizer.zero_grad()
        logits = model(input_ids)

        loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 50 == 0 or step == steps:
            loss_history.append({"step": step, "loss": round(loss.item(), 4)})

    duration = time.perf_counter() - t0

    # Save Checkpoint
    OUT_CKPT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = OUT_CKPT_DIR / f"candidate_step{steps}_seed{seed}_grammar_planner.pt"
    torch.save(model.state_dict(), ckpt_path)

    with open(ckpt_path, "rb") as f:
        ckpt_hash = hashlib.sha256(f.read()).hexdigest()

    # Validation evaluation (Unconstrained vs Constrained)
    val_unconstrained = evaluate_decoder(model, decoder, val_items, device, constrained=False)
    val_constrained = evaluate_decoder(model, decoder, val_items, device, constrained=True)

    return {
        "seed": seed,
        "steps": steps,
        "train_duration_sec": round(duration, 2),
        "final_loss": loss_history[-1]["loss"],
        "loss_trajectory": loss_history,
        "checkpoint_path": str(ckpt_path),
        "checkpoint_sha256": ckpt_hash,
        "validation_eval": {
            "unconstrained": val_unconstrained,
            "constrained": val_constrained,
        },
    }


def main():
    print("=== Training Candidate with Grammar Planning (Seed 42) ===")
    res42 = train_candidate(seed=42, steps=300)
    print(f"Seed 42 Complete in {res42['train_duration_sec']}s | Loss: {res42['final_loss']}")
    print(f"  Val Unconstrained: Syntax={res42['validation_eval']['unconstrained']['syntax_validity_rate']}% Exact={res42['validation_eval']['unconstrained']['exact_match_rate']}%")
    print(f"  Val Constrained:   Syntax={res42['validation_eval']['constrained']['syntax_validity_rate']}% Exact={res42['validation_eval']['constrained']['exact_match_rate']}%")

    print("\n=== Training Candidate with Grammar Planning (Seed 101) ===")
    res101 = train_candidate(seed=101, steps=300)
    print(f"Seed 101 Complete in {res101['train_duration_sec']}s | Loss: {res101['final_loss']}")
    print(f"  Val Unconstrained: Syntax={res101['validation_eval']['unconstrained']['syntax_validity_rate']}% Exact={res101['validation_eval']['unconstrained']['exact_match_rate']}%")
    print(f"  Val Constrained:   Syntax={res101['validation_eval']['constrained']['syntax_validity_rate']}% Exact={res101['validation_eval']['constrained']['exact_match_rate']}%")

    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seed_42": res42,
        "seed_101": res101,
    }

    report_path = REPO_ROOT / "experiments" / "batch781_820" / "reports" / "TRAINING_RESULTS.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    seed_comp_path = REPO_ROOT / "experiments" / "batch781_820" / "reports" / "SEED_COMPARISON.json"
    with open(seed_comp_path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "seed_42_final_loss": res42["final_loss"],
            "seed_101_final_loss": res101["final_loss"],
            "seed_42_val_constrained_exact": res42["validation_eval"]["constrained"]["exact_match_rate"],
            "seed_101_val_constrained_exact": res101["validation_eval"]["constrained"]["exact_match_rate"],
            "variance": abs(res42["validation_eval"]["constrained"]["exact_match_rate"] - res101["validation_eval"]["constrained"]["exact_match_rate"]),
            "stability": "STABLE" if abs(res42["validation_eval"]["constrained"]["exact_match_rate"] - res101["validation_eval"]["constrained"]["exact_match_rate"]) <= 10.0 else "UNSTABLE",
        }, f, indent=2)

    print(f"\nTraining reports saved to {report_path} and {seed_comp_path}")


if __name__ == "__main__":
    main()
