"""
Batches 757-762: Autoregressive Plan DSL Training and Decoding Pipeline.
Location: experiments/batch741_780/train_autoregressive_planner.py

Features:
1. Teacher-Forced Autoregressive Training:
   - Input: Prompt + Target DSL (with loss computed strictly over target tokens, prompt masked with -100).
   - Loss: CrossEntropyLoss(ignore_index=-100).
2. Autoregressive Greedy / Beam Decoding:
   - Conditioned on prompt, decodes token-by-token until <END_PLAN> or max_new_tokens.
3. Multi-seed training (seed 42, seed 101) with AdamW optimizer.
4. Generates candidate checkpoint: experiments/batch741_780/checkpoints/candidate_step250_autoregressive_planner.pt.
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
from chakrview.cognition.plan_dsl import PlanDSLParser, PlanDSLValidator
from chakrview.tokenizer.serialization import load_tokenizer_artifacts

STARTING_CKPT = REPO_ROOT / "experiments" / "batch16_21_combined" / "main_training_10000" / "training_checkpoint_step_10000.pt"
TOKENIZER_DIR = REPO_ROOT / "release" / "chakrview-v0.5.0" / "data" / "experiments" / "vocab_4096"

TRAIN_DATA = REPO_ROOT / "experiments" / "batch741_780" / "data" / "train_plan_dsl.jsonl"
VAL_DATA = REPO_ROOT / "experiments" / "batch741_780" / "data" / "val_plan_dsl.jsonl"
TEST_DATA = REPO_ROOT / "experiments" / "batch741_780" / "data" / "test_plan_dsl_heldout.jsonl"

OUT_CKPT_PATH = REPO_ROOT / "experiments" / "batch741_780" / "checkpoints" / "candidate_step250_autoregressive_planner.pt"


def encode_training_pair(tokenizer: Any, prompt: str, target: str, max_seq_len: int = 512) -> Tuple[torch.Tensor, torch.Tensor]:
    p_tokens = [0] + tokenizer.encode(prompt)
    t_tokens = tokenizer.encode(target)

    # Input: p_tokens + t_tokens
    # Target: [-100]*len(p_tokens) + t_tokens
    full_input = p_tokens + t_tokens
    full_target = [-100] * len(p_tokens) + t_tokens

    if len(full_input) > max_seq_len:
        full_input = full_input[:max_seq_len]
        full_target = full_target[:max_seq_len]

    # For causal LM next-token prediction:
    # input_ids: full_input[:-1]
    # labels: full_target[1:]
    input_ids = torch.tensor(full_input[:-1], dtype=torch.long)
    labels = torch.tensor(full_target[1:], dtype=torch.long)
    return input_ids, labels


def generate_plan(model: ChakrMicro, tokenizer: Any, prompt: str, max_new_tokens: int = 150, device: torch.device = torch.device("cpu")) -> str:
    model.eval()
    p_tokens = [0] + tokenizer.encode(prompt)
    curr_ids = torch.tensor(p_tokens, dtype=torch.long).unsqueeze(0).to(device)

    generated_tokens: List[int] = []

    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(curr_ids)
            next_token_logits = logits[0, -1, :]
            next_tok = torch.argmax(next_token_logits).item()

            generated_tokens.append(next_tok)
            next_tensor = torch.tensor([[next_tok]], dtype=torch.long).to(device)
            curr_ids = torch.cat([curr_ids, next_tensor], dim=1)

            # Check for EOS or text <END_PLAN>
            decoded_so_far = tokenizer.decode(generated_tokens)
            if "<END_PLAN>" in decoded_so_far or next_tok == 0:
                break

    return tokenizer.decode(generated_tokens)


def evaluate_autoregressive_model(model: ChakrMicro, tokenizer: Any, test_items: List[Dict[str, Any]], device: torch.device) -> Dict[str, Any]:
    model.eval()
    total = len(test_items)
    syntax_valid_count = 0
    eos_terminated_count = 0
    exact_dsl_match_count = 0

    per_length_stats: Dict[int, Dict[str, int]] = {}

    for it in test_items:
        length = it.get("length", 2)
        if length not in per_length_stats:
            per_length_stats[length] = {"total": 0, "valid": 0, "eos": 0}
        per_length_stats[length]["total"] += 1

        prompt = it["prompt"]
        target = it["target_dsl"]

        gen_text = generate_plan(model, tokenizer, prompt, max_new_tokens=150, device=device)
        parsed = PlanDSLParser.parse(gen_text)

        if parsed.has_eos:
            eos_terminated_count += 1
            per_length_stats[length]["eos"] += 1

        if parsed.is_valid:
            syntax_valid_count += 1
            per_length_stats[length]["valid"] += 1

        if gen_text.strip() == target.strip():
            exact_dsl_match_count += 1

    return {
        "total": total,
        "syntax_validity_rate": round(syntax_valid_count / total * 100.0, 1),
        "eos_termination_rate": round(eos_terminated_count / total * 100.0, 1),
        "exact_dsl_match_rate": round(exact_dsl_match_count / total * 100.0, 1),
        "per_length_breakdown": per_length_stats,
    }


def train_autoregressive_candidate(seed: int = 42, steps: int = 250) -> Dict[str, Any]:
    torch.manual_seed(seed)
    random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer, _ = load_tokenizer_artifacts(TOKENIZER_DIR)

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
    # Filter state dict if wrapped
    state = raw_ckpt.get("model", raw_ckpt.get("model_state", raw_ckpt))
    clean_state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(clean_state, strict=False)
    model.to(device)

    # Load training and test items
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

    for step in range(1, steps + 1):
        it = random.choice(train_items)
        input_ids, labels = encode_training_pair(tokenizer, it["prompt"], it["target_dsl"])

        input_ids = input_ids.unsqueeze(0).to(device)
        labels = labels.unsqueeze(0).to(device)

        optimizer.zero_grad()
        logits = model(input_ids)  # [1, T, vocab_size]

        loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    duration = time.perf_counter() - t0

    # Evaluate on held-out test suite
    test_eval = evaluate_autoregressive_model(model, tokenizer, test_items, device)

    # Save Checkpoint
    os.makedirs(OUT_CKPT_PATH.parent, exist_ok=True)
    torch.save(model.state_dict(), OUT_CKPT_PATH)

    with open(OUT_CKPT_PATH, "rb") as f:
        ckpt_hash = hashlib.sha256(f.read()).hexdigest()

    return {
        "seed": seed,
        "steps": steps,
        "train_duration_sec": round(duration, 2),
        "checkpoint_path": str(OUT_CKPT_PATH),
        "checkpoint_sha256": ckpt_hash,
        "test_eval": test_eval,
    }


def main():
    print("=== Training Autoregressive Plan DSL Generator (Seed 42) ===")
    res42 = train_autoregressive_candidate(seed=42, steps=250)
    print(f"Seed 42 Complete in {res42['train_duration_sec']}s")
    print(f"  Test Syntax Validity Rate: {res42['test_eval']['syntax_validity_rate']}%")
    print(f"  Test EOS Termination Rate: {res42['test_eval']['eos_termination_rate']}%")
    print(f"  Test Length Breakdown:    {res42['test_eval']['per_length_breakdown']}")

    print("\n=== Training Autoregressive Plan DSL Generator (Seed 101) ===")
    res101 = train_autoregressive_candidate(seed=101, steps=250)
    print(f"Seed 101 Complete in {res101['train_duration_sec']}s")
    print(f"  Test Syntax Validity Rate: {res101['test_eval']['syntax_validity_rate']}%")
    print(f"  Test EOS Termination Rate: {res101['test_eval']['eos_termination_rate']}%")

    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seed_42": res42,
        "seed_101": res101,
    }

    report_path = REPO_ROOT / "experiments" / "batch741_780" / "reports" / "TRAINING_RESULTS.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport written to: {report_path}")


if __name__ == "__main__":
    main()
