"""
Batch 74 Rationale-Style Data Experiment Runner.

Compares Direct-Answer SFT vs Rationale-Style (Intermediate Step) SFT:
- Direct Answer: response = 'c<|im_end|>\n'
- Rationale Style: response = 'a + b = c. Answer: c<|im_end|>\n'
Evaluates on identical held-out validation and test sets (extracting the final answer).
"""

from __future__ import annotations

import json
from pathlib import Path
import re
import sys
import time
from typing import Any, Dict, List, Tuple

sys.path.insert(0, ".")
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.training.prompt_contract import (
    normalize_prompt_response_strings,
    build_causal_sequence,
)
from chakrview.training.weighted_loss import WeightedCausalLoss
from experiments.batch42_reproducible_evaluation.evaluation.evaluator import (
    load_tokenizer_artifacts,
    compute_file_sha256,
)

BASE_CKPT = Path("experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt")
TOKENIZER_DIR = "release/chakrview-v0.5.0/data/experiments/vocab_4096"
DATA_DIR = Path("experiments/batch71_74_controlled_learning/data")


def parse_final_answer(text: str) -> str:
    """Extracts final answer from either direct digit output or 'Answer: X'."""
    clean = text.replace("<|im_end|>", "").strip()
    # Check for 'Answer: X'
    match = re.search(r"Answer:\s*([0-9]+)", clean)
    if match:
        return match.group(1).strip()
    # Fallback to last digit in text
    digits = re.findall(r"[0-9]+", clean)
    if digits:
        return digits[-1]
    return clean


def build_rationale_examples(
    records: List[Dict[str, Any]],
    tokenizer: Any,
    use_rationale: bool = True,
) -> List[Dict[str, Any]]:
    examples = []
    for r in records:
        a, b = r["operands"]
        c = r["expected_answer"]
        prompt = r["prompt"]

        if use_rationale:
            # Intermediate step template
            response = f"{a} + {b} = {c}. Answer: {c}<|im_end|>\n"
        else:
            response = r["response"]

        p_norm, r_norm = normalize_prompt_response_strings(prompt, response, strip_delimiters=False)
        p_toks = tokenizer.encode(p_norm)
        r_toks = tokenizer.encode(r_norm)

        inp, tgt = build_causal_sequence(p_toks, r_toks, bos_id=0, eos_id=1, ignore_index=-100)

        examples.append({
            "id": r["id"],
            "prompt": prompt,
            "response": response,
            "expected_answer": c,
            "inp": inp,
            "tgt": tgt,
            "prompt_in": [0] + p_toks,
        })
    return examples


def run_rationale_experiment(
    condition_name: str,
    use_rationale: bool,
    steps: int = 100,
    batch_size: int = 2,
    lr: float = 5e-4,
    seed: int = 42,
    output_dir: Path = Path("experiments/batch71_74_controlled_learning/experiments"),
) -> Dict[str, Any]:
    tokenizer, _ = load_tokenizer_artifacts(TOKENIZER_DIR)
    train_recs = [json.loads(l) for l in open(DATA_DIR / "train_arithmetic.jsonl") if l.strip()]
    val_recs = [json.loads(l) for l in open(DATA_DIR / "val_arithmetic.jsonl") if l.strip()]
    test_recs = [json.loads(l) for l in open(DATA_DIR / "test_held_out.jsonl") if l.strip()]

    train_exs = build_rationale_examples(train_recs, tokenizer, use_rationale=use_rationale)
    # Validation & Test examples always evaluated on expected answer
    val_exs = build_rationale_examples(val_recs, tokenizer, use_rationale=use_rationale)
    test_exs = build_rationale_examples(test_recs, tokenizer, use_rationale=use_rationale)

    ckpt = torch.load(BASE_CKPT, map_location="cpu", weights_only=False)
    model = ChakrMicro(ModelConfig(max_seq_len=512))
    model.load_state_dict(ckpt["model_state_dict"])

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_ckpt = output_dir / f"sft_{condition_name}_step_{steps}.pt"

    start_time = time.perf_counter()
    loss_history = []

    model.train()
    for step in range(1, steps + 1):
        batch_indices = rng.randint(0, len(train_exs), size=batch_size)
        batch = [train_exs[i] for i in batch_indices]

        max_len = max(len(ex["inp"]) for ex in batch)
        pad_id = 2

        padded_inp = [ex["inp"] + [pad_id] * (max_len - len(ex["inp"])) for ex in batch]
        padded_tgt = [ex["tgt"] + [-100] * (max_len - len(ex["tgt"])) for ex in batch]

        x = torch.tensor(padded_inp, dtype=torch.long)
        y = torch.tensor(padded_tgt, dtype=torch.long)

        optimizer.zero_grad()
        logits = model(x)
        loss = loss_fn(logits.view(-1, 4096), y.view(-1))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        loss_history.append(loss.item())

    elapsed_sec = time.perf_counter() - start_time

    # Save checkpoint
    torch.save({
        "sft_step": steps,
        "condition": condition_name,
        "use_rationale": use_rationale,
        "model_state_dict": model.state_dict(),
        "seed": seed,
    }, out_ckpt)

    # Evaluate exact final answer match on all splits
    def eval_split(exs: List[Dict[str, Any]]) -> Dict[str, Any]:
        model.eval()
        exact_count = 0
        eos_count = 0
        detailed = []

        with torch.no_grad():
            for ex in exs:
                gen = list(ex["prompt_in"])
                eos_reached = False
                max_gen = 25 if use_rationale else 15

                for _ in range(max_gen):
                    logits_step = model(torch.tensor([gen]))[0, -1]
                    nxt = torch.argmax(logits_step).item()
                    gen.append(nxt)
                    if nxt == 1:
                        eos_reached = True
                        break

                raw_dec = tokenizer.decode(gen[len(ex["prompt_in"]):])
                parsed_ans = parse_final_answer(raw_dec)
                exact = (parsed_ans == ex["expected_answer"].strip())

                if exact:
                    exact_count += 1
                if eos_reached:
                    eos_count += 1

                detailed.append({
                    "id": ex["id"],
                    "expected": ex["expected_answer"],
                    "raw_generated": raw_dec,
                    "parsed_answer": parsed_ans,
                    "exact": exact,
                    "eos": eos_reached,
                })

        return {
            "total": len(exs),
            "exact_matches": exact_count,
            "exact_match_acc": round(exact_count / len(exs), 4),
            "eos_count": eos_count,
            "eos_rate": round(eos_count / len(exs), 4),
            "records": detailed,
        }

    return {
        "condition": condition_name,
        "use_rationale": use_rationale,
        "steps": steps,
        "elapsed_sec": round(elapsed_sec, 2),
        "initial_loss": round(loss_history[0], 4),
        "final_loss": round(loss_history[-1], 4),
        "checkpoint": str(out_ckpt),
        "train": eval_split(train_exs),
        "val": eval_split(val_exs),
        "test": eval_split(test_exs),
    }


if __name__ == "__main__":
    out_dir = Path("experiments/batch71_74_controlled_learning/experiments")
    print("--- Running Batch 74: Rationale-Style SFT (100 steps) ---")
    res_rationale = run_rationale_experiment("rationale_step100", use_rationale=True, steps=100, output_dir=out_dir)
    print("Rationale Train Acc:", res_rationale["train"]["exact_match_acc"], "Val Acc:", res_rationale["val"]["exact_match_acc"], "Test Acc:", res_rationale["test"]["exact_match_acc"])

    summary_file = out_dir / "batch74_rationale_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(res_rationale, f, indent=2)
    print(f"Summary written to {summary_file}")
