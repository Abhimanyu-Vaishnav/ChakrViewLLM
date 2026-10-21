"""Evaluation Harness for ChakrView v0.8 Experiment 06 Minimal Copy Diagnostic.

Evaluates on:
- Train Split (100 examples: 50 Stage A, 50 Stage B).
- Validation Split (30 examples: 15 Stage A, 15 Stage B).
- Held-Out Test Split (50 examples: 25 Stage A, 25 Stage B).

Measures:
1. Exact Sequence Match (Free Greedy Autoregressive Generation).
2. Teacher-Forced Target-Token Accuracy and Per-Token Loss specifically across target response tokens.
3. First-Token Teacher-Forced Logit Index and Rank.
4. Stage A vs Stage B performance breakdown.
5. Qualitative evaluation of errors and generic output collapse.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.tokenizer.serialization import load_tokenizer_artifacts
from experiments.research_v08_json_sft.exp06_minimal_copy_diagnostic.verify_objective import evaluate_minimal_copy


def generate_greedy(
    model: ChakrMicro,
    tokenizer: Any,
    input_tokens: List[int],
    max_new_tokens: int = 16,
    stop_token: int = 1,
) -> Dict[str, Any]:
    curr = list(input_tokens)
    gen_tokens = []
    stop_reason = "MAX_TOKENS"

    for _ in range(max_new_tokens):
        x = torch.tensor([curr], dtype=torch.long)
        with torch.no_grad():
            logits = model(x)[0, -1, :]
        nxt = int(torch.argmax(logits).item())
        if nxt == stop_token:
            stop_reason = "STOP_TOKEN"
            break
        gen_tokens.append(nxt)
        curr.append(nxt)

    try:
        text = tokenizer.decode(gen_tokens, errors="replace")
    except Exception:
        text = tokenizer.decode(gen_tokens)

    return {
        "text": text,
        "tokens": gen_tokens,
        "stop_reason": stop_reason,
    }


def evaluate_split(
    model: ChakrMicro,
    tokenizer: Any,
    split_path: Path,
    split_name: str,
) -> Dict[str, Any]:
    records = []
    with open(split_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    loss_fn = nn.CrossEntropyLoss(reduction="none")

    stage_a_exact = 0
    stage_a_total = 0
    stage_b_exact = 0
    stage_b_total = 0

    stage_a_tf_matches = 0
    stage_a_tf_tokens = 0
    stage_b_tf_matches = 0
    stage_b_tf_tokens = 0

    evaluated_examples = []

    for rec in records:
        stage = rec["stage"]
        prompt = rec["prompt"]
        expected_val = rec["expected_value"]
        target_tokens = rec["target_tokens"]

        p_toks = tokenizer.encode(prompt)
        r_toks = tokenizer.encode(expected_val)
        inp = [0] + p_toks

        # 1. Free Greedy Generation
        gen_out = generate_greedy(model, tokenizer, inp, max_new_tokens=16)
        ev = evaluate_minimal_copy(gen_out["text"], expected_val)

        # 2. Teacher-Forced Pass
        tf_input = [0] + p_toks + r_toks
        tf_target = ([-100] * len(p_toks)) + r_toks + [1]
        assert len(tf_input) == len(tf_target)

        with torch.no_grad():
            logits = model(torch.tensor([tf_input], dtype=torch.long))[0]

        # First response token logit index is exactly len(p_toks)
        resp_start_idx = len(p_toks)
        first_token_target = r_toks[0]
        first_token_logits = logits[resp_start_idx, :]
        first_token_pred = int(torch.argmax(first_token_logits).item())
        first_token_rank = int((torch.argsort(first_token_logits, descending=True) == first_token_target).nonzero(as_tuple=True)[0].item()) + 1
        first_token_prob = round(F.softmax(first_token_logits, dim=-1)[first_token_target].item(), 4)

        # Token matches across response tokens
        resp_losses = []
        resp_matches = []
        for i in range(resp_start_idx, len(tf_input)):
            tgt = tf_target[i]
            pr = int(torch.argmax(logits[i, :]).item())
            l = loss_fn(logits[i:i+1, :], torch.tensor([tgt])).item()
            resp_losses.append(l)
            resp_matches.append(pr == tgt)

        # Tally metrics
        is_match = ev["exact_match"]
        tf_token_acc = round(sum(resp_matches) / len(resp_matches) * 100, 1)

        if stage == "Stage_A":
            stage_a_total += 1
            if is_match:
                stage_a_exact += 1
            stage_a_tf_matches += sum(resp_matches)
            stage_a_tf_tokens += len(resp_matches)
        else:
            stage_b_total += 1
            if is_match:
                stage_b_exact += 1
            stage_b_tf_matches += sum(resp_matches)
            stage_b_tf_tokens += len(resp_matches)

        evaluated_examples.append({
            "id": rec["id"],
            "stage": stage,
            "expected_value": expected_val,
            "raw_output": gen_out["text"],
            "exact_match": is_match,
            "first_token_rank": first_token_rank,
            "first_token_prob": first_token_prob,
            "teacher_forced_acc_pct": tf_token_acc,
            "mean_loss": round(sum(resp_losses) / len(resp_losses), 4),
        })

    summary = {
        "split_name": split_name,
        "total_records": len(records),
        "total_exact_match": stage_a_exact + stage_b_exact,
        "total_exact_pct": round((stage_a_exact + stage_b_exact) / max(1, len(records)) * 100, 1),
        "stage_a": {
            "total": stage_a_total,
            "exact_match": stage_a_exact,
            "exact_pct": round(stage_a_exact / max(1, stage_a_total) * 100, 1),
            "teacher_forced_token_acc_pct": round(stage_a_tf_matches / max(1, stage_a_tf_tokens) * 100, 1),
        },
        "stage_b": {
            "total": stage_b_total,
            "exact_match": stage_b_exact,
            "exact_pct": round(stage_b_exact / max(1, stage_b_total) * 100, 1),
            "teacher_forced_token_acc_pct": round(stage_b_tf_matches / max(1, stage_b_tf_tokens) * 100, 1),
        },
        "examples": evaluated_examples,
    }
    return summary


def run_full_evaluation():
    print("=" * 70)
    print("STEP 5: EVALUATION ON TRAIN, VAL, AND HELD-OUT TEST")
    print("=" * 70)

    device = torch.device("cpu")
    tok_dir = REPO_ROOT / "data" / "experiments" / "vocab_4096"
    tokenizer, _ = load_tokenizer_artifacts(tok_dir)

    exp06_dir = Path(__file__).resolve().parent
    best_ckpt_path = exp06_dir / "checkpoints" / "exp06_best_checkpoint.pt"

    model = ChakrMicro(ModelConfig(max_seq_len=512)).to(device)
    st = torch.load(best_ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(st["model_state_dict"])
    model.eval()

    train_path = exp06_dir / "data" / "train.jsonl"
    val_path = exp06_dir / "data" / "val.jsonl"
    test_path = exp06_dir / "data" / "test.jsonl"

    train_summary = evaluate_split(model, tokenizer, train_path, "Train")
    val_summary = evaluate_split(model, tokenizer, val_path, "Validation")
    test_summary = evaluate_split(model, tokenizer, test_path, "Held_Out_Test")

    for s in [train_summary, val_summary, test_summary]:
        print(f"\n--- {s['split_name']} Split ({s['total_records']} examples) ---")
        print(f"  Overall Exact Match: {s['total_exact_match']}/{s['total_records']} ({s['total_exact_pct']}%)")
        print(f"  Stage A (Single-Token): Exact={s['stage_a']['exact_match']}/{s['stage_a']['total']} ({s['stage_a']['exact_pct']}%) | TF Token Acc={s['stage_a']['teacher_forced_token_acc_pct']}%")
        print(f"  Stage B (Multi-Token):  Exact={s['stage_b']['exact_match']}/{s['stage_b']['total']} ({s['stage_b']['exact_pct']}%) | TF Token Acc={s['stage_b']['teacher_forced_token_acc_pct']}%")

    results_payload = {
        "checkpoint_path": str(best_ckpt_path),
        "train": train_summary,
        "validation": val_summary,
        "test": test_summary,
    }

    out_file = exp06_dir / "EVALUATION_RESULTS.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)
    print(f"\nResults saved to {out_file}")


if __name__ == "__main__":
    run_full_evaluation()
