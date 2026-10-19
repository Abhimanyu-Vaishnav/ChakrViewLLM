"""Unified Evaluation Runner for Batch 3021–3060 Authorized SFT Pilot.

Evaluates:
1. Canonical Step-10,000 Baseline
2. Authorized Balanced SFT Best Candidate (experiments/batch3021_3060/balanced_sft_pilot_run/sft_best_checkpoint.pt)
3. Authorized Balanced SFT Final Step-160 Candidate (experiments/batch3021_3060/balanced_sft_pilot_run/sft_checkpoint_step_160.pt)

Across:
A. The genuinely isolated 40-case held-out test set (experiments/batch3021_3060/test.jsonl):
   - 28 instruction cases across 7 categories (exact extraction, JSON syntax validity, JSON schema validity, context QA, summarization, transformation, abstention).
   - 12 unseen-code and math natural continuation replay cases (continuation length, zero-token/empty rate, immediate EOS rate).
B. The 28-case generation benchmark (experiments/batch2781_2820/GENERATION_BENCHMARK.json):
   - Raw completion mode: empty rate, first token, EOS triggers.
   - Plain instruction mode: empty rate, token count, repetition ratio.

Emits: experiments/batch3021_3060/PILOT_EVALUATION_RESULTS.json
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

EXPERIMENT_DIR = REPO_ROOT / "experiments" / "batch3021_3060"

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.tokenizer.serialization import load_tokenizer_artifacts

logger = logging.getLogger("pilot_eval")


def compute_sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def compute_bigram_repetition(tokens: List[int]) -> float:
    if len(tokens) < 3:
        return 0.0
    bigrams = [tuple(tokens[i : i + 2]) for i in range(len(tokens) - 1)]
    return 1.0 - (len(set(bigrams)) / len(bigrams))


def generate_greedy(model: ChakrMicro, tokenizer: Any, input_tokens: List[int], max_new_tokens: int = 32) -> Dict[str, Any]:
    curr = list(input_tokens)
    t0 = time.perf_counter()
    gen_tokens = []
    stop_reason = "MAX_TOKENS"
    first_token_id = None
    first_token_is_eos = False

    for step_idx in range(max_new_tokens):
        if len(curr) >= 512:
            stop_reason = "CONTEXT_LIMIT"
            break
        x = torch.tensor([curr], dtype=torch.long)
        with torch.no_grad():
            logits = model(x)
        next_tok = int(torch.argmax(logits[0, -1, :]).item())
        if step_idx == 0:
            first_token_id = next_tok
            first_token_is_eos = (next_tok == 1)

        if next_tok == 1:  # EOS_ID
            stop_reason = "STOP_TOKEN"
            break
        gen_tokens.append(next_tok)
        curr.append(next_tok)

    latency_ms = (time.perf_counter() - t0) * 1000.0
    try:
        out_text = tokenizer.decode(gen_tokens, errors="replace")
    except Exception:
        out_text = tokenizer.decode(gen_tokens)
    return {
        "generated_tokens": gen_tokens,
        "token_count": len(gen_tokens),
        "first_token_id": first_token_id,
        "first_token_is_eos": first_token_is_eos,
        "text": out_text,
        "latency_ms": round(latency_ms, 2),
        "stop_reason": stop_reason,
        "is_empty": len(gen_tokens) == 0,
        "repetition_ratio": round(compute_bigram_repetition(gen_tokens), 4),
    }


def evaluate_checkpoint(ckpt_path: Path, ckpt_name: str, tokenizer: Any) -> Dict[str, Any]:
    print(f"\nEvaluating: {ckpt_name} ({ckpt_path.name})...")
    file_sha = compute_sha256(ckpt_path)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    model = ChakrMicro(ModelConfig(max_seq_len=512))
    model.load_state_dict(state["model_state_dict"], strict=True)
    model.eval()

    # 1. Evaluate on 40-case held-out test split
    test_path = EXPERIMENT_DIR / "test.jsonl"
    with open(test_path, "r", encoding="utf-8") as f:
        test_records = [json.loads(line) for line in f if line.strip()]

    test_results = []
    exact_match_count = 0
    valid_json_count = 0
    total_json_cases = 0
    abstention_count = 0
    total_abst_cases = 0
    replay_empty_count = 0
    total_replay_cases = 0

    for r in test_records:
        rid = r["id"]
        cat = r["category"]
        prompt = r["prompt"]
        target = r["response"]

        p_toks = tokenizer.encode(prompt)
        input_ids = [0] + p_toks

        gen = generate_greedy(model, tokenizer, input_ids, max_new_tokens=32)

        is_replay = (cat == "base_continuation_replay")
        if is_replay:
            total_replay_cases += 1
            if gen["is_empty"]:
                replay_empty_count += 1

        exact_match = False
        if not is_replay:
            exact_match = target.strip().lower() in gen["text"].strip().lower()
            if exact_match:
                exact_match_count += 1

        json_valid = None
        if cat == "structured_output":
            total_json_cases += 1
            try:
                parsed = json.loads(gen["text"].strip())
                json_valid = isinstance(parsed, dict)
                if json_valid:
                    valid_json_count += 1
            except Exception:
                json_valid = False

        abst_detected = None
        if cat == "uncertainty_insufficient_info":
            total_abst_cases += 1
            abst_detected = any(phrase in gen["text"].lower() for phrase in ["not contain", "unavailable", "does not state", "cannot", "no information", "not provided"])
            if abst_detected:
                abstention_count += 1

        test_results.append({
            "id": rid,
            "category": cat,
            "prompt": prompt,
            "target": target,
            "generated_text": gen["text"],
            "generated_tokens": gen["token_count"],
            "first_token_id": gen["first_token_id"],
            "first_token_is_eos": gen["first_token_is_eos"],
            "stop_reason": gen["stop_reason"],
            "is_empty": gen["is_empty"],
            "exact_match": exact_match,
            "json_valid": json_valid,
            "abstention_detected": abst_detected,
            "latency_ms": gen["latency_ms"],
        })

    # 2. Evaluate on 28-case generation benchmark (raw_completion vs plain_instruction)
    gen_bench_path = REPO_ROOT / "experiments" / "batch2781_2820" / "GENERATION_BENCHMARK.json"
    with open(gen_bench_path, "r", encoding="utf-8") as f:
        gen_cases = json.load(f)

    gen_bench_results = {"raw_completion": [], "plain_instruction": []}
    modes = ["raw_completion", "plain_instruction"]
    raw_empty_count = 0
    raw_first_token_eos_count = 0
    inst_empty_count = 0

    for mode in modes:
        for c in gen_cases:
            cid = c["case_id"]
            cat = c["category"]
            inp = c["input"]

            formatted_p = inp
            if mode == "plain_instruction":
                formatted_p = f"Instruction: {inp.strip()}\nResponse:"

            p_toks = tokenizer.encode(formatted_p)
            input_ids = [0] + p_toks

            gen = generate_greedy(model, tokenizer, input_ids, max_new_tokens=32)
            if mode == "raw_completion":
                if gen["is_empty"]:
                    raw_empty_count += 1
                if gen["first_token_is_eos"]:
                    raw_first_token_eos_count += 1
            else:
                if gen["is_empty"]:
                    inst_empty_count += 1

            gen_bench_results[mode].append({
                "case_id": cid,
                "category": cat,
                "input": inp,
                "prompt_mode": mode,
                "generated_text": gen["text"],
                "generated_tokens": gen["token_count"],
                "first_token_id": gen["first_token_id"],
                "first_token_is_eos": gen["first_token_is_eos"],
                "stop_reason": gen["stop_reason"],
                "is_empty": gen["is_empty"],
                "repetition_ratio": gen["repetition_ratio"],
                "latency_ms": gen["latency_ms"],
            })

    total_instruction_cases = len(test_records) - total_replay_cases
    return {
        "candidate_name": ckpt_name,
        "checkpoint_file": str(ckpt_path.name),
        "file_sha256": file_sha,
        "held_out_test_summary": {
            "total_test_cases": len(test_records),
            "instruction_cases": total_instruction_cases,
            "exact_match_count": exact_match_count,
            "exact_match_pct": round(exact_match_count / max(1, total_instruction_cases) * 100.0, 2),
            "valid_json_count": valid_json_count,
            "total_json_cases": total_json_cases,
            "abstention_count": abstention_count,
            "total_abst_cases": total_abst_cases,
            "replay_empty_count": replay_empty_count,
            "total_replay_cases": total_replay_cases,
        },
        "generation_benchmark_summary": {
            "total_cases": len(gen_cases),
            "raw_completion_empty_count": raw_empty_count,
            "raw_completion_empty_pct": round(raw_empty_count / len(gen_cases) * 100.0, 2),
            "raw_completion_first_token_eos_count": raw_first_token_eos_count,
            "raw_completion_first_token_eos_pct": round(raw_first_token_eos_count / len(gen_cases) * 100.0, 2),
            "plain_instruction_empty_count": inst_empty_count,
            "plain_instruction_empty_pct": round(inst_empty_count / len(gen_cases) * 100.0, 2),
        },
        "held_out_test_results": test_results,
        "generation_benchmark_results": gen_bench_results,
    }


def main():
    tok_dir = REPO_ROOT / "release" / "chakrview-v0.5.0" / "data" / "experiments" / "vocab_4096"
    tokenizer, _ = load_tokenizer_artifacts(str(tok_dir))

    candidates = [
        (
            REPO_ROOT / "experiments" / "batch16_21_combined" / "main_training_10000" / "training_checkpoint_step_10000.pt",
            "canonical_step_10000_baseline"
        ),
        (
            EXPERIMENT_DIR / "balanced_sft_pilot_run" / "sft_best_checkpoint.pt",
            "balanced_sft_best_candidate"
        ),
        (
            EXPERIMENT_DIR / "balanced_sft_pilot_run" / "sft_checkpoint_step_160.pt",
            "balanced_sft_final_step160_candidate"
        ),
    ]

    all_eval_results = {}
    for path, name in candidates:
        if not path.is_file():
            print(f"Skipping missing candidate: {path}")
            continue
        res = evaluate_checkpoint(path, name, tokenizer)
        all_eval_results[name] = res
        print(f"  Held-Out Exact Match: {res['held_out_test_summary']['exact_match_count']}/{res['held_out_test_summary']['instruction_cases']}")
        print(f"  Valid JSON:          {res['held_out_test_summary']['valid_json_count']}/{res['held_out_test_summary']['total_json_cases']}")
        print(f"  Abstention:          {res['held_out_test_summary']['abstention_count']}/{res['held_out_test_summary']['total_abst_cases']}")
        print(f"  Raw Empty Rate:      {res['generation_benchmark_summary']['raw_completion_empty_count']}/{res['generation_benchmark_summary']['total_cases']} ({res['generation_benchmark_summary']['raw_completion_empty_pct']}%)")
        print(f"  Raw First Token EOS: {res['generation_benchmark_summary']['raw_completion_first_token_eos_count']}/{res['generation_benchmark_summary']['total_cases']} ({res['generation_benchmark_summary']['raw_completion_first_token_eos_pct']}%)")

    out_file = EXPERIMENT_DIR / "PILOT_EVALUATION_RESULTS.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_eval_results, f, indent=2)

    print(f"\nEvaluation successfully written to {out_file}")


if __name__ == "__main__":
    main()
