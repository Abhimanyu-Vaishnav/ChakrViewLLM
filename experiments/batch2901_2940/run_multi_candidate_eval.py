"""Comprehensive evaluation runner for Batch 2901–2940.

Evaluates all candidate checkpoints:
1. Canonical Step-10,000 baseline
2. Step-25 candidate
3. Step-50 candidate (closest saved checkpoint to best validation loss at step 40)
4. Step-75 candidate
5. Step-100 candidate

Across:
A. The 24-case held-out SFT test set (with exact match, structured JSON parsing, abstention detection).
B. The 28-case generation benchmark under both raw_completion and plain_instruction modes (measuring empty outputs, EOS triggers, repetition ratio, latency).

Outputs: experiments/batch2901_2940/CANDIDATE_EVALUATION_RESULTS.json
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

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.tokenizer.serialization import load_tokenizer_artifacts

logger = logging.getLogger("batch2901_eval")


def compute_bigram_repetition(tokens: List[int]) -> float:
    if len(tokens) < 3:
        return 0.0
    bigrams = [tuple(tokens[i : i + 2]) for i in range(len(tokens) - 1)]
    unique_bigrams = set(bigrams)
    return 1.0 - (len(unique_bigrams) / len(bigrams))


def generate_greedy(model: ChakrMicro, tokenizer: Any, input_tokens: List[int], max_new_tokens: int = 32) -> Dict[str, Any]:
    curr = list(input_tokens)
    t0 = time.perf_counter()
    gen_tokens = []
    stop_reason = "MAX_TOKENS"

    for _ in range(max_new_tokens):
        if len(curr) >= 512:
            stop_reason = "CONTEXT_LIMIT"
            break
        x = torch.tensor([curr], dtype=torch.long)
        with torch.no_grad():
            logits = model(x)
        next_tok = int(torch.argmax(logits[0, -1, :]).item())
        if next_tok == 1:  # EOS_ID
            stop_reason = "STOP_TOKEN"
            break
        gen_tokens.append(next_tok)
        curr.append(next_tok)

    latency_ms = (time.perf_counter() - t0) * 1000.0
    out_text = tokenizer.decode(gen_tokens)
    return {
        "generated_tokens": gen_tokens,
        "token_count": len(gen_tokens),
        "text": out_text,
        "latency_ms": round(latency_ms, 2),
        "stop_reason": stop_reason,
        "is_empty": len(gen_tokens) == 0,
        "repetition_ratio": round(compute_bigram_repetition(gen_tokens), 4),
    }


def evaluate_model_on_suite(model: ChakrMicro, tokenizer: Any, ckpt_name: str, file_sha: str, param_digest: str) -> Dict[str, Any]:
    # 1. Evaluate on 24-case SFT test set
    sft_test_path = REPO_ROOT / "experiments" / "batch2821_2860" / "SFT_EVALUATION_SET.json"
    with open(sft_test_path, "r", encoding="utf-8") as f:
        sft_cases = json.load(f)["cases"]

    sft_results = []
    exact_match_count = 0
    valid_json_count = 0
    total_json_cases = 0
    abstention_count = 0
    total_abst_cases = 0

    for c in sft_cases:
        cid = c["id"]
        cat = c["category"]
        prompt = c["prompt"]
        target = c["response"]

        p_toks = tokenizer.encode(prompt)
        input_ids = [0] + p_toks  # BOS + prompt

        gen = generate_greedy(model, tokenizer, input_ids, max_new_tokens=32)

        # Metrics
        exact_match = target.strip().lower() in gen["text"].strip().lower()
        if exact_match:
            exact_match_count += 1

        json_valid = None
        if cat == "structured_output":
            total_json_cases += 1
            try:
                json.loads(gen["text"].strip())
                json_valid = True
                valid_json_count += 1
            except Exception:
                json_valid = False

        abst_detected = None
        if cat == "uncertainty_insufficient_info":
            total_abst_cases += 1
            abst_detected = any(phrase in gen["text"].lower() for phrase in ["not contain", "unavailable", "does not state", "cannot", "no information"])
            if abst_detected:
                abstention_count += 1

        sft_results.append({
            "id": cid,
            "category": cat,
            "prompt": prompt,
            "target": target,
            "generated_text": gen["text"],
            "generated_tokens": gen["token_count"],
            "stop_reason": gen["stop_reason"],
            "is_empty": gen["is_empty"],
            "exact_match": exact_match,
            "json_valid": json_valid,
            "abstention_detected": abst_detected,
            "latency_ms": gen["latency_ms"],
        })

    # 2. Evaluate on 28-case generation benchmark under both raw_completion and plain_instruction
    gen_bench_path = REPO_ROOT / "experiments" / "batch2781_2820" / "GENERATION_BENCHMARK.json"
    with open(gen_bench_path, "r", encoding="utf-8") as f:
        gen_cases = json.load(f)

    gen_bench_results = {"raw_completion": [], "plain_instruction": []}
    modes = ["raw_completion", "plain_instruction"]

    empty_counts = {"raw_completion": 0, "plain_instruction": 0}

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
            if gen["is_empty"]:
                empty_counts[mode] += 1

            gen_bench_results[mode].append({
                "case_id": cid,
                "category": cat,
                "input": inp,
                "prompt_mode": mode,
                "generated_text": gen["text"],
                "generated_tokens": gen["token_count"],
                "stop_reason": gen["stop_reason"],
                "is_empty": gen["is_empty"],
                "repetition_ratio": gen["repetition_ratio"],
                "latency_ms": gen["latency_ms"],
            })

    return {
        "candidate_name": ckpt_name,
        "file_sha256": file_sha,
        "parameter_digest": param_digest,
        "sft_test_summary": {
            "total_cases": len(sft_cases),
            "exact_match_count": exact_match_count,
            "exact_match_pct": round((exact_match_count / len(sft_cases)) * 100, 2),
            "valid_json_count": valid_json_count,
            "total_json_cases": total_json_cases,
            "abstention_count": abstention_count,
            "total_abst_cases": total_abst_cases,
        },
        "generation_benchmark_summary": {
            "total_cases": len(gen_cases),
            "raw_completion_empty_count": empty_counts["raw_completion"],
            "raw_completion_empty_pct": round((empty_counts["raw_completion"] / len(gen_cases)) * 100, 2),
            "plain_instruction_empty_count": empty_counts["plain_instruction"],
            "plain_instruction_empty_pct": round((empty_counts["plain_instruction"] / len(gen_cases)) * 100, 2),
        },
        "sft_test_results": sft_results,
        "generation_benchmark_results": gen_bench_results,
    }


def main():
    tok_dir = REPO_ROOT / "release" / "chakrview-v0.5.0" / "data" / "experiments" / "vocab_4096"
    tokenizer, _ = load_tokenizer_artifacts(str(tok_dir))

    checkpoints_to_eval = [
        ("canonical_step_10000", REPO_ROOT / "experiments" / "batch16_21_combined" / "main_training_10000" / "training_checkpoint_step_10000.pt"),
        ("sft_step_25", REPO_ROOT / "experiments" / "batch2821_2860" / "sft_pilot_run" / "sft_checkpoint_step_25.pt"),
        ("sft_step_50", REPO_ROOT / "experiments" / "batch2821_2860" / "sft_pilot_run" / "sft_checkpoint_step_50.pt"),
        ("sft_step_75", REPO_ROOT / "experiments" / "batch2821_2860" / "sft_pilot_run" / "sft_checkpoint_step_75.pt"),
        ("sft_step_100", REPO_ROOT / "experiments" / "batch2821_2860" / "sft_pilot_run" / "sft_checkpoint_step_100.pt"),
    ]

    all_evaluations = {}

    for name, path in checkpoints_to_eval:
        print(f"\n==================================================")
        print(f"Evaluating candidate: {name} ({path.name})")
        print(f"==================================================")

        file_sha = hashlib.sha256(path.read_bytes()).hexdigest()

        model = ChakrMicro(ModelConfig(max_seq_len=512)).to("cpu")
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        model.load_state_dict(sd, strict=True)
        model.eval()

        hasher = hashlib.sha256()
        for _, p in sorted(model.named_parameters()):
            hasher.update(p.detach().cpu().numpy().tobytes())
        param_digest = hasher.hexdigest()

        eval_summary = evaluate_model_on_suite(model, tokenizer, name, file_sha, param_digest)
        all_evaluations[name] = eval_summary

        sft_sum = eval_summary["sft_test_summary"]
        gen_sum = eval_summary["generation_benchmark_summary"]
        print(f"[{name}] SFT Exact Match: {sft_sum['exact_match_count']}/{sft_sum['total_cases']}, JSON Valid: {sft_sum['valid_json_count']}/{sft_sum['total_json_cases']}")
        print(f"[{name}] Raw Empty: {gen_sum['raw_completion_empty_count']}/{gen_sum['total_cases']}, Plain Empty: {gen_sum['plain_instruction_empty_count']}/{gen_sum['total_cases']}")

    output_path = REPO_ROOT / "experiments" / "batch2901_2940" / "CANDIDATE_EVALUATION_RESULTS.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_evaluations, f, indent=2, ensure_ascii=False)
    print(f"\nSaved full multi-candidate results to {output_path}")


if __name__ == "__main__":
    main()
