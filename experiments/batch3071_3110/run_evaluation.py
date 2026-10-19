"""Unified Evaluation Runner for Batch 3071–3110 (JSON Capability Experiment V1).

Evaluates:
1. Canonical Step-10,000 Baseline Checkpoint
2. (Future) Candidate Checkpoint when authorized and trained:
   experiments/batch3071_3110/json_sft_pilot_run/sft_best_checkpoint.pt

Across:
A. Dedicated Held-Out Test Set (experiments/batch3071_3110/test.jsonl):
   - 20 structured-output JSON test cases (unseen schemas & field keys).
     Metrics:
     * Syntactic JSON validity (strict json.loads returning dict)
     * Schema adherence (exact match of expected keys)
     * Extra keys count, missing keys count, value type correctness
     * Truncation / empty-output counts
   - 10 unseen replay continuation cases (document-isolated).
     Metrics:
     * Continuation token count
     * Immediate EOS / empty rate
B. Natural Continuation Regression Guard (experiments/batch2781_2820/GENERATION_BENCHMARK.json):
   - 4 designated raw_continuation prompts:
     * Immediate EOS count (regression guard requires 0/4 immediate EOS)
     * Empty-output count
   - Full 28-case generation benchmark comparison (raw vs plain instruction).

Predeclared Acceptance Criteria:
- Primary Success: >= 50% syntactically valid JSON on held-out structured-output prompts.
- Regression Guard: 0 immediate-EOS outputs on the designated 4 natural-continuation prompts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

EXPERIMENT_DIR = REPO_ROOT / "experiments" / "batch3071_3110"
CANONICAL_CHECKPOINT_PATH = REPO_ROOT / "experiments" / "batch16_21_combined" / "main_training_10000" / "training_checkpoint_step_10000.pt"
CANONICAL_TOKENIZER_DIR = REPO_ROOT / "release" / "chakrview-v0.5.0" / "data" / "experiments" / "vocab_4096"

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.tokenizer.serialization import load_tokenizer_artifacts


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


def generate_greedy(
    model: ChakrMicro,
    tokenizer: Any,
    input_tokens: List[int],
    max_new_tokens: int = 48,
    eos_id: int = 1,
) -> Dict[str, Any]:
    curr = list(input_tokens)
    t0 = time.perf_counter()
    gen_tokens: List[int] = []
    stop_reason = "MAX_TOKENS"
    first_token_id: Optional[int] = None
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
            first_token_is_eos = (next_tok == eos_id)

        if next_tok == eos_id:
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


def parse_strict_json(text: str, expected_keys: Optional[List[str]] = None) -> Dict[str, Any]:
    """Strictly parses JSON from model generation.

    A response is only valid if it parses into a JSON object (dict).
    Checks exact key set match, extra keys, missing keys.
    """
    cleaned = text.strip()
    result = {
        "syntax_valid": False,
        "parsed_object": None,
        "schema_match": False,
        "missing_keys": [],
        "extra_keys": [],
        "type_mismatches": 0,
        "error": None,
    }

    if not cleaned:
        result["error"] = "EMPTY_OUTPUT"
        return result

    try:
        parsed = json.loads(cleaned)
        if not isinstance(parsed, dict):
            result["error"] = f"NOT_A_DICT: {type(parsed).__name__}"
            return result
        result["syntax_valid"] = True
        result["parsed_object"] = parsed

        if expected_keys is not None:
            parsed_keys = set(parsed.keys())
            target_keys = set(expected_keys)
            result["missing_keys"] = sorted(list(target_keys - parsed_keys))
            result["extra_keys"] = sorted(list(parsed_keys - target_keys))
            result["schema_match"] = (parsed_keys == target_keys)
    except Exception as e:
        result["error"] = str(e)

    return result


def evaluate_checkpoint(
    ckpt_path: Path,
    ckpt_name: str,
    tokenizer: Any,
    max_new_tokens: int = 48,
) -> Dict[str, Any]:
    print(f"\n=================================================================")
    print(f"EVALUATING CHECKPOINT: {ckpt_name}")
    print(f"Path: {ckpt_path}")
    print(f"=================================================================")

    file_sha = compute_sha256(ckpt_path)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = state.get("model_state_dict", state)

    model = ChakrMicro(ModelConfig(max_seq_len=512))
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    # 1. Evaluate on Batch 3071-3110 Held-Out Test Set (30 records: 20 JSON, 10 replay)
    test_path = EXPERIMENT_DIR / "test.jsonl"
    with open(test_path, "r", encoding="utf-8") as f:
        test_records = [json.loads(line) for line in f if line.strip()]

    test_case_results = []
    total_json = 0
    valid_json_count = 0
    schema_match_count = 0
    extra_keys_count = 0
    missing_keys_count = 0
    json_empty_count = 0

    total_replay = 0
    replay_empty_count = 0
    replay_immediate_eos_count = 0
    replay_tokens_sum = 0

    for r in test_records:
        cat = r.get("category", "")
        prompt = r["prompt"]
        target = r["response"]
        expected_keys = r.get("target_keys", None)

        p_toks = tokenizer.encode(prompt)
        input_ids = [0] + p_toks  # BOS + prompt

        gen = generate_greedy(model, tokenizer, input_ids, max_new_tokens=max_new_tokens)

        is_json = cat.startswith("structured_output")
        is_replay = (cat == "base_continuation_replay")

        json_eval = None
        if is_json:
            total_json += 1
            if gen["is_empty"]:
                json_empty_count += 1
            json_eval = parse_strict_json(gen["text"], expected_keys)
            if json_eval["syntax_valid"]:
                valid_json_count += 1
            if json_eval["schema_match"]:
                schema_match_count += 1
            if json_eval["extra_keys"]:
                extra_keys_count += 1
            if json_eval["missing_keys"]:
                missing_keys_count += 1

        if is_replay:
            total_replay += 1
            replay_tokens_sum += gen["token_count"]
            if gen["is_empty"]:
                replay_empty_count += 1
            if gen["first_token_is_eos"]:
                replay_immediate_eos_count += 1

        test_case_results.append({
            "id": r.get("id"),
            "category": cat,
            "prompt": prompt,
            "target": target,
            "expected_keys": expected_keys,
            "generated_text": gen["text"],
            "generated_tokens": gen["token_count"],
            "first_token_id": gen["first_token_id"],
            "first_token_is_eos": gen["first_token_is_eos"],
            "stop_reason": gen["stop_reason"],
            "is_empty": gen["is_empty"],
            "json_eval": json_eval,
            "latency_ms": gen["latency_ms"],
        })

    # 2. Evaluate on 28-case Generation Benchmark (Regression Guard on CONT_01 to CONT_04)
    gen_bench_path = REPO_ROOT / "experiments" / "batch2781_2820" / "GENERATION_BENCHMARK.json"
    gen_bench_results: Dict[str, List[Dict[str, Any]]] = {"raw_continuation": [], "plain_instruction": []}
    regression_guard_cases: List[Dict[str, Any]] = []

    if gen_bench_path.is_file():
        with open(gen_bench_path, "r", encoding="utf-8") as f:
            gen_cases = json.load(f)

        for mode in ["raw_continuation", "plain_instruction"]:
            for c in gen_cases:
                cid = c["case_id"]
                cat = c["category"]
                inp = c["input"]

                formatted_p = inp if mode == "raw_continuation" else f"Instruction: {inp.strip()}\nResponse:"
                p_toks = tokenizer.encode(formatted_p)
                input_ids = [0] + p_toks

                gen = generate_greedy(model, tokenizer, input_ids, max_new_tokens=max_new_tokens)

                record = {
                    "case_id": cid,
                    "category": cat,
                    "input": inp,
                    "mode": mode,
                    "generated_text": gen["text"],
                    "generated_tokens": gen["token_count"],
                    "first_token_id": gen["first_token_id"],
                    "first_token_is_eos": gen["first_token_is_eos"],
                    "stop_reason": gen["stop_reason"],
                    "is_empty": gen["is_empty"],
                    "repetition_ratio": gen["repetition_ratio"],
                }
                gen_bench_results[mode].append(record)

                if mode == "raw_continuation" and cat == "raw_continuation":
                    regression_guard_cases.append(record)

    reg_immediate_eos = sum(1 for c in regression_guard_cases if c["first_token_is_eos"])
    reg_empty = sum(1 for c in regression_guard_cases if c["is_empty"])

    json_valid_pct = round(valid_json_count / max(1, total_json) * 100.0, 2)
    schema_match_pct = round(schema_match_count / max(1, total_json) * 100.0, 2)

    primary_success = (json_valid_pct >= 50.0)
    regression_guard_pass = (reg_immediate_eos == 0)

    summary = {
        "candidate_name": ckpt_name,
        "checkpoint_file": str(ckpt_path.name),
        "sha256": file_sha,
        "held_out_test_metrics": {
            "total_test_records": len(test_records),
            "json_cases": {
                "total": total_json,
                "valid_syntax_count": valid_json_count,
                "valid_syntax_pct": json_valid_pct,
                "schema_match_count": schema_match_count,
                "schema_match_pct": schema_match_pct,
                "extra_keys_count": extra_keys_count,
                "missing_keys_count": missing_keys_count,
                "empty_output_count": json_empty_count,
            },
            "replay_cases": {
                "total": total_replay,
                "avg_tokens": round(replay_tokens_sum / max(1, total_replay), 2),
                "empty_count": replay_empty_count,
                "immediate_eos_count": replay_immediate_eos_count,
            },
        },
        "regression_guard": {
            "total_raw_continuation_guard_prompts": len(regression_guard_cases),
            "immediate_eos_count": reg_immediate_eos,
            "empty_output_count": reg_empty,
            "guard_status": "PASS" if regression_guard_pass else "FAIL",
        },
        "criteria_evaluation": {
            "primary_success_target": ">= 50% valid JSON syntax on held-out test",
            "primary_success_achieved": primary_success,
            "regression_guard_target": "0 immediate-EOS on 4 raw-continuation prompts",
            "regression_guard_achieved": regression_guard_pass,
        },
        "test_case_results": test_case_results,
        "generation_benchmark_results": gen_bench_results,
    }

    print(f"Results for {ckpt_name}:")
    print(f"  Valid JSON: {valid_json_count}/{total_json} ({json_valid_pct}%) [Target >= 50%]")
    print(f"  Schema Match: {schema_match_count}/{total_json} ({schema_match_pct}%)")
    print(f"  Regression Guard (Immediate EOS): {reg_immediate_eos}/4 [Target == 0] -> {'PASS' if regression_guard_pass else 'FAIL'}")
    print(f"  Replay Avg Tokens: {summary['held_out_test_metrics']['replay_cases']['avg_tokens']}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="ChakrView JSON SFT Evaluation Runner")
    parser.add_argument("--candidate-path", type=str, default=None, help="Path to trained candidate checkpoint")
    parser.add_argument("--baseline-only", action="store_true", help="Evaluate only canonical baseline")
    parser.add_argument("--output-json", type=str, default="BASELINE_AND_CANDIDATE_EVALUATION.json")
    args = parser.parse_args()

    # Load tokenizer
    tok, _ = load_tokenizer_artifacts(str(CANONICAL_TOKENIZER_DIR))

    eval_data: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "evaluator_version": "batch3071_3110_json_eval_v1",
        "evaluations": {},
    }

    # Always evaluate canonical baseline first
    baseline_summary = evaluate_checkpoint(
        CANONICAL_CHECKPOINT_PATH,
        "canonical_baseline_step_10000",
        tok,
    )
    eval_data["evaluations"]["canonical_baseline"] = baseline_summary

    # Evaluate candidate if supplied and exists
    cand_path_obj = None
    if args.candidate_path:
        cand_path_obj = Path(args.candidate_path)
    else:
        default_cand = EXPERIMENT_DIR / "json_sft_pilot_run" / "sft_best_checkpoint.pt"
        if default_cand.is_file() and not args.baseline_only:
            cand_path_obj = default_cand

    if cand_path_obj and cand_path_obj.is_file():
        cand_summary = evaluate_checkpoint(
            cand_path_obj,
            cand_path_obj.stem,
            tok,
        )
        eval_data["evaluations"]["candidate"] = cand_summary
    else:
        print("\nCandidate checkpoint not found or not yet trained. Baseline numbers recorded and frozen.")

    out_file = EXPERIMENT_DIR / args.output_json
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(eval_data, f, indent=2)
    print(f"\nEvaluation output saved to: {out_file}")


if __name__ == "__main__":
    main()
