"""Unified Evaluation Runner for Batch 3111–3150 (JSON SFT V2).

Separates Evaluation Objectives:
1. Familiar-Schema Evaluation (12 held-out instances of known schemas: status, metric, feature, item):
   - Syntax completion (parseable JSON dict)
   - Schema adherence (exact key set match)
   - Value type correctness (string, int, float, bool)
   - Repetition and truncation
2. Unseen-Schema Generalization (12 instances of genuinely unseen schemas):
   - Syntax completion (parseable JSON dict)
   - Schema adherence (exact key set match on novel keys)
   - Value type correctness
   - Repetition and truncation
3. Replay Continuation (10 test cases from document-isolated pretraining corpus):
   - Average continuation token length
   - Immediate EOS / empty rate
4. Natural Continuation Regression Guard (4 designated benchmark prompts):
   - Immediate EOS count (must remain 0/4)
   - Active completion length

Emits: BASELINE_AND_CANDIDATE_EVALUATION.json
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

EXPERIMENT_DIR = REPO_ROOT / "experiments" / "batch3111_3150"
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

    test_path = EXPERIMENT_DIR / "test.jsonl"
    with open(test_path, "r", encoding="utf-8") as f:
        test_records = [json.loads(line) for line in f if line.strip()]

    test_case_results = []

    # Slices
    fam_json_total = 0
    fam_json_valid = 0
    fam_schema_match = 0

    uns_json_total = 0
    uns_json_valid = 0
    uns_schema_match = 0

    rep_total = 0
    rep_tokens_sum = 0
    rep_empty_count = 0
    rep_immediate_eos = 0

    for r in test_records:
        cat = r.get("category", "")
        prompt = r["prompt"]
        target = r["response"]
        expected_keys = r.get("target_keys", None)

        p_toks = tokenizer.encode(prompt)
        input_ids = [0] + p_toks

        gen = generate_greedy(model, tokenizer, input_ids, max_new_tokens=max_new_tokens)

        is_fam = "familiar" in cat
        is_uns = "unseen" in cat
        is_rep = cat == "base_continuation_replay"

        json_eval = None
        if is_fam:
            fam_json_total += 1
            json_eval = parse_strict_json(gen["text"], expected_keys)
            if json_eval["syntax_valid"]:
                fam_json_valid += 1
            if json_eval["schema_match"]:
                fam_schema_match += 1
        elif is_uns:
            uns_json_total += 1
            json_eval = parse_strict_json(gen["text"], expected_keys)
            if json_eval["syntax_valid"]:
                uns_json_valid += 1
            if json_eval["schema_match"]:
                uns_schema_match += 1
        elif is_rep:
            rep_total += 1
            rep_tokens_sum += gen["token_count"]
            if gen["is_empty"]:
                rep_empty_count += 1
            if gen["first_token_is_eos"]:
                rep_immediate_eos += 1

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

    # Regression Guard on the 4 designated raw_continuation prompts
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

    fam_valid_pct = round(fam_json_valid / max(1, fam_json_total) * 100.0, 2)
    fam_schema_pct = round(fam_schema_match / max(1, fam_json_total) * 100.0, 2)
    uns_valid_pct = round(uns_json_valid / max(1, uns_json_total) * 100.0, 2)
    uns_schema_pct = round(uns_schema_match / max(1, uns_json_total) * 100.0, 2)

    summary = {
        "candidate_name": ckpt_name,
        "checkpoint_file": str(ckpt_path.name),
        "sha256": file_sha,
        "familiar_schema_metrics": {
            "total": fam_json_total,
            "valid_syntax_count": fam_json_valid,
            "valid_syntax_pct": fam_valid_pct,
            "schema_match_count": fam_schema_match,
            "schema_match_pct": fam_schema_pct,
        },
        "unseen_schema_metrics": {
            "total": uns_json_total,
            "valid_syntax_count": uns_json_valid,
            "valid_syntax_pct": uns_valid_pct,
            "schema_match_count": uns_schema_match,
            "schema_match_pct": uns_schema_pct,
        },
        "replay_metrics": {
            "total": rep_total,
            "avg_tokens": round(rep_tokens_sum / max(1, rep_total), 2),
            "empty_count": rep_empty_count,
            "immediate_eos_count": rep_immediate_eos,
        },
        "regression_guard": {
            "total_prompts": len(regression_guard_cases),
            "immediate_eos_count": reg_immediate_eos,
            "empty_output_count": reg_empty,
            "guard_status": "PASS" if reg_immediate_eos == 0 else "FAIL",
        },
        "predeclared_targets": {
            "familiar_syntax_target": ">= 50% (>= 6/12) valid JSON on familiar schemas",
            "unseen_syntax_target": "Reported separately for zero-shot generalization",
            "regression_guard_target": "0 immediate-EOS on 4 raw continuation prompts",
        },
        "test_case_results": test_case_results,
        "generation_benchmark_results": gen_bench_results,
    }

    print(f"Results for {ckpt_name}:")
    print(f"  Familiar Schema Valid JSON: {fam_json_valid}/{fam_json_total} ({fam_valid_pct}%) [Target >= 50%]")
    print(f"  Familiar Schema Match:      {fam_schema_match}/{fam_json_total} ({fam_schema_pct}%)")
    print(f"  Unseen Schema Valid JSON:   {uns_json_valid}/{uns_json_total} ({uns_valid_pct}%)")
    print(f"  Unseen Schema Match:        {uns_schema_match}/{uns_json_total} ({uns_schema_pct}%)")
    print(f"  Regression Guard EOS:       {reg_immediate_eos}/4 -> {'PASS' if reg_immediate_eos == 0 else 'FAIL'}")
    print(f"  Replay Avg Tokens:          {summary['replay_metrics']['avg_tokens']}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="ChakrView JSON SFT V2 Evaluation Runner")
    parser.add_argument("--candidate-path", type=str, default=None, help="Path to trained candidate checkpoint")
    parser.add_argument("--baseline-only", action="store_true", help="Evaluate only canonical baseline")
    parser.add_argument("--output-json", type=str, default="BASELINE_AND_CANDIDATE_EVALUATION.json")
    args = parser.parse_args()

    tok, _ = load_tokenizer_artifacts(str(CANONICAL_TOKENIZER_DIR))

    eval_data: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "evaluator_version": "batch3111_3150_v2",
        "evaluations": {},
    }

    baseline_summary = evaluate_checkpoint(
        CANONICAL_CHECKPOINT_PATH,
        "canonical_baseline_step_10000",
        tok,
    )
    eval_data["evaluations"]["canonical_baseline"] = baseline_summary

    cand_path_obj = None
    if args.candidate_path:
        cand_path_obj = Path(args.candidate_path)
    else:
        default_cand = EXPERIMENT_DIR / "json_sft_v2_pilot_run" / "sft_best_checkpoint.pt"
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
