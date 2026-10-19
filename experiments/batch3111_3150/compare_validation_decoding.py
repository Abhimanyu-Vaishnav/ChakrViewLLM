"""Validation-only comparative harness: unconstrained greedy vs experimental constrained decoding."""

import json
import torch
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

TOK_DIR = REPO_ROOT / "release" / "chakrview-v0.5.0" / "data" / "experiments" / "vocab_4096"
CKPT_PATH = REPO_ROOT / "experiments" / "batch3111_3150" / "json_sft_v2_pilot_run" / "sft_best_checkpoint.pt"
VAL_PATH = REPO_ROOT / "experiments" / "batch3111_3150" / "val.jsonl"

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.tokenizer.serialization import load_tokenizer_artifacts
from experiments.batch3111_3150.json_constrained_decoder import generate_json_constrained
from experiments.batch3111_3150.run_evaluation import generate_greedy, parse_strict_json


def run_validation_comparison(max_records: int = 20) -> Dict[str, Any]:
    tok, _ = load_tokenizer_artifacts(str(TOK_DIR))
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    model = ChakrMicro(ModelConfig(max_seq_len=512))
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    with open(VAL_PATH, "r", encoding="utf-8") as f:
        val_records = [json.loads(line) for line in f if line.strip()]

    # Filter strictly to structured JSON records in validation
    structured_val = [r for r in val_records if r["category"].startswith("structured_output")][:max_records]

    comparison_results = []
    unconstrained_valid_count = 0
    constrained_valid_count = 0
    unconstrained_schema_count = 0
    constrained_schema_count = 0
    constrained_complete_count = 0

    unconstrained_tokens_total = 0
    constrained_tokens_total = 0

    for idx, r in enumerate(structured_val):
        rid = r["id"]
        prompt = r["prompt"]
        target = r["response"]
        expected_keys = r.get("target_keys")
        inp = [0] + tok.encode(prompt)

        # 1. Unconstrained Greedy Decoding (baseline)
        res_u = generate_greedy(model, tok, inp, max_new_tokens=48)
        eval_u = parse_strict_json(res_u["text"], expected_keys)
        if eval_u["syntax_valid"]:
            unconstrained_valid_count += 1
        if eval_u["schema_match"]:
            unconstrained_schema_count += 1
        unconstrained_tokens_total += res_u["token_count"]

        # 2. Experimental Constrained Decoding
        res_c = generate_json_constrained(model, tok, inp, max_new_tokens=48)
        eval_c = parse_strict_json(res_c["text"], expected_keys)
        if eval_c["syntax_valid"]:
            constrained_valid_count += 1
        if eval_c["schema_match"]:
            constrained_schema_count += 1
        if res_c["stop_reason"] == "COMPLETE_JSON":
            constrained_complete_count += 1
        constrained_tokens_total += res_c["token_count"]

        comparison_results.append({
            "id": rid,
            "category": r["category"],
            "prompt": prompt,
            "target": target,
            "expected_keys": expected_keys,
            "unconstrained": {
                "text": res_u["text"],
                "token_count": res_u["token_count"],
                "syntax_valid": eval_u["syntax_valid"],
                "schema_match": eval_u["schema_match"],
                "error": eval_u["error"],
            },
            "constrained": {
                "text": res_c["text"],
                "token_count": res_c["token_count"],
                "stop_reason": res_c["stop_reason"],
                "syntax_valid": eval_c["syntax_valid"],
                "schema_match": eval_c["schema_match"],
                "error": eval_c["error"],
            },
        })

    total = len(structured_val)
    summary = {
        "total_records_evaluated": total,
        "unconstrained_metrics": {
            "valid_syntax_count": unconstrained_valid_count,
            "valid_syntax_pct": round((unconstrained_valid_count / max(1, total)) * 100, 2),
            "schema_match_count": unconstrained_schema_count,
            "schema_match_pct": round((unconstrained_schema_count / max(1, total)) * 100, 2),
            "avg_tokens": round(unconstrained_tokens_total / max(1, total), 2),
        },
        "constrained_metrics": {
            "valid_syntax_count": constrained_valid_count,
            "valid_syntax_pct": round((constrained_valid_count / max(1, total)) * 100, 2),
            "completed_objects": constrained_complete_count,
            "schema_match_count": constrained_schema_count,
            "schema_match_pct": round((constrained_schema_count / max(1, total)) * 100, 2),
            "avg_tokens": round(constrained_tokens_total / max(1, total), 2),
        },
        "per_record_comparison": comparison_results,
    }

    return summary


def main():
    print("=================================================================")
    print("CHAKRVIEW BATCH 3111-3150: VALIDATION-ONLY COMPARATIVE HARNESS")
    print("=================================================================")
    summary = run_validation_comparison(max_records=20)
    print(f"Total Records Evaluated: {summary['total_records_evaluated']}")
    print(f"Unconstrained Valid Syntax: {summary['unconstrained_metrics']['valid_syntax_count']}/{summary['total_records_evaluated']} ({summary['unconstrained_metrics']['valid_syntax_pct']}%)")
    print(f"Constrained Valid Syntax:   {summary['constrained_metrics']['valid_syntax_count']}/{summary['total_records_evaluated']} ({summary['constrained_metrics']['valid_syntax_pct']}%)")
    print(f"Constrained Schema Matches: {summary['constrained_metrics']['schema_match_count']}/{summary['total_records_evaluated']} ({summary['constrained_metrics']['schema_match_pct']}%)")
    print(f"Constrained Completed Objs: {summary['constrained_metrics']['completed_objects']}/{summary['total_records_evaluated']}")
    print("=================================================================")

    out_file = REPO_ROOT / "experiments" / "batch3111_3150" / "VALIDATION_COMPARISON_RESULTS.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Detailed comparison saved to: {out_file}")


if __name__ == "__main__":
    main()
