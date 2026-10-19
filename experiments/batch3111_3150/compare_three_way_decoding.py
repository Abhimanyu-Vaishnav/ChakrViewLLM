"""Three-Way Validation Comparison Harness:
1. Unconstrained Greedy Decoding
2. Grammar-Only Constrained Decoding
3. Schema-Constrained Trie Decoding

Evaluates the 20 structured output records from val.jsonl using identical candidate checkpoint and budget.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch

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
from experiments.batch3111_3150.schema_prefix_trie import (
    JSONSchema,
    JSONType,
    generate_schema_constrained,
)
from experiments.batch3111_3150.run_evaluation import generate_greedy, parse_strict_json


def determine_schema_for_record(record: Dict[str, Any]) -> JSONSchema:
    """Builds an explicit schema specification based on category and target keys."""
    cat = record.get("category", "")
    target_keys = record.get("target_keys", [])
    
    # Map category to allowed types
    type_map = {}
    if "server_status" in cat:
        type_map = {"status": JSONType.STRING, "code": JSONType.INTEGER}
    elif "network_metric" in cat:
        type_map = {"metric": JSONType.STRING, "value": JSONType.NUMBER}
    elif "feature_flag" in cat:
        type_map = {"feature": JSONType.STRING, "enabled": JSONType.BOOLEAN}
    elif "inventory_item" in cat:
        type_map = {"name": JSONType.STRING, "count": JSONType.INTEGER}
    else:
        type_map = {k: JSONType.ANY for k in target_keys}

    return JSONSchema.from_keys(
        required_keys=target_keys,
        type_map=type_map,
        allow_additional_properties=False,
    )


def run_three_way_comparison(max_records: int = 20) -> Dict[str, Any]:
    tok, _ = load_tokenizer_artifacts(str(TOK_DIR))
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    model = ChakrMicro(ModelConfig(max_seq_len=512))
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    with open(VAL_PATH, "r", encoding="utf-8") as f:
        val_records = [json.loads(line) for line in f if line.strip()]

    structured_val = [r for r in val_records if r["category"].startswith("structured_output")][:max_records]

    results = []

    # Aggregators
    modes = ["unconstrained", "grammar_only", "schema_trie"]
    counts = {
        m: {
            "valid_syntax": 0,
            "completed_objects": 0,
            "schema_match": 0,
            "value_exact_match": 0,
            "budget_exhausted": 0,
            "deadlock_or_no_valid": 0,
            "total_tokens": 0,
        }
        for m in modes
    }

    for idx, r in enumerate(structured_val):
        rid = r["id"]
        cat = r["category"]
        prompt = r["prompt"]
        target = r["response"]
        expected_keys = r.get("target_keys", [])
        expected_values = r.get("target_values", [])
        schema = determine_schema_for_record(r)
        inp = [0] + tok.encode(prompt)

        # 1. Unconstrained Greedy
        res_u = generate_greedy(model, tok, inp, max_new_tokens=48)
        eval_u = parse_strict_json(res_u["text"], expected_keys)
        counts["unconstrained"]["total_tokens"] += res_u["token_count"]
        if eval_u["syntax_valid"]: counts["unconstrained"]["valid_syntax"] += 1
        if eval_u["schema_match"]: counts["unconstrained"]["schema_match"] += 1
        if res_u["token_count"] >= 48: counts["unconstrained"]["budget_exhausted"] += 1

        # 2. Grammar-Only Constrained
        res_g = generate_json_constrained(model, tok, inp, max_new_tokens=48)
        eval_g = parse_strict_json(res_g["text"], expected_keys)
        counts["grammar_only"]["total_tokens"] += res_g["token_count"]
        if eval_g["syntax_valid"]: counts["grammar_only"]["valid_syntax"] += 1
        if eval_g["schema_match"]: counts["grammar_only"]["schema_match"] += 1
        if res_g["stop_reason"] == "COMPLETE_JSON": counts["grammar_only"]["completed_objects"] += 1
        if res_g["stop_reason"] == "MAX_TOKENS": counts["grammar_only"]["budget_exhausted"] += 1
        if res_g["stop_reason"] in ("GRAMMAR_DEADLOCK", "NO_VALID_TOKENS"): counts["grammar_only"]["deadlock_or_no_valid"] += 1

        # 3. Schema-Trie Constrained
        res_s = generate_schema_constrained(model, tok, inp, schema, max_new_tokens=48)
        eval_s = parse_strict_json(res_s["text"], expected_keys)
        counts["schema_trie"]["total_tokens"] += res_s["token_count"]
        if eval_s["syntax_valid"]: counts["schema_trie"]["valid_syntax"] += 1
        if eval_s["schema_match"]: counts["schema_trie"]["schema_match"] += 1
        if res_s["stop_reason"] == "COMPLETE_JSON": counts["schema_trie"]["completed_objects"] += 1
        if res_s["stop_reason"] == "MAX_TOKENS": counts["schema_trie"]["budget_exhausted"] += 1
        if res_s["stop_reason"] in ("NO_VALID_TOKENS", "GRAMMAR_DEADLOCK"): counts["schema_trie"]["deadlock_or_no_valid"] += 1

        # Value exact match check (if completed valid json)
        value_match_s = False
        if res_s["is_valid_json"] and isinstance(res_s["parsed_object"], dict):
            obj = res_s["parsed_object"]
            # Check if values match target values
            try:
                target_obj = json.loads(target)
                if obj == target_obj:
                    value_match_s = True
                    counts["schema_trie"]["value_exact_match"] += 1
            except Exception:
                pass

        results.append({
            "id": rid,
            "category": cat,
            "prompt": prompt,
            "target": target,
            "expected_keys": expected_keys,
            "unconstrained": {
                "text": res_u["text"],
                "token_count": res_u["token_count"],
                "syntax_valid": eval_u["syntax_valid"],
                "schema_match": eval_u["schema_match"],
            },
            "grammar_only": {
                "text": res_g["text"],
                "token_count": res_g["token_count"],
                "stop_reason": res_g["stop_reason"],
                "syntax_valid": eval_g["syntax_valid"],
                "schema_match": eval_g["schema_match"],
            },
            "schema_trie": {
                "text": res_s["text"],
                "token_count": res_s["token_count"],
                "stop_reason": res_s["stop_reason"],
                "syntax_valid": eval_s["syntax_valid"],
                "schema_match": eval_s["schema_match"],
                "emitted_keys": res_s["emitted_keys"],
                "missing_keys": res_s["missing_keys"],
                "value_match": value_match_s,
                "parsed_object": res_s["parsed_object"],
            },
        })

    total = len(structured_val)
    summary = {
        "total_records_evaluated": total,
        "metrics": {
            m: {
                "valid_syntax_count": counts[m]["valid_syntax"],
                "valid_syntax_pct": round((counts[m]["valid_syntax"] / max(1, total)) * 100, 2),
                "completed_objects": counts[m]["completed_objects"],
                "completed_objects_pct": round((counts[m]["completed_objects"] / max(1, total)) * 100, 2),
                "schema_match_count": counts[m]["schema_match"],
                "schema_match_pct": round((counts[m]["schema_match"] / max(1, total)) * 100, 2),
                "value_exact_match_count": counts[m]["value_exact_match"],
                "value_exact_match_pct": round((counts[m]["value_exact_match"] / max(1, total)) * 100, 2),
                "budget_exhausted_count": counts[m]["budget_exhausted"],
                "deadlock_count": counts[m]["deadlock_or_no_valid"],
                "avg_tokens": round(counts[m]["total_tokens"] / max(1, total), 2),
            }
            for m in modes
        },
        "per_record_comparison": results,
    }
    return summary


def main():
    print("=================================================================")
    print("CHAKRVIEW BATCH 3111-3150: THREE-WAY VALIDATION COMPARISON")
    print("=================================================================")
    summary = run_three_way_comparison(max_records=20)
    total = summary["total_records_evaluated"]
    print(f"Total Records Evaluated: {total}")
    for m in ["unconstrained", "grammar_only", "schema_trie"]:
        met = summary["metrics"][m]
        print(f"\n[{m.upper()}]")
        print(f"  Valid Syntax:       {met['valid_syntax_count']}/{total} ({met['valid_syntax_pct']}%)")
        print(f"  Completed Objects:  {met['completed_objects']}/{total} ({met['completed_objects_pct']}%)")
        print(f"  Schema Match:       {met['schema_match_count']}/{total} ({met['schema_match_pct']}%)")
        print(f"  Value Exact Match:  {met['value_exact_match_count']}/{total} ({met['value_exact_match_pct']}%)")
        print(f"  Budget Exhausted:   {met['budget_exhausted_count']}/{total}")
        print(f"  Deadlock / No Valid:{met['deadlock_count']}/{total}")
        print(f"  Avg Tokens:         {met['avg_tokens']}")

    out_file = REPO_ROOT / "experiments" / "batch3111_3150" / "SCHEMA_TRIE_VALIDATION_COMPARISON.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved detailed 3-way comparison to: {out_file}")


if __name__ == "__main__":
    main()
