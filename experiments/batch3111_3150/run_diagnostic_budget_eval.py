"""Diagnostic evaluation of Schema-Constrained Trie Decoding with larger token budget (96 tokens)."""

import json
import sys
from pathlib import Path
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
from experiments.batch3111_3150.schema_prefix_trie import (
    JSONSchema,
    JSONType,
    generate_schema_constrained,
)
from experiments.batch3111_3150.compare_three_way_decoding import determine_schema_for_record
from experiments.batch3111_3150.run_evaluation import parse_strict_json


def run_diagnostic_budget_eval(max_new_tokens: int = 96):
    tok, _ = load_tokenizer_artifacts(str(TOK_DIR))
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    model = ChakrMicro(ModelConfig(max_seq_len=512))
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    with open(VAL_PATH, "r", encoding="utf-8") as f:
        val_records = [json.loads(line) for line in f if line.strip()]

    structured_val = [r for r in val_records if r["category"].startswith("structured_output")][:20]

    valid_syntax_count = 0
    completed_objects = 0
    schema_match_count = 0
    budget_exhausted_count = 0
    deadlock_count = 0
    total_tokens = 0
    token_counts = []

    per_record = []
    for r in structured_val:
        rid = r["id"]
        cat = r["category"]
        prompt = r["prompt"]
        expected_keys = r.get("target_keys", [])
        schema = determine_schema_for_record(r)
        inp = [0] + tok.encode(prompt)

        res = generate_schema_constrained(model, tok, inp, schema, max_new_tokens=max_new_tokens, strict_spacing=True)
        eval_res = parse_strict_json(res["text"], expected_keys)

        total_tokens += res["token_count"]
        token_counts.append(res["token_count"])
        if eval_res["syntax_valid"]: valid_syntax_count += 1
        if eval_res["schema_match"]: schema_match_count += 1
        if res["stop_reason"] == "COMPLETE_JSON": completed_objects += 1
        if res["stop_reason"] == "MAX_TOKENS": budget_exhausted_count += 1
        if res["stop_reason"] in ("NO_VALID_TOKENS", "GRAMMAR_DEADLOCK"): deadlock_count += 1

        per_record.append({
            "id": rid,
            "category": cat,
            "text": res["text"],
            "token_count": res["token_count"],
            "stop_reason": res["stop_reason"],
            "syntax_valid": eval_res["syntax_valid"],
            "schema_match": eval_res["schema_match"],
            "parsed_object": res["parsed_object"],
        })

    token_counts.sort()
    median_tokens = token_counts[len(token_counts)//2]

    summary = {
        "budget": max_new_tokens,
        "total_records": len(structured_val),
        "valid_syntax_count": valid_syntax_count,
        "valid_syntax_pct": round((valid_syntax_count / len(structured_val)) * 100, 2),
        "completed_objects": completed_objects,
        "completed_objects_pct": round((completed_objects / len(structured_val)) * 100, 2),
        "schema_match_count": schema_match_count,
        "schema_match_pct": round((schema_match_count / len(structured_val)) * 100, 2),
        "budget_exhausted_count": budget_exhausted_count,
        "deadlock_count": deadlock_count,
        "mean_tokens": round(total_tokens / len(structured_val), 2),
        "median_tokens": median_tokens,
        "per_record": per_record,
    }

    out_file = REPO_ROOT / "experiments" / "batch3111_3150" / "DIAGNOSTIC_BUDGET_96_RESULTS.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Diagnostic Budget {max_new_tokens} Results:")
    print(f"  Valid Syntax:      {valid_syntax_count}/{len(structured_val)} ({summary['valid_syntax_pct']}%)")
    print(f"  Completed Objects: {completed_objects}/{len(structured_val)} ({summary['completed_objects_pct']}%)")
    print(f"  Schema Matches:    {schema_match_count}/{len(structured_val)} ({summary['schema_match_pct']}%)")
    print(f"  Budget Exhausted:  {budget_exhausted_count}/{len(structured_val)}")
    print(f"  Deadlocks:         {deadlock_count}/{len(structured_val)}")
    print(f"  Mean Tokens:       {summary['mean_tokens']}")
    print(f"  Median Tokens:     {median_tokens}")


if __name__ == "__main__":
    run_diagnostic_budget_eval(96)
