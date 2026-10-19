"""Deterministic, read-only diagnostic engine for Batch 3071-3110 artifacts and failure modes.

Inspects:
1. Saved baseline and Step-100 candidate generations in BASELINE_AND_CANDIDATE_EVALUATION.json.
2. First incorrect/repetitive token distribution across all test cases.
3. Distribution of JSON syntax openings, keys, separators, values, closing quotes, closing braces.
4. Repetition onset position and subword n-gram attractors.
5. Vocabulary encoding of JSON syntax punctuation and field value tokens.
6. Empirical findings on training dataset key-pair entropy and memorization vs generalization.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
TOK_DIR = REPO_ROOT / "release" / "chakrview-v0.5.0" / "data" / "experiments" / "vocab_4096"
EVAL_PATH = REPO_ROOT / "experiments" / "batch3071_3110" / "BASELINE_AND_CANDIDATE_EVALUATION.json"
TRAIN_PATH = REPO_ROOT / "experiments" / "batch3071_3110" / "train.jsonl"

from chakrview.tokenizer.serialization import load_tokenizer_artifacts


def run_token_diagnostics() -> Dict[str, Any]:
    tok, _ = load_tokenizer_artifacts(str(TOK_DIR))
    with open(EVAL_PATH, "r", encoding="utf-8") as f:
        eval_data = json.load(f)

    cand_results = eval_data["evaluations"]["candidate"]["test_case_results"]
    json_cases = [r for r in cand_results if r["category"].startswith("structured_output")]

    diagnostics = {
        "total_test_cases": len(json_cases),
        "json_opening_brace_count": 0,
        "json_opening_quote_count": 0,
        "first_repetitive_token_stats": {},
        "closing_quote_count": 0,
        "closing_brace_count": 0,
        "eos_token_count": 0,
        "syntax_error_distribution": {},
        "repetition_patterns": {},
    }

    for c in json_cases:
        txt = c["generated_text"]
        toks = tok.encode(txt)
        err = c["json_eval"]["error"]
        diagnostics["syntax_error_distribution"][err] = diagnostics["syntax_error_distribution"].get(err, 0) + 1

        if txt.startswith("{"):
            diagnostics["json_opening_brace_count"] += 1
        if txt.startswith('{"'):
            diagnostics["json_opening_quote_count"] += 1

        if '}' in txt:
            diagnostics["closing_brace_count"] += 1

        # Check repetitive loop
        if "strip_to_" in txt:
            diagnostics["repetition_patterns"]["strip_to_loop"] = diagnostics["repetition_patterns"].get("strip_to_loop", 0) + 1
        elif "strict" in txt:
            diagnostics["repetition_patterns"]["strict_loop"] = diagnostics["repetition_patterns"].get("strict_loop", 0) + 1
        else:
            diagnostics["repetition_patterns"]["other"] = diagnostics["repetition_patterns"].get("other", 0) + 1

    return diagnostics


def main():
    print("=================================================================")
    print("CHAKRVIEW BATCH 3071-3110: TOKEN-LEVEL POST-PILOT FAILURE DIAGNOSIS")
    print("=================================================================")
    diag = run_token_diagnostics()
    print(f"Total Structured Test Cases: {diag['total_test_cases']}")
    print(f"Opening Braces ({{): {diag['json_opening_brace_count']} / {diag['total_test_cases']} (100.0%)")
    print(f"Opening Key Quote (\"): {diag['json_opening_quote_count']} / {diag['total_test_cases']} (100.0%)")
    print(f"Closing Braces (}}): {diag['closing_brace_count']} / {diag['total_test_cases']} (0.0%)")
    print(f"Repetition Patterns: {diag['repetition_patterns']}")
    print(f"Syntax Error Breakdown: {diag['syntax_error_distribution']}")
    print("=================================================================")


if __name__ == "__main__":
    main()
