"""Evaluation runner for the trained JSON SFT candidate on the frozen test set.

Evaluates:
- Candidate Checkpoint: experiments/research_v08_json_sft/checkpoints/sft_best_checkpoint.pt
- Tokenizer: data/experiments/vocab_4096
- Frozen test split: experiments/research_v08_json_sft/data/test.jsonl (SHA-256: b28bd0332712462fad162b416597d5ec180896e5a408ce4a6d0cce0d60a9c890)
- Emits: experiments/research_v08_json_sft/CANDIDATE_EVALUATION_RESULTS.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.research_v08_json_sft.evaluate_checkpoint import (
    compute_sha256,
    evaluate_checkpoint_on_test_split,
)

CANDIDATE_CKPT = REPO_ROOT / "experiments" / "research_v08_json_sft" / "checkpoints" / "sft_best_checkpoint.pt"
CANONICAL_TOK_DIR = REPO_ROOT / "data" / "experiments" / "vocab_4096"
FROZEN_TEST_FILE = REPO_ROOT / "experiments" / "research_v08_json_sft" / "data" / "test.jsonl"
EXPECTED_TEST_SHA = "b28bd0332712462fad162b416597d5ec180896e5a408ce4a6d0cce0d60a9c890"

OUTPUT_RESULTS_FILE = REPO_ROOT / "experiments" / "research_v08_json_sft" / "CANDIDATE_EVALUATION_RESULTS.json"


def main():
    print("=" * 60)
    print("EVALUATING TRAINED CANDIDATE ON FROZEN TEST SET")
    print("=" * 60)

    actual_ckpt_sha = compute_sha256(CANDIDATE_CKPT)
    print(f"Candidate Checkpoint: {CANDIDATE_CKPT.name} ({actual_ckpt_sha})")

    actual_test_sha = compute_sha256(FROZEN_TEST_FILE)
    assert actual_test_sha == EXPECTED_TEST_SHA, "Test hash mismatch!"
    print(f"Frozen Test Dataset: {FROZEN_TEST_FILE.name} ({actual_test_sha[:16]}...)")

    results = evaluate_checkpoint_on_test_split(
        checkpoint_path=CANDIDATE_CKPT,
        tokenizer_dir=CANONICAL_TOK_DIR,
        test_jsonl_path=FROZEN_TEST_FILE,
        max_new_tokens=64,
    )

    with open(OUTPUT_RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n[PASS] Candidate Evaluation complete. Saved to: {OUTPUT_RESULTS_FILE.name}")
    print("\n--- CANDIDATE SUMMARY METRICS ---")
    summary = results["summary"]
    jm = summary["json_metrics"]
    print("JSON EXTRACTION METRICS (N=20):")
    print(f"  JSON Parse Rate:       {jm['json_parsed_numerator']}/{jm['json_parsed_denominator']} ({jm['json_parsed_pct']}%)")
    print(f"  Schema Valid Rate:     {jm['schema_valid_numerator']}/{jm['schema_valid_denominator']} ({jm['schema_valid_pct']}%)")
    print(f"  Status Match Rate:     {jm['status_match_numerator']}/{jm['status_match_denominator']} ({jm['status_match_pct']}%)")
    print(f"  Code Match Rate:       {jm['code_match_numerator']}/{jm['code_match_denominator']} ({jm['code_match_pct']}%)")
    print(f"  Full Semantic Success: {jm['full_semantic_success_numerator']}/{jm['full_semantic_success_denominator']} ({jm['full_semantic_success_pct']}%)")
    print(f"  Empty Outputs:         {jm['empty_outputs_numerator']}/{jm['empty_outputs_denominator']}")
    print(f"  Immediate EOS:         {jm['immediate_eos_numerator']}/{jm['immediate_eos_denominator']}")
    print(f"  Truncated (Max Tokens):{jm['max_tokens_truncated_numerator']}/{jm['max_tokens_truncated_denominator']}")

    rm = summary["replay_metrics"]
    print("\nCONTINUATION REPLAY METRICS (N=10):")
    print(f"  Non-Empty Rate:        {rm['replay_non_empty_numerator']}/{rm['replay_non_empty_denominator']} ({rm['replay_non_empty_pct']}%)")
    print(f"  Meaningful Rate:       {rm['replay_meaningful_numerator']}/{rm['replay_meaningful_denominator']} ({rm['replay_meaningful_pct']}%)")
    print("=" * 60)


if __name__ == "__main__":
    main()
