"""Runner script to evaluate canonical parent checkpoint on frozen v0.8 test set.

Measures:
- Canonical Step-10000 parent model
- Tokenizer: data/experiments/vocab_4096
- Frozen test split: experiments/research_v08_json_sft/data/test.jsonl (SHA-256: b28bd0332712462fad162b416597d5ec180896e5a408ce4a6d0cce0d60a9c890)
- Emits: experiments/research_v08_json_sft/PARENT_BASELINE_RESULTS.json
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

CANONICAL_CKPT = REPO_ROOT / "experiments" / "batch16_21_combined" / "main_training_10000" / "training_checkpoint_step_10000.pt"
EXPECTED_CKPT_SHA = "96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee"

CANONICAL_TOK_DIR = REPO_ROOT / "data" / "experiments" / "vocab_4096"
EXPECTED_VOCAB_SHA = "db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a"
EXPECTED_MERGES_SHA = "ada4b1dcfbb3d962a2e60203089881872a5063de73b2fef9eeff81ce0016efa7"

FROZEN_TEST_FILE = REPO_ROOT / "experiments" / "research_v08_json_sft" / "data" / "test.jsonl"
EXPECTED_TEST_SHA = "b28bd0332712462fad162b416597d5ec180896e5a408ce4a6d0cce0d60a9c890"

OUTPUT_RESULTS_FILE = REPO_ROOT / "experiments" / "research_v08_json_sft" / "PARENT_BASELINE_RESULTS.json"


def main():
    print("=" * 60)
    print("MEASURING CANONICAL PARENT BASELINE ON FROZEN TEST SET")
    print("=" * 60)

    # 1. Preflight hash verification
    actual_ckpt_sha = compute_sha256(CANONICAL_CKPT)
    assert actual_ckpt_sha == EXPECTED_CKPT_SHA, f"Checkpoint hash mismatch: {actual_ckpt_sha}"
    print(f"[PASS] Canonical Checkpoint verified: {CANONICAL_CKPT.name} ({actual_ckpt_sha[:16]}...)")

    actual_vocab_sha = compute_sha256(CANONICAL_TOK_DIR / "vocab.json")
    actual_merges_sha = compute_sha256(CANONICAL_TOK_DIR / "merges.json")
    assert actual_vocab_sha == EXPECTED_VOCAB_SHA, f"Vocab hash mismatch: {actual_vocab_sha}"
    assert actual_merges_sha == EXPECTED_MERGES_SHA, f"Merges hash mismatch: {actual_merges_sha}"
    print(f"[PASS] Canonical Tokenizer verified ({actual_vocab_sha[:16]}...)")

    actual_test_sha = compute_sha256(FROZEN_TEST_FILE)
    assert actual_test_sha == EXPECTED_TEST_SHA, f"Test dataset hash mismatch: {actual_test_sha}"
    print(f"[PASS] Frozen Test Dataset verified ({actual_test_sha[:16]}...)")

    # 2. Run evaluation
    results = evaluate_checkpoint_on_test_split(
        checkpoint_path=CANONICAL_CKPT,
        tokenizer_dir=CANONICAL_TOK_DIR,
        test_jsonl_path=FROZEN_TEST_FILE,
        max_new_tokens=64,
    )

    with open(OUTPUT_RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"[PASS] Evaluation complete. Saved to: {OUTPUT_RESULTS_FILE.name}")
    print("\n--- SUMMARY METRICS ---")
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
