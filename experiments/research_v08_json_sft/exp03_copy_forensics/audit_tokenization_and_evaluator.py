"""Step 2 & 4: Tokenization & Supervision Audit + Evaluator Unit Tests.

Audits:
1. Tokenization parity: independent string encoding vs embedded in prompt vs embedded in JSON response.
2. Token boundary differences between single quotes in prompt ('status_val') vs double quotes in response ("status_val").
3. Automated assertions that status values in prompt match expected response and all response tokens are supervised.
4. Comprehensive synthetic tests for the evaluation harness (rejecting wrong values, wrong keys, truncated strings, etc.).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chakrview.tokenizer.serialization import load_tokenizer_artifacts
from chakrview.training.sft_engine import SFTDataset
from experiments.research_v08_json_sft.exp02_string_copy.evaluate_exp02 import evaluate_output_case


def run_evaluator_validation_tests():
    print("=" * 70)
    print("RUNNING SYNTHETIC EVALUATOR UNIT TESTS")
    print("=" * 70)

    # Case 1: Exact positive match
    res1 = evaluate_output_case("single_field_string_extraction", '{"status": "nominal"}', expected_status="nominal")
    assert res1["json_parsed"] is True
    assert res1["schema_valid"] is True
    assert res1["status_match"] is True
    assert res1["full_semantic_success"] is True
    print("  [PASS] Test 1: Exact positive match accepted.")

    # Case 2: Correct syntax with WRONG value
    res2 = evaluate_output_case("single_field_string_extraction", '{"status": "degraded"}', expected_status="nominal")
    assert res2["json_parsed"] is True
    assert res2["schema_valid"] is True
    assert res2["status_match"] is False
    assert res2["full_semantic_success"] is False
    print("  [PASS] Test 2: Correct syntax with wrong value rejected.")

    # Case 3: Correct value in WRONG field key
    res3 = evaluate_output_case("single_field_string_extraction", '{"state": "nominal"}', expected_status="nominal")
    assert res3["json_parsed"] is True
    assert res3["schema_valid"] is False
    assert res3["status_match"] is False
    assert res3["full_semantic_success"] is False
    print("  [PASS] Test 3: Correct value in wrong field key rejected.")

    # Case 4: Truncated value
    res4 = evaluate_output_case("single_field_string_extraction", '{"status": "nomin"}', expected_status="nominal")
    assert res4["json_parsed"] is True
    assert res4["schema_valid"] is True
    assert res4["status_match"] is False
    assert res4["full_semantic_success"] is False
    print("  [PASS] Test 4: Truncated value rejected.")

    # Case 5: Extra characters
    res5 = evaluate_output_case("single_field_string_extraction", '{"status": "nominal_extra"}', expected_status="nominal")
    assert res5["json_parsed"] is True
    assert res5["schema_valid"] is True
    assert res5["status_match"] is False
    assert res5["full_semantic_success"] is False
    print("  [PASS] Test 5: Extra characters rejected.")

    # Case 6: Extra field present
    res6 = evaluate_output_case("single_field_string_extraction", '{"status": "nominal", "extra": 1}', expected_status="nominal")
    assert res6["json_parsed"] is True
    assert res6["schema_valid"] is False
    assert res6["status_match"] is False
    print("  [PASS] Test 6: Extra field rejected (strict schema).")

    # Case 7: Empty output
    res7 = evaluate_output_case("single_field_string_extraction", "", expected_status="nominal", is_empty=True)
    assert res7["json_parsed"] is False
    assert res7["full_semantic_success"] is False
    print("  [PASS] Test 7: Empty output rejected.")

    # Case 8: Immediate EOS
    res8 = evaluate_output_case("single_field_string_extraction", "", expected_status="nominal", first_token_is_eos=True)
    assert res8["json_parsed"] is False
    assert res8["full_semantic_success"] is False
    print("  [PASS] Test 8: Immediate EOS rejected.")

    # Case 9: Markdown wrapped json
    res9 = evaluate_output_case("single_field_string_extraction", '```json\n{"status": "nominal"}\n```', expected_status="nominal")
    assert res9["json_parsed"] is True
    assert res9["schema_valid"] is True
    assert res9["status_match"] is True
    print("  [PASS] Test 9: Markdown-wrapped json accepted.")

    print("ALL EVALUATOR UNIT TESTS PASSED SUCCESSFULLY.\n")


def audit_tokenization_and_supervision():
    print("=" * 70)
    print("AUDITING TOKENIZATION & CAUSAL SUPERVISION")
    print("=" * 70)

    tok_dir = REPO_ROOT / "data" / "experiments" / "vocab_4096"
    tokenizer, _ = load_tokenizer_artifacts(tok_dir)

    train_file = REPO_ROOT / "experiments" / "research_v08_json_sft" / "exp02_string_copy" / "data" / "train.jsonl"
    test_file = REPO_ROOT / "experiments" / "research_v08_json_sft" / "exp02_string_copy" / "data" / "test.jsonl"

    train_ds = SFTDataset(train_file, tokenizer, max_seq_len=256, strip_prompt_delimiters=False)
    test_ds = SFTDataset(test_file, tokenizer, max_seq_len=256, strip_prompt_delimiters=False, enforce_trust_gate=False)

    records_to_trace = [
        ("Train Example 0 (nominal)", train_ds.records[0]),
        ("Train Example 1 (active)", train_ds.records[1]),
        ("Held-out Test Example 0 (optical_alignment_fault)", test_ds.records[0]),
        ("Held-out Test Example 1 (beam_misaligned)", test_ds.records[1]),
    ]

    for label, rec in records_to_trace:
        print(f"\n--- {label} ---")
        prompt = rec["prompt"]
        response = rec["response"]
        expected_status = rec["expected_status"]

        # Assertions
        assert expected_status in prompt, f"Expected status {expected_status} not found in prompt"
        assert f'"{expected_status}"' in response, f"Expected status {expected_status} not formatted as string in response"

        p_toks = tokenizer.encode(prompt)
        r_toks = tokenizer.encode(response)
        s_toks = tokenizer.encode(expected_status)

        inp, tgt = train_ds.encode_example(prompt, response, append_eos=True)

        prompt_len = sum(1 for t in tgt if t == -100)
        resp_len = sum(1 for t in tgt if t != -100)

        print(f"Prompt text: {repr(prompt)}")
        print(f"Response text: {repr(response)}")
        print(f"Status value: {repr(expected_status)} -> Tokens: {s_toks} ({[tokenizer.decode([t]) for t in s_toks]})")
        print(f"Total input tokens: {len(inp)}, Masked prompt tokens: {prompt_len}, Supervised response tokens: {resp_len}")

        # Check prompt context tokens for status
        prompt_str_with_single_quotes = f"'{expected_status}'"
        prompt_sq_toks = tokenizer.encode(prompt_str_with_single_quotes)
        # Check response tokens for status
        resp_str_with_double_quotes = f'"{expected_status}"'
        resp_dq_toks = tokenizer.encode(resp_str_with_double_quotes)

        print(f"In prompt with single quotes: {repr(prompt_str_with_single_quotes)} -> {prompt_sq_toks} ({[tokenizer.decode([t]) for t in prompt_sq_toks]})")
        print(f"In response with double quotes: {repr(resp_str_with_double_quotes)} -> {resp_dq_toks} ({[tokenizer.decode([t]) for t in resp_dq_toks]})")

        # Trace supervised positions
        print("Supervised target positions:")
        for pos in range(prompt_len, len(inp)):
            prev_tok = inp[pos]
            target_tok = tgt[pos]
            prev_str = tokenizer.decode([prev_tok]) if prev_tok not in (0, 1, 2) else f"<SP_{prev_tok}>"
            target_str = tokenizer.decode([target_tok]) if target_tok != 1 else "<EOS>"
            print(f"  Pos {pos:3d} | Prev input: {prev_tok:5d} ({repr(prev_str):12s}) -> Target to predict: {target_tok:5d} ({repr(target_str):12s})")

    # Automated check across all 130 train and 30 test records
    print("\nRunning automated integrity assertions across all records...")
    for split_name, ds in [("Train", train_ds), ("Test", test_ds)]:
        for i, rec in enumerate(ds.records):
            if rec.get("category") == "single_field_string_extraction":
                st = rec["expected_status"]
                assert st in rec["prompt"], f"Split {split_name} record {i}: status not in prompt"
                assert f'"{st}"' in rec["response"], f"Split {split_name} record {i}: status not in response"
                inp, tgt = ds.encode_example(rec["prompt"], rec["response"], append_eos=True)
                assert len(inp) == len(tgt)
                # verify response is completely unmasked
                resp_tokens = tokenizer.encode(rec["response"])
                # +1 for EOS
                supervised_count = sum(1 for t in tgt if t != -100)
                assert supervised_count == len(resp_tokens) + 1, f"Mismatch in supervised token count: {supervised_count} vs {len(resp_tokens) + 1}"
    print("  [PASS] All records verified: exact status present in context, exactly formatted in response, 100% supervised.")

if __name__ == "__main__":
    run_evaluator_validation_tests()
    audit_tokenization_and_supervision()
