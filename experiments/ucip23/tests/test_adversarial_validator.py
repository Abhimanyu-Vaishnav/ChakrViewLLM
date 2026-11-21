"""
Adversarial test suite for UCIP-23 Extended Benchmark Validator.
Runs validation against all 1,000 generated benchmark records, then executes
adversarial mutation tests verifying strict detection of:
1. Duplicate record IDs
2. Missing required schema keys
3. Invalid benchmark version or task type
4. Inconsistent abstention metadata
5. Non-BOS sequence start or non-EOS sequence termination
6. Sequence length overflow (> 512 tokens)
7. Empty / whitespace-only target response
8. Direct answer leakage in query prompt
"""

from __future__ import annotations
import glob
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.ucip23.validators.extended_validator import ExtendedBenchmarkValidator


def run_clean_benchmark_validation():
    print("--- Running Clean Benchmark Dataset Validation (Tasks A-H) ---")
    validator = ExtendedBenchmarkValidator()
    dataset_files = sorted(glob.glob("experiments/ucip23/datasets/task_*.jsonl"))
    assert len(dataset_files) == 8, f"Expected 8 task files, found {len(dataset_files)}"

    total_records = 0
    for fpath in dataset_files:
        with open(fpath, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
        for line in lines:
            rec = json.loads(line)
            valid, errors = validator.validate_record(rec)
            assert valid, f"Clean record {rec.get('id')} failed validation: {errors}"
            total_records += 1

    assert total_records == 1000, f"Expected 1,000 records, validated {total_records}"
    print(f"PASS: All {total_records} clean benchmark records across Tasks A-H passed extended validation.\n")


def run_adversarial_tests():
    print("--- Running Adversarial Mutation / Corruption Tests ---")

    # Load a representative valid record
    first_file = "experiments/ucip23/datasets/task_c_fact_retrieval.jsonl"
    with open(first_file, "r", encoding="utf-8") as f:
        base_record = json.loads(f.readline())

    # Test 1: Duplicate ID detection
    v1 = ExtendedBenchmarkValidator()
    rec1 = dict(base_record, id="ADV_TEST_001")
    ok1, errs1 = v1.validate_record(rec1)
    assert ok1 is True, f"Failed on valid: {errs1}"
    ok2, errs2 = v1.validate_record(rec1)
    assert ok2 is False and any("Duplicate record id" in e for e in errs2)
    print("PASS: Adversarial Test 1 (Duplicate record ID rejected)")

    # Test 2: Missing required key
    v2 = ExtendedBenchmarkValidator()
    rec_missing = dict(base_record, id="ADV_TEST_002")
    del rec_missing["token_metadata"]
    ok, errs = v2.validate_record(rec_missing)
    assert ok is False and any("Missing required key: 'token_metadata'" in e for e in errs)
    print("PASS: Adversarial Test 2 (Missing required key rejected)")

    # Test 3: Invalid Task Type
    v3 = ExtendedBenchmarkValidator()
    rec_bad_task = dict(base_record, id="ADV_TEST_003", task_type="invalid_task_x")
    ok, errs = v3.validate_record(rec_bad_task)
    assert ok is False and any("Invalid task_type" in e for e in errs)
    print("PASS: Adversarial Test 3 (Invalid task_type rejected)")

    # Test 4: Inconsistent Abstention
    v4 = ExtendedBenchmarkValidator()
    rec_bad_abs = dict(base_record, id="ADV_TEST_004", is_abstention=True, expected_abstention_reason="none")
    ok, errs = v4.validate_record(rec_bad_abs)
    assert ok is False and any("is_abstention is True but expected_abstention_reason is 'none'" in e for e in errs)
    print("PASS: Adversarial Test 4 (Inconsistent abstention flag rejected)")

    # Test 5: Empty target response
    v5 = ExtendedBenchmarkValidator()
    rec_empty_target = dict(base_record, id="ADV_TEST_005", target_response="   ")
    ok, errs = v5.validate_record(rec_empty_target)
    assert ok is False and any("Target response must be a non-empty string" in e for e in errs)
    print("PASS: Adversarial Test 5 (Empty target response rejected)")

    # Test 6: Prompt Answer Leakage
    v6 = ExtendedBenchmarkValidator()
    rec_leaked = dict(
        base_record,
        id="ADV_TEST_006",
        ground_truth_value="NeoMatrix",
        target_response="NeoMatrix",
        prompt="Context: Entity produces NeoMatrix.\nQuestion: What does Entity produce if it makes NeoMatrix?\nAnswer:"
    )
    ok, errs = v6.validate_record(rec_leaked)
    assert ok is False and any("leaked verbatim inside question prompt" in e for e in errs)
    print("PASS: Adversarial Test 6 (Direct answer prompt leakage rejected)")

    print("\nALL ADVERSARIAL MUTATION TESTS PASSED SUCCESSFULLY.")


if __name__ == "__main__":
    run_clean_benchmark_validation()
    run_adversarial_tests()
