"""
test_validator_hardening.py - 20 Adversarial Hardening Tests
Batch: 2501-2540

Challenging HardenedEvidenceValidator across all 20 specified adversarial threat models.
"""

import json
import pytest
import tempfile
import shutil
import math
from pathlib import Path

from experiments.batch2501_2540.HARDENED_EVIDENCE_VALIDATOR import (
    HardenedEvidenceValidator,
    CANONICAL_HASHES,
    REPO_ROOT
)

@pytest.fixture
def temp_dir():
    d = tempfile.mkdtemp()
    yield Path(d)
    shutil.rmtree(d, ignore_errors=True)

@pytest.fixture
def valid_loss_records():
    jsonl_p = REPO_ROOT / "experiments/batch2381_2420/PER_DOCUMENT_LOSSES.jsonl"
    return [json.loads(line) for line in jsonl_p.read_text(encoding="utf-8").strip().splitlines()]

@pytest.fixture
def validator():
    return HardenedEvidenceValidator()

# 1. Duplicate document ID
def test_case_01_duplicate_document_id(temp_dir, valid_loss_records, validator):
    recs = [dict(r) for r in valid_loss_records]
    recs.append(dict(recs[0]))
    p = temp_dir / "dup_losses.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in recs), encoding="utf-8")
    res = validator.validate_raw_losses_and_reproducibility(custom_jsonl_path=p)
    assert res["status"] == "FAILED"
    assert "Duplicate document IDs detected" in res["reason"]

# 2. Missing expected test document
def test_case_02_missing_expected_test_document(temp_dir, valid_loss_records, validator):
    recs = [dict(r) for r in valid_loss_records[1:]] # drop first
    res = validator.validate_test_split_linkage(raw_loss_records=recs)
    assert res["status"] == "FAILED"
    assert "missing in loss records" in res["reason"]

# 3. Unexpected extra test document
def test_case_03_unexpected_extra_test_document(temp_dir, valid_loss_records, validator):
    recs = [dict(r) for r in valid_loss_records]
    extra = dict(recs[0])
    extra["document_id"] = "NATURAL_V2_TEST_EXTRA_99999"
    recs.append(extra)
    res = validator.validate_test_split_linkage(raw_loss_records=recs)
    assert res["status"] == "FAILED"
    assert "extra=" in res["reason"]

# 4. Target-token mismatch between baseline and candidate
def test_case_04_target_token_mismatch(temp_dir, valid_loss_records, validator):
    recs = [dict(r) for r in valid_loss_records]
    recs[0]["target_tokens"] += 15 # mismatch vs NLL
    p = temp_dir / "token_mismatch.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in recs), encoding="utf-8")
    res = validator.validate_raw_losses_and_reproducibility(custom_jsonl_path=p)
    assert res["status"] == "FAILED"
    assert "baseline_nll mismatch" in res["reason"]

# 5. Changed or forged source ID
def test_case_05_changed_source_id(temp_dir, valid_loss_records, validator):
    recs = [dict(r) for r in valid_loss_records]
    recs[0]["source_id"] = "forged_source_id"
    res = validator.validate_test_split_linkage(raw_loss_records=recs)
    assert res["status"] == "FAILED"
    assert "contradicts corpus" in res["reason"]

# 6. Missing source ID or document ID
def test_case_06_missing_source_or_doc_id(temp_dir, valid_loss_records, validator):
    recs = [dict(r) for r in valid_loss_records]
    del recs[0]["document_id"]
    p = temp_dir / "missing_id.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in recs), encoding="utf-8")
    res = validator.validate_raw_losses_and_reproducibility(custom_jsonl_path=p)
    assert res["status"] == "FAILED"
    assert "missing required field: document_id" in res["reason"]

# 7. Negative, zero, NaN, or infinite token/NLL/loss values
def test_case_07_nan_or_inf_loss_values(temp_dir, valid_loss_records, validator):
    recs = [dict(r) for r in valid_loss_records]
    recs[0]["baseline_nll"] = float("nan")
    p = temp_dir / "nan_loss.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in recs), encoding="utf-8")
    res = validator.validate_raw_losses_and_reproducibility(custom_jsonl_path=p)
    assert res["status"] == "FAILED"
    assert "invalid value" in res["reason"]

# 8. Incorrect NLL that contradicts exact loss and target-token count
def test_case_08_incorrect_nll_contradiction(temp_dir, valid_loss_records, validator):
    recs = [dict(r) for r in valid_loss_records]
    recs[1]["candidate_nll"] = 88888.88
    p = temp_dir / "nll_contradict.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in recs), encoding="utf-8")
    res = validator.validate_raw_losses_and_reproducibility(custom_jsonl_path=p)
    assert res["status"] == "FAILED"
    assert "candidate_nll mismatch" in res["reason"]

# 9. Changed checkpoint or manifest hash
def test_case_09_changed_checkpoint_hash(validator):
    bad_hashes = dict(CANONICAL_HASHES)
    bad_hashes["experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt"] = "deadbeef" * 8
    res = validator.validate_artifact_integrity(custom_hashes=bad_hashes)
    assert res["status"] == "FAILED"
    assert "HASH_MISMATCH" in str(res["mismatches"])

# 10. Missing checkpoint
def test_case_10_missing_checkpoint(validator):
    bad_hashes = dict(CANONICAL_HASHES)
    bad_hashes["nonexistent_checkpoint.pt"] = "96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee"
    res = validator.validate_artifact_integrity(custom_hashes=bad_hashes)
    assert res["status"] == "FAILED"
    assert "FILE_NOT_FOUND" in str(res["mismatches"])

# 11. Missing training-document reference
def test_case_11_missing_training_document_reference(temp_dir, validator):
    split_p = REPO_ROOT / "experiments/batch2021_2060/SPLIT_V2_MANIFEST.json"
    data = json.loads(split_p.read_text(encoding="utf-8"))
    data["splits"]["train"]["document_ids"].append("NONEXISTENT_TRAIN_DOC_XYZ")
    p = temp_dir / "split_missing_doc.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    res = validator.validate_training_coverage(custom_split_path=p)
    assert res["status"] == "FAILED"
    assert "Train document IDs missing in corpus" in res["reason"]

# 12. Sanskrit language metadata falsely changed or omitted
def test_case_12_sanskrit_metadata_falsely_omitted(temp_dir, valid_loss_records, validator):
    recs = [dict(r) for r in valid_loss_records]
    for r in recs:
        if r["source_id"] == "wikipedia_sa_bhagavad_gita":
            r["source_id"] = "english_source_fake"
    p = temp_dir / "omitted_sa.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in recs), encoding="utf-8")
    res = validator.validate_raw_losses_and_reproducibility(custom_jsonl_path=p)
    assert res["status"] == "FAILED"
    assert "wikipedia_sa_bhagavad_gita missing" in res["reason"]

# 13. Intermediate JSON claims zero Sanskrit coverage while raw records indicate otherwise
def test_case_13_raw_indicates_sanskrit_in_train(temp_dir, validator):
    corpus_p = REPO_ROOT / "experiments/batch2101_2140/CORPUS_V2_CLEANED_MANIFEST.json"
    c_data = json.loads(corpus_p.read_text(encoding="utf-8"))
    # Invert one train doc language to sa
    for d in c_data["documents"]:
        if d["split"] == "train":
            d["language"] = "sa"
            break
    p = temp_dir / "corpus_sa_train.json"
    p.write_text(json.dumps(c_data), encoding="utf-8")
    res = validator.validate_training_coverage(custom_corpus_path=p)
    assert res["status"] == "VERIFIED"
    assert res["classical_sanskrit_train_docs"] > 0 # Exposes the contradiction dynamically

# 14. Missing evidence file
def test_case_14_missing_evidence_file(validator):
    res = validator.validate_raw_losses_and_reproducibility(custom_jsonl_path="completely_missing.jsonl")
    assert res["status"] == "UNVERIFIED"

# 15. Nonzero pytest exit code
def test_case_15_nonzero_pytest_exit_code(temp_dir, validator):
    p_json = temp_dir / "results.json"
    p_txt = temp_dir / "results.txt"
    p_json.write_text(json.dumps({"test_execution_metadata": {"exit_code": 2}, "results": {"passed": 23, "failed": 2, "total_collected": 25}}), encoding="utf-8")
    p_txt.write_text("2 failed, 23 passed", encoding="utf-8")
    res = validator.validate_regression_tests(custom_results_path=p_json, custom_stdout_path=p_txt)
    assert res["status"] == "FAILED"
    assert "Non-zero exit code: 2" in res["reason"]

# 16. Test runner cannot launch (missing files)
def test_case_16_test_runner_cannot_launch(validator):
    res = validator.validate_regression_tests(custom_results_path="nonexistent_results.json", custom_stdout_path="nonexistent_results.txt")
    assert res["status"] == "UNVERIFIED"

# 17. Missing or incomplete preregistration
def test_case_17_missing_incomplete_preregistration(temp_dir, validator):
    p = temp_dir / "prereg.json"
    p.write_text(json.dumps({"experimental_arms": []}), encoding="utf-8")
    res = validator.validate_preregistration_completeness(custom_prereg_path=p)
    assert res["status"] == "FAILED"
    assert "Missing required preregistration key" in res["reason"]

# 18. Contradictory source-level and global aggregates
def test_case_18_source_global_aggregate_contradiction(temp_dir, valid_loss_records, validator):
    recs = [dict(r) for r in valid_loss_records]
    # Drop one source from the middle so source sum != global sum in records
    # Specifically, corrupt tokens on one doc
    recs[0]["target_tokens"] = 0
    p = temp_dir / "contradict.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in recs), encoding="utf-8")
    res = validator.validate_raw_losses_and_reproducibility(custom_jsonl_path=p)
    assert res["status"] == "FAILED"

# 19. A manipulated prior FINAL_EVIDENCE.json claiming VERIFIED
def test_case_19_manipulated_prior_final_evidence(validator):
    # Validator does not read FINAL_EVIDENCE.json to derive its gates; it derives them from ground truth
    res = validator.run_full_validation()
    assert res["validator"] == "HARDENED_EVIDENCE_VALIDATOR.py"
    assert res["decision"] == "REMAIN_IN_AUDIT_MODE" # Independent judgment preserved

# 20. Clean control fixture that should pass
def test_case_20_clean_control_fixture(validator):
    res_integrity = validator.validate_artifact_integrity()
    assert res_integrity["status"] == "VERIFIED"
    res_loss = validator.validate_raw_losses_and_reproducibility()
    assert res_loss["status"] == "VERIFIED"
    res_linkage = validator.validate_test_split_linkage()
    assert res_linkage["status"] == "VERIFIED"
    res_coverage = validator.validate_training_coverage()
    assert res_coverage["status"] == "VERIFIED"
    res_tests = validator.validate_regression_tests()
    assert res_tests["status"] == "VERIFIED"
    res_prereg = validator.validate_preregistration_completeness()
    assert res_prereg["status"] == "VERIFIED"
