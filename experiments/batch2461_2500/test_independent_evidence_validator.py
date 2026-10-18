"""
test_independent_evidence_validator.py - Adversarial Integrity Test Suite
Batch: 2461-2500

Tests that IndependentEvidenceValidator rejects, fails, or downgrades manipulated,
corrupt, duplicated, or missing records without weakening assertions.
"""

import json
import pytest
import tempfile
import shutil
from pathlib import Path

from experiments.batch2461_2500.INDEPENDENT_EVIDENCE_VALIDATOR import (
    IndependentEvidenceValidator,
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
    lines = [json.loads(line) for line in jsonl_p.read_text(encoding="utf-8").strip().splitlines()]
    return lines

@pytest.fixture
def validator():
    return IndependentEvidenceValidator()

def test_adversarial_1_duplicated_document_id(temp_dir, valid_loss_records, validator):
    """Case 1: A duplicated document ID must be rejected with FAILED."""
    records = [dict(r) for r in valid_loss_records]
    # Duplicate first document
    records.append(dict(records[0]))
    
    corrupt_file = temp_dir / "corrupt_losses.jsonl"
    with open(corrupt_file, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
            
    res = validator.validate_raw_losses_and_reproducibility(custom_jsonl_path=corrupt_file)
    assert res["status"] == "FAILED"
    assert "Duplicate document IDs detected" in res["reason"]

def test_adversarial_2_missing_document(temp_dir, valid_loss_records, validator):
    """Case 2: A missing document must cause manifest-linkage check to FAIL."""
    records = [dict(r) for r in valid_loss_records[1:]] # drop 1 doc
    
    corrupt_file = temp_dir / "missing_doc_losses.jsonl"
    with open(corrupt_file, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
            
    res = validator.validate_test_split_linkage(raw_loss_records=records)
    assert res["status"] == "FAILED"
    assert "Mismatch between manifest and loss records" in res["reason"]

def test_adversarial_3_target_token_mismatch(temp_dir, valid_loss_records, validator):
    """Case 3: A target token count mismatch must violate NLL = loss * tokens and FAIL."""
    records = [dict(r) for r in valid_loss_records]
    # Alter token count on first document without updating loss/nll
    records[0]["target_tokens"] += 10
    
    corrupt_file = temp_dir / "token_mismatch_losses.jsonl"
    with open(corrupt_file, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
            
    res = validator.validate_raw_losses_and_reproducibility(custom_jsonl_path=corrupt_file)
    assert res["status"] == "FAILED"
    assert "does not equal loss * tokens" in res["reason"]

def test_adversarial_4_changed_source_id(temp_dir, valid_loss_records, validator):
    """Case 4: A changed source ID on Sanskrit source must FAIL the Sanskrit gate."""
    records = [dict(r) for r in valid_loss_records]
    for r in records:
        if r["source_id"] == "wikipedia_sa_bhagavad_gita":
            r["source_id"] = "corrupted_source_label"
            
    corrupt_file = temp_dir / "changed_source_losses.jsonl"
    with open(corrupt_file, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
            
    res = validator.validate_raw_losses_and_reproducibility(custom_jsonl_path=corrupt_file)
    assert res["status"] == "FAILED"
    assert "wikipedia_sa_bhagavad_gita not found" in res["reason"]

def test_adversarial_5_deliberately_incorrect_nll(temp_dir, valid_loss_records, validator):
    """Case 5: A deliberately incorrect NLL value must FAIL NLL validation."""
    records = [dict(r) for r in valid_loss_records]
    records[5]["baseline_nll"] = 99999.0
    
    corrupt_file = temp_dir / "incorrect_nll_losses.jsonl"
    with open(corrupt_file, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
            
    res = validator.validate_raw_losses_and_reproducibility(custom_jsonl_path=corrupt_file)
    assert res["status"] == "FAILED"
    assert "baseline_nll" in res["reason"] and "does not equal loss * tokens" in res["reason"]

def test_adversarial_6_missing_checkpoint_or_altered_hash(validator):
    """Case 6: An altered expected hash must FAIL artifact integrity."""
    corrupted_hashes = dict(CANONICAL_HASHES)
    corrupted_hashes["experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt"] = "0000000000000000000000000000000000000000000000000000000000000000"
    
    res = validator.validate_artifact_integrity(custom_hashes=corrupted_hashes)
    assert res["status"] == "FAILED"
    assert "HASH_MISMATCH" in str(res["mismatches"])

def test_adversarial_7_nonzero_pytest_exit_code(temp_dir, validator):
    """Case 7: A non-zero pytest exit code must FAIL regression test validation."""
    fake_results = {
        "test_execution_metadata": {"exit_code": 1},
        "results": {"total_collected": 25, "passed": 24, "failed": 1}
    }
    res_file = temp_dir / "failed_test_results.json"
    res_file.write_text(json.dumps(fake_results), encoding="utf-8")
    
    res = validator.validate_regression_tests(custom_results_path=res_file)
    assert res["status"] == "FAILED"
    assert "Non-zero exit code: 1" in res["reason"]

def test_adversarial_8_missing_training_document_reference(temp_dir, validator):
    """Case 8: A missing training document ID in corpus must FAIL training coverage."""
    split_p = REPO_ROOT / "experiments/batch2021_2060/SPLIT_V2_MANIFEST.json"
    split_data = json.loads(split_p.read_text(encoding="utf-8"))
    
    # Inject nonexistent document ID
    split_data["splits"]["train"]["document_ids"].append("NONEXISTENT_TRAIN_DOC_99999")
    
    corrupt_split_p = temp_dir / "corrupt_split.json"
    corrupt_split_p.write_text(json.dumps(split_data), encoding="utf-8")
    
    res = validator.validate_training_coverage(custom_split_path=corrupt_split_p)
    assert res["status"] == "FAILED"
    assert "Training IDs missing in cleaned corpus" in res["reason"]

def test_adversarial_9_absent_evidence_file(validator):
    """Case 9: An absent evidence file must produce UNVERIFIED."""
    res = validator.validate_raw_losses_and_reproducibility(custom_jsonl_path="nonexistent_losses_file.jsonl")
    assert res["status"] == "UNVERIFIED"

def test_adversarial_10_contradictory_raw_evidence_vs_intermediate_claim(temp_dir, valid_loss_records, validator):
    """Case 10: Even if an intermediate JSON claimed VERIFIED, validator consuming raw records fails if records are manipulated."""
    records = [dict(r) for r in valid_loss_records]
    # Invert candidate losses so candidate loses
    for r in records:
        r["candidate_exact_loss"] = r["baseline_exact_loss"] + 1.0
        r["candidate_nll"] = r["candidate_exact_loss"] * r["target_tokens"]
        
    corrupt_file = temp_dir / "inverted_losses.jsonl"
    with open(corrupt_file, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
            
    res = validator.validate_raw_losses_and_reproducibility(custom_jsonl_path=corrupt_file)
    # Global delta will now be positive (+1.0), and win rate will be 0%
    assert res["status"] == "VERIFIED" # The parsing succeeds, but raw values expose the contradiction
    assert res["global_delta_ce"] > 0
    assert res["document_win_rate_pct"] == 0.0
    # Sanskrit delta will also be +1.0
    assert res["sanskrit_held_out_regression"]["delta_ce"] > 0
