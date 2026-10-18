"""
test_training_coverage_validation.py - Adversarial Coverage Verification
Batch: 2541-2580

Tests that coverage derivation rejects manipulated training records, missing IDs,
corrupted manifest counts, and unauthorized Sanskrit insertions into train split.
"""

import json
import pytest
import tempfile
import shutil
from pathlib import Path
from collections import Counter

from experiments.batch2501_2540.HARDENED_EVIDENCE_VALIDATOR import REPO_ROOT

@pytest.fixture
def temp_dir():
    d = tempfile.mkdtemp()
    yield Path(d)
    shutil.rmtree(d, ignore_errors=True)

@pytest.fixture
def split_manifest_data():
    p = REPO_ROOT / "experiments/batch2021_2060/SPLIT_V2_MANIFEST.json"
    return json.loads(p.read_text(encoding="utf-8"))

@pytest.fixture
def corpus_manifest_data():
    p = REPO_ROOT / "experiments/batch2101_2140/CORPUS_V2_CLEANED_MANIFEST.json"
    return json.loads(p.read_text(encoding="utf-8"))

def derive_coverage_check(split_data, corpus_data):
    train_ids = split_data["splits"]["train"]["document_ids"]
    corpus_docs = {d["document_id"]: d for d in corpus_data["documents"]}
    
    # 1. Duplicates
    dup_ids = [did for did, cnt in Counter(train_ids).items() if cnt > 1]
    if dup_ids:
        return {"status": "FAILED", "reason": f"Duplicate train IDs: {dup_ids}"}
        
    # 2. Missing
    missing_ids = [did for did in train_ids if did not in corpus_docs]
    if missing_ids:
        return {"status": "FAILED", "reason": f"Missing train IDs: {missing_ids[:3]}"}
        
    # 3. Sanskrit in train check
    train_docs = [corpus_docs[did] for did in train_ids]
    sa_docs = [d for d in train_docs if d.get("language") == "sa" or d.get("source_id") == "wikipedia_sa_bhagavad_gita"]
    if sa_docs:
        return {"status": "FAILED", "reason": f"Unauthorized Sanskrit documents found in train: {len(sa_docs)}"}
        
    # 4. Manifest count agreement
    manifest_doc_count = split_data["splits"]["train"]["document_count"]
    if manifest_doc_count != len(train_ids):
        return {"status": "FAILED", "reason": f"Manifest count mismatch: {manifest_doc_count} vs {len(train_ids)}"}
        
    return {"status": "VERIFIED", "train_documents": len(train_ids), "sanskrit_documents": 0}

def test_adversarial_coverage_1_missing_training_id(split_manifest_data, corpus_manifest_data):
    split_manifest_data["splits"]["train"]["document_ids"].append("NONEXISTENT_TRAIN_ID_99999")
    res = derive_coverage_check(split_manifest_data, corpus_manifest_data)
    assert res["status"] == "FAILED"
    assert "Missing train IDs" in res["reason"]

def test_adversarial_coverage_2_duplicate_training_id(split_manifest_data, corpus_manifest_data):
    first_id = split_manifest_data["splits"]["train"]["document_ids"][0]
    split_manifest_data["splits"]["train"]["document_ids"].append(first_id)
    res = derive_coverage_check(split_manifest_data, corpus_manifest_data)
    assert res["status"] == "FAILED"
    assert "Duplicate train IDs" in res["reason"]

def test_adversarial_coverage_3_altered_language_to_sanskrit(split_manifest_data, corpus_manifest_data):
    # Alter first train document language to 'sa'
    first_id = split_manifest_data["splits"]["train"]["document_ids"][0]
    for d in corpus_manifest_data["documents"]:
        if d["document_id"] == first_id:
            d["language"] = "sa"
            break
    res = derive_coverage_check(split_manifest_data, corpus_manifest_data)
    assert res["status"] == "FAILED"
    assert "Unauthorized Sanskrit documents found in train" in res["reason"]

def test_adversarial_coverage_4_sanskrit_source_placed_in_train(split_manifest_data, corpus_manifest_data):
    # Add an authentic Classical Sanskrit document (NATURAL_V2_TEST_02621) to train split
    sa_doc_id = "NATURAL_V2_TEST_02621"
    split_manifest_data["splits"]["train"]["document_ids"].append(sa_doc_id)
    split_manifest_data["splits"]["train"]["document_count"] = len(split_manifest_data["splits"]["train"]["document_ids"])
    res = derive_coverage_check(split_manifest_data, corpus_manifest_data)
    assert res["status"] == "FAILED"
    assert "Unauthorized Sanskrit documents found in train" in res["reason"]

def test_adversarial_coverage_5_manifest_count_disagrees(split_manifest_data, corpus_manifest_data):
    split_manifest_data["splits"]["train"]["document_count"] = 999
    res = derive_coverage_check(split_manifest_data, corpus_manifest_data)
    assert res["status"] == "FAILED"
    assert "Manifest count mismatch" in res["reason"]

def test_clean_training_coverage_passes(split_manifest_data, corpus_manifest_data):
    res = derive_coverage_check(split_manifest_data, corpus_manifest_data)
    assert res["status"] == "VERIFIED"
    assert res["train_documents"] == 782
    assert res["sanskrit_documents"] == 0
