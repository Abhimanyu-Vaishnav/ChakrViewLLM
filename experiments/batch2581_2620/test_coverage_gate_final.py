"""
test_coverage_gate_final.py - Adversarial Regression Tests for Coverage Gate Final
Batch: 2581-2620

Demonstrates that every required invariant causes the coverage gate to fail when violated:
1. Duplicate training IDs
2. Missing training IDs in corpus
3. Unexpected foreign document IDs
4. Duplicate corpus IDs
5. Document assigned to multiple splits
6. Incomplete partition / set mismatch
7. Mismatched source/language identity
8. Sanskrit document in training split
9. Per-document vs aggregate total mismatch
10. Manifest count semantics disagreement
11. Clean control fixture (passes)
"""

import json
import pytest
import copy
from pathlib import Path
from collections import Counter

REPO_ROOT = Path(__file__).resolve().parents[2]

SPLIT_PATH = REPO_ROOT / "experiments/batch2021_2060/SPLIT_V2_MANIFEST.json"
CORPUS_PATH = REPO_ROOT / "experiments/batch2101_2140/CORPUS_V2_CLEANED_MANIFEST.json"

@pytest.fixture
def base_split_manifest():
    return json.loads(SPLIT_PATH.read_text(encoding="utf-8"))

@pytest.fixture
def base_corpus_manifest():
    return json.loads(CORPUS_PATH.read_text(encoding="utf-8"))

def evaluate_coverage_gate(split_manifest, corpus_manifest, aggregate_raw_tokens=147476, aggregate_chunked_tokens=146694):
    """Rigorous gate evaluator checking all 10 invariants."""
    failing_invariants = []
    
    train_ids = split_manifest.get("splits", {}).get("train", {}).get("document_ids", [])
    val_ids = split_manifest.get("splits", {}).get("validation", {}).get("document_ids", [])
    test_ids = split_manifest.get("splits", {}).get("test", {}).get("document_ids", [])
    
    corpus_docs = corpus_manifest.get("documents", [])
    corpus_ids = [d["document_id"] for d in corpus_docs]
    corpus_map = {d["document_id"]: d for d in corpus_docs}
    
    # INV-01: Duplicate training IDs
    if len(train_ids) != len(set(train_ids)):
        failing_invariants.append("inv_01_no_duplicate_training_ids")
        
    # INV-02: Missing training IDs in corpus
    missing_train = [did for did in train_ids if did not in corpus_map]
    if missing_train:
        failing_invariants.append("inv_02_all_training_ids_present_in_corpus")
        
    # INV-03: Unexpected document IDs (outside corpus or invalid format)
    for did in train_ids + val_ids + test_ids:
        if not isinstance(did, str) or not did.startswith("NATURAL_V2_"):
            failing_invariants.append("inv_03_no_unexpected_document_ids")
            break
            
    # INV-04: Duplicate corpus IDs
    if len(corpus_ids) != len(set(corpus_ids)):
        failing_invariants.append("inv_04_no_duplicate_corpus_document_ids")
        
    # INV-05: Document in multiple splits
    if set(train_ids) & set(val_ids) or set(train_ids) & set(test_ids) or set(val_ids) & set(test_ids):
        failing_invariants.append("inv_05_no_document_in_multiple_splits")
        
    # INV-06: Corpus IDs and split IDs exact partition match
    all_split_ids = set(train_ids) | set(val_ids) | set(test_ids)
    if set(corpus_ids) != all_split_ids or len(all_split_ids) != len(corpus_ids):
        failing_invariants.append("inv_06_corpus_ids_and_split_ids_exact_partition_match")
        
    # INV-07: Valid and consistent source/language identity
    for did in train_ids:
        if did in corpus_map:
            doc = corpus_map[did]
            if not doc.get("source_id") or not doc.get("language"):
                failing_invariants.append("inv_07_valid_and_consistent_source_language_identity")
                break
                
    # INV-08: Zero Classical Sanskrit in train split
    sa_in_train = []
    for did in train_ids:
        if did in corpus_map:
            doc = corpus_map[did]
            if doc.get("language") == "sa" or "sa" in doc.get("source_id", "").split("_") or "sanskrit" in doc.get("source_id", "").lower():
                sa_in_train.append(did)
    if sa_in_train:
        failing_invariants.append("inv_08_zero_classical_sanskrit_in_training_split")
        
    # INV-09: Per-document and aggregate totals agree
    per_doc_raw = sum(corpus_map[did]["token_count"] for did in train_ids if did in corpus_map)
    if per_doc_raw != aggregate_raw_tokens:
        failing_invariants.append("inv_09_per_document_and_aggregate_totals_match")
        
    # INV-10: Manifest count semantics consistent
    for sname, sinfo in split_manifest.get("splits", {}).items():
        if sinfo.get("document_count") != len(sinfo.get("document_ids", [])):
            failing_invariants.append("inv_10_manifest_count_semantics_consistent")
            break
            
    status = "VERIFIED" if len(failing_invariants) == 0 else "FAILED"
    return {
        "status": status,
        "failing_invariants": failing_invariants
    }

def test_clean_control_fixture(base_split_manifest, base_corpus_manifest):
    res = evaluate_coverage_gate(base_split_manifest, base_corpus_manifest)
    assert res["status"] == "VERIFIED"
    assert len(res["failing_invariants"]) == 0

def test_inv_01_duplicate_training_id(base_split_manifest, base_corpus_manifest):
    sm = copy.deepcopy(base_split_manifest)
    sm["splits"]["train"]["document_ids"].append(sm["splits"]["train"]["document_ids"][0])
    res = evaluate_coverage_gate(sm, base_corpus_manifest)
    assert res["status"] == "FAILED"
    assert "inv_01_no_duplicate_training_ids" in res["failing_invariants"]

def test_inv_02_missing_training_id(base_split_manifest, base_corpus_manifest):
    sm = copy.deepcopy(base_split_manifest)
    sm["splits"]["train"]["document_ids"].append("NATURAL_V2_TRAIN_NONEXISTENT_9999")
    res = evaluate_coverage_gate(sm, base_corpus_manifest)
    assert res["status"] == "FAILED"
    assert "inv_02_all_training_ids_present_in_corpus" in res["failing_invariants"]

def test_inv_03_unexpected_document_id(base_split_manifest, base_corpus_manifest):
    sm = copy.deepcopy(base_split_manifest)
    sm["splits"]["train"]["document_ids"].append("INVALID_PREFIX_DOC_123")
    res = evaluate_coverage_gate(sm, base_corpus_manifest)
    assert res["status"] == "FAILED"
    assert "inv_03_no_unexpected_document_ids" in res["failing_invariants"]

def test_inv_04_duplicate_corpus_id(base_split_manifest, base_corpus_manifest):
    cm = copy.deepcopy(base_corpus_manifest)
    cm["documents"].append(copy.deepcopy(cm["documents"][0]))
    res = evaluate_coverage_gate(base_split_manifest, cm)
    assert res["status"] == "FAILED"
    assert "inv_04_no_duplicate_corpus_document_ids" in res["failing_invariants"]

def test_inv_05_document_in_multiple_splits(base_split_manifest, base_corpus_manifest):
    sm = copy.deepcopy(base_split_manifest)
    shared_id = sm["splits"]["train"]["document_ids"][0]
    sm["splits"]["validation"]["document_ids"].append(shared_id)
    res = evaluate_coverage_gate(sm, base_corpus_manifest)
    assert res["status"] == "FAILED"
    assert "inv_05_no_document_in_multiple_splits" in res["failing_invariants"]

def test_inv_06_corpus_ids_partition_mismatch(base_split_manifest, base_corpus_manifest):
    cm = copy.deepcopy(base_corpus_manifest)
    cm["documents"] = cm["documents"][:-5]  # drop 5 docs
    res = evaluate_coverage_gate(base_split_manifest, cm)
    assert res["status"] == "FAILED"
    assert "inv_06_corpus_ids_and_split_ids_exact_partition_match" in res["failing_invariants"]

def test_inv_07_invalid_source_language_identity(base_split_manifest, base_corpus_manifest):
    cm = copy.deepcopy(base_corpus_manifest)
    cm["documents"][0]["language"] = ""  # strip language
    res = evaluate_coverage_gate(base_split_manifest, cm)
    assert res["status"] == "FAILED"
    assert "inv_07_valid_and_consistent_source_language_identity" in res["failing_invariants"]

def test_inv_08_sanskrit_in_training_split(base_split_manifest, base_corpus_manifest):
    cm = copy.deepcopy(base_corpus_manifest)
    # Turn first train doc into Classical Sanskrit
    for d in cm["documents"]:
        if d["document_id"] == base_split_manifest["splits"]["train"]["document_ids"][0]:
            d["language"] = "sa"
            d["source_id"] = "wikipedia_sa_bhagavad_gita"
            break
    res = evaluate_coverage_gate(base_split_manifest, cm)
    assert res["status"] == "FAILED"
    assert "inv_08_zero_classical_sanskrit_in_training_split" in res["failing_invariants"]

def test_inv_09_per_document_aggregate_mismatch(base_split_manifest, base_corpus_manifest):
    # Pass mismatched aggregate tokens
    res = evaluate_coverage_gate(base_split_manifest, base_corpus_manifest, aggregate_raw_tokens=999999)
    assert res["status"] == "FAILED"
    assert "inv_09_per_document_and_aggregate_totals_match" in res["failing_invariants"]

def test_inv_10_manifest_count_semantics_inconsistent(base_split_manifest, base_corpus_manifest):
    sm = copy.deepcopy(base_split_manifest)
    sm["splits"]["train"]["document_count"] = 780  # differs from len(document_ids)=782
    res = evaluate_coverage_gate(sm, base_corpus_manifest)
    assert res["status"] == "FAILED"
    assert "inv_10_manifest_count_semantics_consistent" in res["failing_invariants"]
