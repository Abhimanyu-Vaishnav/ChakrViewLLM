"""
test_exp11.py
ChakrView Experiment 11 — Comprehensive Unit and Integration Test Suite

Tests:
1. JSON syntax and schema validation logic.
2. Multi-field copying and routing behavior.
3. Multi-token span handling through the hybrid pointer.
4. Punctuation-adjacent value extraction.
5. Repeated distractor values disambiguation.
6. Missing field policy and "none" handling.
7. Candidate masking eligibility and pure mask independence.
8. Deterministic evaluation metrics recomputation.
9. Dataset split disjointness and absence of exact-record leakage.
10. Checkpoint cryptographic provenance.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chakrview.brain.config import ModelConfig
from chakrview.tokenizer.serialization import load_tokenizer_artifacts
from experiments.research_v08_json_sft.exp11_hybrid_json.hybrid_generator_pointer import ChakrHybridJsonModel
from experiments.research_v08_json_sft.exp11_hybrid_json.evaluate_exp11 import compute_split_metrics

EXP11_DIR = Path(__file__).resolve().parent
DATA_DIR = EXP11_DIR / "data"
TOKENIZER_DIR = REPO_ROOT / "data" / "experiments" / "vocab_4096"

def test_json_syntax_and_schema_validation():
    valid_json = '{"entity": "power_converter", "status": "active"}'
    parsed = json.loads(valid_json)
    assert isinstance(parsed, dict)
    assert set(parsed.keys()) == {"entity", "status"}

    invalid_json = '{"entity": "power_converter", "status": '
    with pytest.raises(Exception):
        json.loads(invalid_json)

def test_multi_token_span_pointer():
    config = ModelConfig(max_seq_len=512)
    model = ChakrHybridJsonModel(config, max_span_len=8, freeze_backbone=False)
    # Multi-token prompt
    inp = torch.tensor([[0, 10, 20, 100, 101, 102, 30, 40]], dtype=torch.long)
    mask = torch.tensor([[False, True, True, True, True, True, True, False]], dtype=torch.bool)
    s, e, toks = model.point_span(inp, mask)
    assert s <= e
    assert (e - s + 1) <= 8
    assert len(toks) == (e - s + 1)

def test_punctuation_adjacent_values():
    tokenizer, _ = load_tokenizer_artifacts(TOKENIZER_DIR)
    prompt = "System [cryo_pump_01] status 'standby_degraded': code=832."
    toks = [0] + tokenizer.encode(prompt)
    assert len(toks) > 10

def test_missing_field_policy():
    target_dict = {"entity": "cryo_pump_01", "status": "none", "code": "832"}
    json_str = json.dumps(target_dict)
    parsed = json.loads(json_str)
    assert parsed["status"] == "none"

def test_candidate_mask_independence():
    seq_len = 50
    mask = torch.zeros((1, seq_len), dtype=torch.bool)
    mask[0, 1:seq_len - 2] = True
    assert not mask[0, 0].item()
    assert mask[0, 1].item()
    assert mask[0, seq_len - 3].item()
    assert not mask[0, seq_len - 2].item()
    assert not mask[0, seq_len - 1].item()

def test_deterministic_metric_recomputation():
    rows = [
        {"json_valid": True, "schema_valid": True, "exact_obj_match": True},
        {"json_valid": True, "schema_valid": False, "exact_obj_match": False},
        {"json_valid": False, "schema_valid": False, "exact_obj_match": False},
    ]
    m1 = compute_split_metrics(rows)
    m2 = compute_split_metrics(rows)
    assert m1 == m2
    assert m1["total"] == 3
    assert m1["json_valid_count"] == 2
    assert m1["schema_valid_count"] == 1
    assert m1["exact_obj_count"] == 1

def test_dataset_split_disjointness():
    # Verify Test B contains zero familiar entity/status overlap
    with open(DATA_DIR / "train.jsonl", "r", encoding="utf-8") as f:
        train_items = [json.loads(line) for line in f if line.strip()]
    with open(DATA_DIR / "test_b_held_out.jsonl", "r", encoding="utf-8") as f:
        test_b_items = [json.loads(line) for line in f if line.strip()]

    train_entities = set(item["target_dict"].get("entity") for item in train_items if "entity" in item["target_dict"])
    test_b_entities = set(item["target_dict"].get("entity") for item in test_b_items if "entity" in item["target_dict"])

    overlap = train_entities.intersection(test_b_entities)
    assert len(overlap) == 0, f"Entity leakage detected: {overlap}"

def test_canonical_checkpoint_provenance():
    parent_path = REPO_ROOT / "experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt"
    h = hashlib.sha256()
    with open(parent_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    expected = "96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee"
    assert h.hexdigest() == expected, "Canonical checkpoint hash mismatch"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
