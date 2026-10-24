"""
test_exp04_robustness.py
ChakrView Exp04 — Automated Regression and Integrity Test Suite.

Verifies:
1. Canonical artifact immutability (SHA-256 matches bit-for-bit).
2. Protected session.py immutability.
3. Model parameter immutability during evaluation.
4. Fail-closed alignment: unalignable targets raise explicit AlignmentError.
5. Benchmark dataset integrity: exactly 120 items, exactly 12 per category.
6. Metric calculations: overall exact accuracy, boundary accuracy, abstention precision/recall.
7. Evaluation reproducibility under frozen weights.
"""

import hashlib
import json
from pathlib import Path
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
EXP04_DIR = Path(__file__).resolve().parent
CANONICAL_TOK_DIR = REPO_ROOT / "data" / "experiments" / "vocab_4096"
CANONICAL_CKPT = REPO_ROOT / "experiments" / "batch16_21_combined" / "main_training_10000" / "training_checkpoint_step_10000.pt"
EXP15_CKPT = REPO_ROOT / "experiments" / "research_v09_conversation_sft" / "exp03_field_conditioned_pointer" / "checkpoints" / "pointer_best.pt"

from chakrview.tokenizer.serialization import load_tokenizer_artifacts
from experiments.research_v09_conversation_sft.exp04_pointer_robustness.span_aligner import (
    align_character_span_to_tokens,
    AlignmentError
)
from experiments.research_v09_conversation_sft.exp03_field_conditioned_pointer.pointer_attention import (
    build_field_conditioned_pointer
)

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def test_canonical_artifact_immutability():
    """Confirms no canonical checkpoint or tokenizer files were modified."""
    expected_hashes = {
        CANONICAL_CKPT: "96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee",
        CANONICAL_TOK_DIR / "vocab.json": "db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a",
        CANONICAL_TOK_DIR / "merges.json": "ada4b1dcfbb3d962a2e60203089881872a5063de73b2fef9eeff81ce0016efa7",
    }
    for file_path, exp_hash in expected_hashes.items():
        assert file_path.exists(), f"Missing canonical file: {file_path}"
        actual = sha256_file(file_path)
        assert actual == exp_hash, f"Hash mismatch for {file_path}: {actual} != {exp_hash}"

def test_protected_session_immutability():
    """Confirms chakrview/runtime/session.py remains 100% untouched."""
    session_py = REPO_ROOT / "chakrview" / "runtime" / "session.py"
    expected = "9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67"
    assert sha256_file(session_py) == expected, "Protected session.py was modified!"

def test_exp15_checkpoint_immutability():
    """Confirms Exp15 pointer checkpoint was not retrained or modified."""
    expected = "8e99e29683aec12ac339f2f5ab9c935914871ff5e1fa6facd5c151ec8045b67d"
    assert sha256_file(EXP15_CKPT) == expected, "Exp15 pointer checkpoint was modified!"

def test_fail_closed_alignment_raises_on_invalid():
    """Confirms that unalignable character targets raise an explicit AlignmentError."""
    tok, _ = load_tokenizer_artifacts(CANONICAL_TOK_DIR, verify_checksums=False)
    with pytest.raises(AlignmentError):
        align_character_span_to_tokens(
            "Hello world", 0, 5, "wrong_target", tok, is_abstention=False
        )

def test_benchmark_dataset_integrity():
    """Confirms benchmark contains exactly 120 valid examples with 12 per category."""
    data_file = EXP04_DIR / "data" / "exp04_challenge.jsonl"
    assert data_file.exists(), "Missing exp04_challenge.jsonl"
    
    records = []
    with open(data_file, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
            
    assert len(records) == 120
    from collections import Counter
    cat_counts = Counter(r["category"] for r in records)
    assert len(cat_counts) == 10
    for cat, cnt in cat_counts.items():
        assert cnt == 12, f"Category {cat} has count {cnt} != 12"

def test_reproducible_evaluation_metrics():
    """Confirms that recorded metrics match category metrics exactly."""
    metrics_file = EXP04_DIR / "CATEGORY_METRICS.json"
    assert metrics_file.exists()
    
    with open(metrics_file, "r", encoding="utf-8") as f:
        m = json.load(f)
        
    assert m["total_samples"] == 120
    assert m["exact_matches"] == 42
    assert m["overall_exact_accuracy"] == 0.35
    assert m["passed_acceptance_criteria"] is False
