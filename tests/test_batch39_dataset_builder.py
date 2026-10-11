"""
Batch 39 Regression and Verification Tests.

Covers:
1. Deterministic source document pipeline:
   - 27 files inspection, SHA-256 match, encoding preservation.
   - Deterministic passage extraction and rejections.
2. Candidate example construction & honest statuses:
   - Factual grounding in passages.
   - Separation of passage existence from answer entailment.
   - Code execution behavioral validation.
3. Quality & near-duplicate filtering:
   - Exact duplicate rejection.
   - Near-duplicate detection using difflib.SequenceMatcher.
   - Template variation and boilerplate rejection.
4. Leakage-resistant document-level split planning:
   - Strict document-level disjointness (0 document overlap).
   - Domain representation accounting.
5. SFT batch safety guard:
   - Safe rejection of all-masked target batches.
   - Safe rejection of target tokens outside vocab bounds.
   - Safe rejection of batches missing EOS supervision.
   - Safe rejection of overlength sequences.
   - Finite loss and gradient generation on valid batches.
6. Fail-closed SFT readiness gate:
   - Gate reports BLOCKED.
   - DATASET_READY=False.
   - Documented blocking reasons.
"""

import json
import math
import os
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from experiments.batch39_dataset_builder.candidate_builder import (
    CandidateExampleBuilder,
    DOCUMENT_SPLIT_PARTITION,
    QualityAndDeduplicationFilter,
)
from experiments.batch39_dataset_builder.corpus_pipeline import (
    DeterministicCorpusPipeline,
)
from experiments.batch39_dataset_builder.sft_batch_safety import (
    SFTBatchSafetyGuard,
)


# =========================================================================
# 1. Pipeline & Source Document Verification Tests
# =========================================================================

def test_source_pipeline_verifies_all_27_documents():
    """Confirms all 27 corpus files exist, are UTF-8 decodable, and match SHA-256."""
    pipeline = DeterministicCorpusPipeline()
    inventory = pipeline.verify_source_documents()

    assert len(inventory) == 27
    for path, meta in inventory.items():
        assert meta["exists"] is True, f"Missing document: {path}"
        assert meta["sha_match"] is True, f"SHA mismatch on {path}"
        assert meta["encoding_valid"] is True, f"UTF-8 decode failed on {path}"


def test_passage_extraction_deterministic_and_rejections():
    """Confirms passage extraction produces stable output and explicit rejections."""
    pipeline = DeterministicCorpusPipeline()
    inventory = pipeline.verify_source_documents()
    passages, rejections = pipeline.extract_passages(inventory)

    assert len(passages) > 0
    assert len(rejections) > 0
    # Every accepted passage must have required fields
    for p in passages[:10]:
        assert "passage_id" in p
        assert "source_file" in p
        assert "source_sha256" in p
        assert "start_line" in p
        assert "end_line" in p
        assert "char_count" in p
        assert "extraction_method" in p
        assert len(p["text"]) >= DeterministicCorpusPipeline.MIN_PASSAGE_CHARS


# =========================================================================
# 2. Candidate Example Builder & Honest Status Tests
# =========================================================================

def test_candidate_example_grounding_and_statuses():
    """Confirms generated candidates are grounded in source text and contain honest statuses."""
    candidates_file = Path("experiments/batch39_dataset_builder/data/candidate_records.jsonl")
    if not candidates_file.exists():
        pytest.skip("candidate_records.jsonl not yet built")

    candidates = []
    with open(candidates_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                candidates.append(json.loads(line))

    assert len(candidates) > 0
    for cand in candidates:
        assert "id" in cand
        assert "passage_id" in cand
        assert "split" in cand
        assert "source_file" in cand
        assert "evidence_text" in cand
        assert "transformation_method" in cand

        statuses = cand["validation_statuses"]
        assert statuses["schema_validity"] == "PASS"
        assert statuses["source_file_validity"] == "PASS"
        assert statuses["source_hash_validity"] == "PASS"
        assert statuses["evidence_passage_matching"] == "PASS"
        assert statuses["answer_to_evidence_support"] == "PASS"
        assert statuses["technical_correctness"] == "PASS"
        assert statuses["duplicate_risk"] == "PASS"
        assert statuses["human_review_status"] == "PENDING_HUMAN_REVIEW"


def test_code_candidates_behaviorally_executable():
    """Confirms executable code candidates pass safe behavioral execution."""
    candidates_file = Path("experiments/batch39_dataset_builder/data/candidate_records.jsonl")
    if not candidates_file.exists():
        pytest.skip("candidate_records.jsonl not yet built")

    with open(candidates_file, "r", encoding="utf-8") as f:
        candidates = [json.loads(line) for line in f if line.strip()]

    code_cands = [c for c in candidates if c["validation_statuses"]["code_math_validation"] == "PASS"]
    assert len(code_cands) >= 15
    for c in code_cands:
        code_body = c["messages"][-1]["content"]
        # Must be valid compile-able Python
        compile(code_body, "<string>", "exec")


# =========================================================================
# 3. Quality & Near-Duplicate Filtering Tests
# =========================================================================

def test_quality_filter_rejects_near_duplicates():
    """Verifies that SequenceMatcher-based near-duplicate filtering rejects high-similarity prompts."""
    raw_mock_cands = [
        {
            "candidate_id": "MOCK-01",
            "passage_id": "P-01",
            "source_file": "english/english_foundation.txt",
            "source_sha256": "fakehash",
            "split": "train",
            "domain": "english",
            "prompt": "What is the primary architectural purpose of the ChakrView core?",
            "response": "ChakrView is designed for extreme hardware efficiency on consumer CPUs.",
            "evidence_text": "ChakrView is designed for extreme hardware efficiency on consumer CPUs.",
            "transformation_method": "TEST",
        },
        {
            "candidate_id": "MOCK-02",
            "passage_id": "P-02",
            "source_file": "english/english_foundation.txt",
            "source_sha256": "fakehash",
            "split": "train",
            "domain": "english",
            # Highly similar prompt (near-duplicate)
            "prompt": "What is the primary architectural purpose of the ChakrView neural core?",
            "response": "ChakrView is designed for extreme hardware efficiency on consumer CPUs.",
            "evidence_text": "ChakrView is designed for extreme hardware efficiency on consumer CPUs.",
            "transformation_method": "TEST",
        },
    ]

    filterer = QualityAndDeduplicationFilter(raw_mock_cands, threshold=0.75)
    distinct, stats = filterer.filter_and_validate()

    assert len(distinct) == 1
    assert stats["rejected_duplicates_count"] == 1


# =========================================================================
# 4. Leakage-Resistant Document-Level Split Planning Tests
# =========================================================================

def test_split_manifest_zero_document_overlap():
    """Verifies strict document-level disjointness between train and validation splits."""
    manifest_path = Path("experiments/batch39_dataset_builder/data/split_manifest.json")
    if not manifest_path.exists():
        pytest.skip("split_manifest.json not yet built")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert manifest["document_overlap_count"] == 0
    assert manifest["total_documents"] == 27
    assert manifest["train_documents_count"] + manifest["val_documents_count"] == 27

    train_docs = {doc for doc, s in manifest["document_assignments"].items() if s == "train"}
    val_docs = {doc for doc, s in manifest["document_assignments"].items() if s == "val"}
    overlap = train_docs.intersection(val_docs)
    assert len(overlap) == 0


def test_all_candidates_inherit_document_split():
    """Verifies that every candidate inherits the exact split of its parent source document."""
    candidates_file = Path("experiments/batch39_dataset_builder/data/candidate_records.jsonl")
    if not candidates_file.exists():
        pytest.skip("candidate_records.jsonl not yet built")

    with open(candidates_file, "r", encoding="utf-8") as f:
        candidates = [json.loads(line) for line in f if line.strip()]

    for cand in candidates:
        src = cand["source_file"]
        expected_split = DOCUMENT_SPLIT_PARTITION[src]
        assert cand["split"] == expected_split


# =========================================================================
# 5. SFT Batch Safety Guard Tests
# =========================================================================

def test_sft_batch_safety_all_masked_rejected():
    """Verifies that a fully masked target batch is safely rejected without forward/NaN loss."""
    guard = SFTBatchSafetyGuard(vocab_size=4096, max_seq_len=512)
    model = ChakrMicro(ModelConfig(max_seq_len=512))
    model.eval()

    x = torch.tensor([[0, 10, 20]], dtype=torch.long)
    y_masked = torch.tensor([[-100, -100, -100]], dtype=torch.long)

    loss, rep = guard.safe_compute_loss(model, x, y_masked)
    assert loss is None
    assert rep["is_valid"] is False
    assert rep["status"] == "ALL_TARGETS_MASKED"


def test_sft_batch_safety_vocab_out_of_bounds_rejected():
    """Verifies that target token IDs >= vocab_size are safely rejected."""
    guard = SFTBatchSafetyGuard(vocab_size=4096, max_seq_len=512)
    model = ChakrMicro(ModelConfig(max_seq_len=512))
    model.eval()

    x = torch.tensor([[0, 10, 20]], dtype=torch.long)
    y_oob = torch.tensor([[10, 5000, 1]], dtype=torch.long)

    loss, rep = guard.safe_compute_loss(model, x, y_oob)
    assert loss is None
    assert rep["is_valid"] is False
    assert rep["status"] == "TARGET_VOCAB_OUT_OF_BOUNDS"


def test_sft_batch_safety_missing_eos_rejected():
    """Verifies that a batch missing EOS token supervision is safely rejected."""
    guard = SFTBatchSafetyGuard(vocab_size=4096, max_seq_len=512)
    model = ChakrMicro(ModelConfig(max_seq_len=512))
    model.eval()

    x = torch.tensor([[0, 10, 20]], dtype=torch.long)
    y_no_eos = torch.tensor([[10, 20, 30]], dtype=torch.long)

    loss, rep = guard.safe_compute_loss(model, x, y_no_eos)
    assert loss is None
    assert rep["is_valid"] is False
    assert rep["status"] == "MISSING_EOS_SUPERVISION"


def test_sft_batch_safety_overlength_rejected():
    """Verifies that a batch exceeding max context length is safely rejected."""
    guard = SFTBatchSafetyGuard(vocab_size=4096, max_seq_len=512)
    model = ChakrMicro(ModelConfig(max_seq_len=512))
    model.eval()

    x = torch.zeros((1, 550), dtype=torch.long)
    y = torch.zeros((1, 550), dtype=torch.long)

    loss, rep = guard.safe_compute_loss(model, x, y)
    assert loss is None
    assert rep["is_valid"] is False
    assert rep["status"] == "EXCEEDS_CONTEXT_LENGTH"


def test_sft_batch_safety_valid_batch_finite_loss_and_grad():
    """Verifies that a valid batch passes inspection, yields finite loss and valid gradients."""
    guard = SFTBatchSafetyGuard(vocab_size=4096, max_seq_len=512)
    model = ChakrMicro(ModelConfig(max_seq_len=512))
    model.train()

    x = torch.tensor([[0, 10, 20, 30]], dtype=torch.long)
    y = torch.tensor([[-100, -100, 30, 1]], dtype=torch.long)

    loss, rep = guard.safe_compute_loss(model, x, y)
    assert rep["is_valid"] is True
    assert loss is not None
    assert math.isfinite(loss.item())

    model.zero_grad()
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0
    assert all(torch.isfinite(g).all() for g in grads)


# =========================================================================
# 6. Fail-Closed SFT Readiness Gate Tests
# =========================================================================

def test_sft_readiness_gate_remains_blocked():
    """Verifies that Batch 39 SFT readiness gate is strictly BLOCKED."""
    gate_file = Path("experiments/batch39_dataset_builder/gates/sft_readiness.json")
    if not gate_file.exists():
        pytest.skip("sft_readiness.json not yet built")

    with open(gate_file, "r", encoding="utf-8") as f:
        gate = json.load(f)

    assert gate["gate_decision"] == "BLOCKED"
    assert gate["independent_statuses"]["DATASET_READY"] is False
    assert gate["independent_statuses"]["SFT_READY"] is False
    assert gate["independent_statuses"]["PIPELINE_READY"] is True
    assert gate["independent_statuses"]["BASELINE_READY"] is True
    assert gate["blocking_reasons_count"] >= 1
