"""
Unit and Integration Test Suite for Batches 211–220:
Relational Knowledge Learning, Exact-Value Binding, and Generalization Validation.
Location: tests/test_batch211_220_relational_learning.py
"""
import json
from pathlib import Path
import pytest
import torch

from chakrview.brain.model import ChakrMicro
from chakrview.runtime.cognitive_router import CognitiveRouter, RouteType, ComponentOrigin
from chakrview.runtime.evidence_extractor import EvidenceExtractor
from chakrview.runtime.persistent_memory import PersistentSessionMemory
from chakrview.runtime.pointer_mechanism import ChakrPointerHead
from chakrview.runtime.retrieval_adapter import RetrievalAdapter
from chakrview.runtime.session_adapter import GroundedSessionAdapter
from chakrview.tokenizer.serialization import load_tokenizer_artifacts


@pytest.fixture(scope="module")
def tokenizer():
    tok_dir = Path("release/chakrview-v0.5.0/data/experiments/vocab_4096")
    if not tok_dir.exists():
        pytest.skip("Tokenizer directory not found")
    tok, _ = load_tokenizer_artifacts(tok_dir)
    return tok


@pytest.fixture(scope="module")
def canonical_model():
    ckpt_path = Path("experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt")
    if not ckpt_path.exists():
        pytest.skip("Canonical checkpoint not found")
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = ChakrMicro(ck["config"])
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    return model


def test_canonical_checkpoint_hash_and_params(canonical_model):
    """Verify canonical model has exactly 3,443,136 parameters and expected SHA-256."""
    import hashlib
    ckpt_path = Path("experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt")
    hasher = hashlib.sha256()
    with open(ckpt_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    assert hasher.hexdigest() == "96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee"

    param_count = sum(p.numel() for p in canonical_model.parameters())
    assert param_count == 3443136


def test_relational_pointer_checkpoint_load():
    """Verify trained relational pointer head loads cleanly without dimension mismatch."""
    p_ckpt = Path("experiments/batch211_220_relational_learning/checkpoints/relational_pointer_head_step1000.pt")
    assert p_ckpt.exists()
    head = ChakrPointerHead(d_model=192, vocab_size=4096)
    state = torch.load(p_ckpt, map_location="cpu", weights_only=False)
    head.load_state_dict(state)
    head.eval()
    assert head.query_proj.in_features == 192
    assert head.query_proj.out_features == 64


def test_session_adapter_sequence_length_guard(tokenizer, canonical_model):
    """Verify session adapter gracefully handles long prompts without exceeding 512 context limit."""
    class MockEngine:
        def __init__(self, m, t):
            self.model = m
            self.tokenizer = t

    import tempfile
    temp_mem = Path(tempfile.mkdtemp())
    memory = PersistentSessionMemory(storage_dir=temp_mem / "sessions")
    adapter = GroundedSessionAdapter(
        inference_engine=MockEngine(canonical_model, tokenizer),
        memory=memory,
        router=CognitiveRouter(enable_pointer_copy=False),
    )
    long_query = "word " * 600
    resp = adapter.execute_turn("test_overflow_session", long_query)
    assert resp.answer is not None
    assert resp.component_origin == ComponentOrigin.NEURAL_CORE


def test_adversarial_abstention_on_unanswerable():
    """Verify deterministic extractor abstains when queried relation is missing."""
    ctx = "Gold has atomic number 79 and density 19.3 g/cm3."
    query = "What is the orbital speed of Gold?"
    res = EvidenceExtractor.extract_slot(ctx, query)
    assert res.is_abstention
    assert "Not mentioned" in res.extracted_value


def test_reports_and_manifest_integrity_batch211_220():
    """Verify all 8 Batch 211-220 reports exist and contain non-zero byte size."""
    rep_dir = Path("experiments/batch211_220_relational_learning/reports")
    expected = [
        "initial_audit.md",
        "task_contract.md",
        "data_and_leakage_audit.md",
        "baseline_results.json",
        "training_results.json",
        "generalization_results.json",
        "runtime_regression_results.json",
        "failure_analysis.md",
        "milestone_summary.md",
    ]
    for r in expected:
        p = rep_dir / r
        assert p.exists() and p.stat().st_size > 0, f"Missing report: {r}"

    manifest_p = Path("experiments/batch211_220_relational_learning/experiment_manifest.json")
    assert manifest_p.exists() and manifest_p.stat().st_size > 0
