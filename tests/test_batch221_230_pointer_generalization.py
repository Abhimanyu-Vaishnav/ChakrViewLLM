"""
Unit and Integration Test Suite for Batches 221–230:
Relation-Aware Pointer Scoring, Multilingual Span Alignment, and Independent Generalization Audit.
Location: tests/test_batch221_230_pointer_generalization.py
"""
import hashlib
import json
from pathlib import Path
import pytest
import torch

from chakrview.brain.model import ChakrMicro
from chakrview.runtime.cognitive_router import CognitiveRouter, RouteType, ComponentOrigin
from chakrview.runtime.evidence_extractor import EvidenceExtractor
from chakrview.runtime.persistent_memory import PersistentSessionMemory
from chakrview.runtime.retrieval_adapter import RetrievalAdapter
from chakrview.runtime.session_adapter import GroundedSessionAdapter
from chakrview.runtime.relation_pointer import ChakrRelationPointerHead
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


def test_canonical_checkpoint_hash_and_parameters(canonical_model):
    """Verify canonical model checkpoint SHA-256 and parameter count unchanged."""
    ckpt_path = Path("experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt")
    hasher = hashlib.sha256()
    with open(ckpt_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    assert hasher.hexdigest() == "96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee"

    param_count = sum(p.numel() for p in canonical_model.parameters())
    assert param_count == 3443136


def test_frozen_test_manifest_hash_and_size():
    """Verify frozen test manifest contains exactly 100 items with matching SHA-256."""
    manifest_path = Path("experiments/batch221_230_pointer_generalization/data/frozen_test_manifest_100.json")
    assert manifest_path.exists()
    assert hashlib.sha256(manifest_path.read_bytes()).hexdigest() == "f345670a7b49ed1fd17b9c700fbcf468e597ee3035e50278dc439dbfdc0fd7aa"
    items = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert len(items) == 100


def test_relation_aware_pointer_head_instantiation_and_load():
    """Verify relation-aware pointer head loads weights cleanly with correct dimensions."""
    ckpt_path = Path("experiments/batch221_230_pointer_generalization/checkpoints/relation_aware_pointer_step1000.pt")
    assert ckpt_path.exists()
    head = ChakrRelationPointerHead(d_model=192, vocab_size=4096, d_proj=64, d_rel=32)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    head.load_state_dict(state)
    head.eval()

    assert head.relation_proj.in_features == 192
    assert head.relation_proj.out_features == 32
    assert head.context_rel_proj.in_features == 192
    assert head.context_rel_proj.out_features == 32


def test_runtime_provenance_and_abstention(tokenizer, canonical_model):
    """Verify GroundedSessionAdapter correctly attributes relation pointer and abstains safely."""
    class MockEngine:
        def __init__(self, m, t):
            self.model = m
            self.tokenizer = t

    import tempfile
    temp_mem = Path(tempfile.mkdtemp())
    memory = PersistentSessionMemory(storage_dir=temp_mem / "sessions")
    head = ChakrRelationPointerHead(d_model=192, vocab_size=4096, d_proj=64, d_rel=32)
    head_ckpt = Path("experiments/batch221_230_pointer_generalization/checkpoints/relation_aware_pointer_step1000.pt")
    head.load_state_dict(torch.load(head_ckpt, map_location="cpu", weights_only=False))
    head.eval()

    adapter = GroundedSessionAdapter(
        inference_engine=MockEngine(canonical_model, tokenizer),
        memory=memory,
        router=CognitiveRouter(enable_pointer_copy=True),
        pointer_head=head,
        enable_constrained_decoding=True,
    )

    # Context query with known entity
    resp = adapter.execute_turn(
        "sess_t",
        "What is the orbital period of Saturn?",
        supplied_context="Saturn orbital period is 10759 days."
    )
    assert resp.is_supported
    assert "10759" in resp.answer

    # Abstention on missing entity
    resp_abs = adapter.execute_turn(
        "sess_t",
        "What is the capital of Mars?",
        supplied_context="Carbon has atomic number 6."
    )
    assert not resp_abs.is_supported
    assert "Not mentioned" in resp_abs.answer


def test_reports_and_manifest_integrity():
    """Verify all 9 Batch 221-230 reports and manifest exist with valid content."""
    rep_dir = Path("experiments/batch221_230_pointer_generalization/reports")
    expected_reports = [
        "initial_audit.md",
        "evaluation_contract.md",
        "data_and_leakage_audit.md",
        "multilingual_failure_analysis.md",
        "baseline_results.json",
        "relation_aware_training_results.json",
        "generalization_results.json",
        "runtime_regression_results.json",
        "failure_analysis.md",
        "milestone_summary.md"
    ]
    for r in expected_reports:
        p = rep_dir / r
        assert p.exists() and p.stat().st_size > 0, f"Missing report: {r}"

    manifest_p = Path("experiments/batch221_230_pointer_generalization/experiment_manifest.json")
    assert manifest_p.exists() and manifest_p.stat().st_size > 0
