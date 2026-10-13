"""
Unit and Integration Test Suite for Batches 231–240:
Cross-Attention, Multi-Hop Grounding, and Independent Generalization Validation.
Location: tests/test_batch231_240_multihop_reasoning.py
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
from chakrview.runtime.cross_attention_pointer import ChakrCrossAttentionPointerHead
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


def test_frozen_multihop_manifest_integrity():
    """Verify primary frozen multi-hop manifest contains exactly 100 items with matching SHA-256."""
    manifest_path = Path("experiments/batch231_240_multihop_reasoning/data/frozen_multihop_manifest_100.json")
    assert manifest_path.exists()
    assert hashlib.sha256(manifest_path.read_bytes()).hexdigest() == "b70b38a880e307be77d5eaa1df68bd86b977ad67a2f9d13a79e4b1b7e0939016"
    items = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert len(items) == 100


def test_cross_attention_pointer_checkpoint_load():
    """Verify cross-attention pointer head loads weights cleanly with correct dimensions."""
    ckpt_path = Path("experiments/batch231_240_multihop_reasoning/checkpoints/cross_attention_pointer_step1000.pt")
    assert ckpt_path.exists()
    head = ChakrCrossAttentionPointerHead(d_model=192, vocab_size=4096, d_proj=64, n_heads=2)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    head.load_state_dict(state)
    head.eval()

    assert head.query_proj.in_features == 192
    assert head.cross_attn.embed_dim == 192


def test_multihop_runtime_provenance_and_missing_link(tokenizer, canonical_model):
    """Verify GroundedSessionAdapter correctly attributes multi-hop answers and abstains on missing links."""
    class MockEngine:
        def __init__(self, m, t):
            self.model = m
            self.tokenizer = t

    import tempfile
    temp_mem = Path(tempfile.mkdtemp())
    memory = PersistentSessionMemory(storage_dir=temp_mem / "sessions")
    head = ChakrCrossAttentionPointerHead(d_model=192, vocab_size=4096, d_proj=64, n_heads=2)
    head_ckpt = Path("experiments/batch231_240_multihop_reasoning/checkpoints/cross_attention_pointer_step1000.pt")
    head.load_state_dict(torch.load(head_ckpt, map_location="cpu", weights_only=False))
    head.eval()

    adapter = GroundedSessionAdapter(
        inference_engine=MockEngine(canonical_model, tokenizer),
        memory=memory,
        router=CognitiveRouter(enable_pointer_copy=True),
        pointer_head=head,
        enable_constrained_decoding=True,
    )

    # Multi-hop answerable query
    ctx_2hop = "Document A: The natural satellite of Earth is Moon.\nDocument B: Moon has orbital period 27.3 days."
    resp = adapter.execute_turn(
        "sess_hop_t",
        "What is the orbital period of the natural satellite of Earth?",
        supplied_context=ctx_2hop
    )
    assert resp.is_supported
    assert "27.3 days" in resp.answer

    # Missing link abstention
    ctx_missing = "Document A: Earth has natural satellite Moon."
    resp_abs = adapter.execute_turn(
        "sess_hop_t",
        "What is the orbital period of the natural satellite of Mars?",
        supplied_context=ctx_missing
    )
    assert not resp_abs.is_supported
    assert "Not mentioned" in resp_abs.answer


def test_reports_and_manifest_integrity():
    """Verify all 8 Batch 231-240 reports and manifest exist with valid non-zero content."""
    rep_dir = Path("experiments/batch231_240_multihop_reasoning/reports")
    expected_reports = [
        "initial_audit.md",
        "evaluation_contract.md",
        "data_and_leakage_audit.md",
        "baseline_results.json",
        "cross_attention_training_results.json",
        "multihop_results.json",
        "runtime_regression_results.json",
        "failure_analysis.md",
        "milestone_summary.md"
    ]
    for r in expected_reports:
        p = rep_dir / r
        assert p.exists() and p.stat().st_size > 0, f"Missing report: {r}"

    manifest_p = Path("experiments/batch231_240_multihop_reasoning/experiment_manifest.json")
    assert manifest_p.exists() and manifest_p.stat().st_size > 0
