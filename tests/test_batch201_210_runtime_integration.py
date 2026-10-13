"""
Test Suite for Batches 201–210: Interactive Runtime Integration & Grounded Conversation.
Location: tests/test_batch201_210_runtime_integration.py
"""
import json
from pathlib import Path
import pytest
import torch

from chakrview.runtime.cognitive_router import CognitiveRouter, RouteType, ComponentOrigin
from chakrview.runtime.evidence_extractor import EvidenceExtractor
from chakrview.runtime.session_adapter import GroundedSessionAdapter
from chakrview.runtime.pointer_mechanism import ChakrPointerHead
from chakrview.tokenizer.serialization import load_tokenizer_artifacts


@pytest.fixture(scope="module")
def tokenizer():
    tok_dir = Path("release/chakrview-v0.5.0/data/experiments/vocab_4096")
    if not tok_dir.exists():
        pytest.skip("Tokenizer directory not found")
    tok, _ = load_tokenizer_artifacts(tok_dir)
    return tok


def test_component_origin_enum():
    """Verify that all intended component origins exist."""
    assert ComponentOrigin.CONSTRAINED_TRIE.value == "constrained_trie"
    assert ComponentOrigin.NEURAL_POINTER.value == "neural_pointer"
    assert ComponentOrigin.DETERMINISTIC_EXTRACTOR.value == "deterministic_extractor"


def test_evidence_extractor_temperature_conflict():
    """Verify conflicting temperature reports trigger has_conflicts."""
    ctx = "The boiling point of nitrogen is -196 C. Another text says the boiling point of nitrogen is -190 C."
    res = EvidenceExtractor.extract_slot(ctx, "What is the boiling point of nitrogen?")
    assert res.has_conflicts
    assert "-196 C" in res.conflicting_values
    assert "-190 C" in res.conflicting_values


def test_session_adapter_entity_abstention():
    """Verify session adapter abstains when queried entity is completely absent from context."""
    class MockEngine:
        def __init__(self, tok):
            self.tokenizer = tok
            self.model = torch.nn.Linear(10, 10)

    # Use dummy tokenizer mock
    class DummyTok:
        def encode(self, s, **kwargs): return [1, 2]
        def decode(self, ids, **kwargs): return "dummy"

    adapter = GroundedSessionAdapter(
        inference_engine=MockEngine(DummyTok()),
        router=CognitiveRouter(enable_pointer_copy=True),
    )
    resp = adapter.execute_turn(
        session_id="test_sess",
        user_input="What is the capital of Mars?",
        supplied_context="Carbon has atomic number 6."
    )
    assert not resp.is_supported
    assert "Not mentioned in context" in resp.answer


def test_manifest_and_reports_integrity_batch201_210():
    """Verify all Batch 201-210 artifacts exist and are non-empty."""
    base_dir = Path("experiments/batch201_210_runtime_integration/reports")
    expected_reports = [
        "initial_audit.md",
        "runtime_integration.md",
        "data_and_leakage_audit.md",
        "pointer_arms_training_results.json",
        "constrained_decoding_results.json",
        "interactive_runtime_benchmark.json",
        "recovery_and_resource_tests.json",
        "evaluation_results.json",
        "failure_analysis.md",
        "milestone_summary.md",
    ]
    for r in expected_reports:
        p = base_dir / r
        assert p.exists() and p.stat().st_size > 0, f"Missing or empty report: {p}"

    manifest_p = Path("experiments/batch201_210_runtime_integration/experiment_manifest.json")
    assert manifest_p.exists() and manifest_p.stat().st_size > 0
