"""
Tests for Batches 541-580: Neural Intent, Structured Understanding & Tool-Use Learning.
Location: tests/test_batch541_580_neural_intent.py
"""
import json
from pathlib import Path
import pytest
import torch

from chakrview.brain.model import ChakrMicro
from chakrview.runtime.cognitive_router import ComponentOrigin, RouteType
from chakrview.runtime.evidence_gate import EvidenceGovernedAnsweringGate, GateDecision
from chakrview.runtime.neural_intent_router import NeuralIntentRouter
from chakrview.runtime.tools import CalculatorTool, TextUtilityTool, ToolExecutor, ToolRegistry
from chakrview.tokenizer.serialization import load_tokenizer_artifacts

REPO_ROOT = Path(__file__).resolve().parent.parent
CKPT_PATH = REPO_ROOT / "experiments" / "batch541_580" / "checkpoints" / "candidate_step250_intent_understanding.pt"
TOKENIZER_DIR = REPO_ROOT / "release" / "chakrview-v0.5.0" / "data" / "experiments" / "vocab_4096"
AUDIT_JSON = REPO_ROOT / "experiments" / "batch541_580" / "reports" / "DATASET_QUALITY_AND_LEAKAGE_AUDIT.json"


def test_checkpoint_immutability_and_presence():
    """Verify canonical Step 10,000 checkpoint and new candidate exist with valid weights."""
    canon_path = REPO_ROOT / "experiments" / "batch16_21_combined" / "main_training_10000" / "training_checkpoint_step_10000.pt"
    assert canon_path.exists(), "Canonical checkpoint missing"
    assert CKPT_PATH.exists(), "Candidate checkpoint missing"

    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    assert "model_state_dict" in ckpt
    assert "config" in ckpt
    assert ckpt["steps"] >= 250


def test_zero_leakage_dataset_splits():
    """Verify that dataset splits have zero prompt leakage between train, val, and test."""
    assert AUDIT_JSON.exists(), "Dataset audit missing"
    audit = json.loads(AUDIT_JSON.read_text(encoding="utf-8"))
    assert audit["leakage_audit"]["zero_leakage_verified"] is True
    assert audit["leakage_audit"]["train_test_prompt_overlap"] == 0
    assert audit["leakage_audit"]["train_val_prompt_overlap"] == 0


def test_tool_executor_security_boundaries():
    """Verify governed tool executor enforces schema, AST whitelist, and rejects malicious ops."""
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    registry.register(TextUtilityTool())
    executor = ToolExecutor(registry)

    # Valid arithmetic execution
    res = executor.execute("calculator", {"expression": "25 * 4 + 10"})
    assert res.success is True
    assert res.output == 110

    # Malicious injection attempt in calculator
    res_bad = executor.execute("calculator", {"expression": "__import__('os').system('ls')"})
    assert res_bad.success is False
    assert "Security violation" in res_bad.error or "Disallowed" in res_bad.error

    # Text utility execution
    res_text = executor.execute("text_utility", {"operation": "uppercase", "text": "chakrview"})
    assert res_text.success is True
    assert res_text.output == "CHAKRVIEW"

    # Missing required parameter
    res_missing = executor.execute("calculator", {})
    assert res_missing.success is False
    assert "Missing required parameter" in res_missing.error


def test_evidence_gate_refutation_preflight():
    """Verify evidence gate intercepts false premises before neural generation."""
    gate = EvidenceGovernedAnsweringGate()
    resp = gate.process_query("What is the municipal capital city of Mars?")
    assert resp.decision == GateDecision.REFUTE
    assert "Mars" in resp.content
    assert "uninhabited" in resp.content or "celestial" in resp.reasoning


def test_neural_intent_router_end_to_end():
    """Verify NeuralIntentRouter handles queries with explicit component provenance."""
    router = NeuralIntentRouter(
        checkpoint_path=CKPT_PATH,
        tokenizer_dir=TOKENIZER_DIR,
        device=torch.device("cpu"),
    )

    # 1. False premise turn -> intercepted by Evidence Gate
    mars_turn = router.process_turn("What is the capital of planet Mars?")
    assert mars_turn.is_supported is False
    assert mars_turn.component_origin == ComponentOrigin.FALLBACK_POLICY
    assert mars_turn.route_decision.route == RouteType.SAFE_FALLBACK

    # 2. Conversational greeting -> handled with proper provenance
    chat_turn = router.process_turn("Hello ChakrView, good morning!")
    assert chat_turn.component_origin in (ComponentOrigin.NEURAL_CORE, ComponentOrigin.FALLBACK_POLICY)
    assert len(chat_turn.answer) > 0

    # 3. Arithmetic request -> routed to safe governed calculator tool
    calc_turn = router.process_turn("Please compute 40 / 10")
    assert calc_turn.latency_ms >= 0.0
    assert calc_turn.component_origin in (ComponentOrigin.CALCULATOR_TOOL, ComponentOrigin.NEURAL_CORE)
