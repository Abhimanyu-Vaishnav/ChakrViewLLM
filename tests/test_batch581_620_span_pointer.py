"""
Regression and Integration Tests for Batches 581-620.
Location: tests/test_batch581_620_span_pointer.py
"""
import json
from pathlib import Path
import pytest
import torch

from chakrview.brain.model import ChakrMicro
from chakrview.brain.span_pointer_model import ChakrSpanPointerModel
from chakrview.runtime.cognitive_router import ComponentOrigin, RouteType
from chakrview.runtime.evidence_gate import EvidenceGovernedAnsweringGate, GateDecision
from chakrview.runtime.neural_intent_router import NeuralIntentRouter
from chakrview.runtime.tools import CalculatorTool, TextUtilityTool, ToolExecutor, ToolRegistry
from chakrview.tokenizer.serialization import load_tokenizer_artifacts

REPO_ROOT = Path(__file__).resolve().parent.parent
CKPT_SPAN = REPO_ROOT / "experiments" / "batch581_620" / "checkpoints" / "candidate_step250_span_pointer.pt"
TOKENIZER_DIR = REPO_ROOT / "release" / "chakrview-v0.5.0" / "data" / "experiments" / "vocab_4096"
AUDIT_JSON = REPO_ROOT / "experiments" / "batch581_620" / "reports" / "SPAN_DATASET_QUALITY_AND_LEAKAGE_AUDIT.json"


def test_span_pointer_checkpoint_presence_and_weights():
    """Verify that ChakrSpanPointerModel checkpoint is present, loadable, and structurally sound."""
    assert CKPT_SPAN.exists(), "candidate_step250_span_pointer.pt missing"
    ckpt = torch.load(CKPT_SPAN, map_location="cpu", weights_only=False)
    assert "model_state_dict" in ckpt
    assert "base_config" in ckpt
    assert ckpt["steps"] >= 250


def test_span_dataset_leakage_and_audit():
    """Verify zero prompt overlap across train, val, and test splits."""
    assert AUDIT_JSON.exists(), "Audit file missing"
    audit = json.loads(AUDIT_JSON.read_text(encoding="utf-8"))
    assert audit["leakage_audit"]["zero_leakage_verified"] is True
    assert audit["leakage_audit"]["train_test_overlap"] == 0


def test_neural_pointer_exact_argument_extraction():
    """Verify neural pointer extracts unseen arithmetic expression and passes to CalculatorTool."""
    router = NeuralIntentRouter(
        checkpoint_path=CKPT_SPAN,
        tokenizer_dir=TOKENIZER_DIR,
        device=torch.device("cpu"),
    )

    resp = router.process_turn("Please compute 36 / 12")
    assert resp.is_supported is True
    assert resp.component_origin == ComponentOrigin.CALCULATOR_TOOL
    assert resp.answer == "3"

    resp2 = router.process_turn("Can you compute 94 - 62?")
    assert resp2.is_supported is True
    assert resp2.component_origin == ComponentOrigin.CALCULATOR_TOOL
    assert resp2.answer == "32"


def test_text_utility_span_extraction():
    """Verify text utility extracts quoted string and performs uppercase transform."""
    router = NeuralIntentRouter(
        checkpoint_path=CKPT_SPAN,
        tokenizer_dir=TOKENIZER_DIR,
        device=torch.device("cpu"),
    )

    resp = router.process_turn("Convert 'indigenous architecture' to uppercase")
    assert resp.is_supported is True
    assert "INDIGENOUS ARCHITECTURE" in resp.answer


def test_clarification_and_security_safeguards():
    """Verify incomplete requests require clarification and false premises are refuted."""
    router = NeuralIntentRouter(
        checkpoint_path=CKPT_SPAN,
        tokenizer_dir=TOKENIZER_DIR,
        device=torch.device("cpu"),
    )

    # Incomplete request -> Clarification required
    resp_clar = router.process_turn("Multiply 75 by")
    assert "Clarification required" in resp_clar.answer
    assert resp_clar.route_decision.route == RouteType.DIRECT_CHAT

    # False premise -> Refutation by evidence gate
    resp_mars = router.process_turn("What is the capital city of planet Mars?")
    assert resp_mars.is_supported is False
    assert resp_mars.component_origin == ComponentOrigin.FALLBACK_POLICY
    assert "Mars" in resp_mars.answer
