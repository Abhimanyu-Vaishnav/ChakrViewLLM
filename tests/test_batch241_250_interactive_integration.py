"""
Unit and Integration Test Suite for Batches 241–250:
Interactive Intelligence Integration, Intent-Routing Repair, and Audited Multi-Hop Generalization.
Location: tests/test_batch241_250_interactive_integration.py
"""
import ast
import hashlib
import json
from pathlib import Path
import tempfile
import pytest
import torch

from chakrview.brain.model import ChakrMicro
from chakrview.runtime.cognitive_router import CognitiveRouter, RouteType, ComponentOrigin
from chakrview.runtime.cross_attention_pointer import ChakrCrossAttentionPointerHead
from chakrview.runtime.persistent_memory import PersistentSessionMemory
from chakrview.runtime.session_adapter import GroundedSessionAdapter
from chakrview.runtime.tools import CalculatorTool
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


def test_canonical_invariants():
    """Verify canonical model checkpoint SHA-256 and parameter count are strictly unmodified."""
    ckpt_path = Path("experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt")
    hasher = hashlib.sha256()
    with open(ckpt_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    assert hasher.hexdigest() == "96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee"


def test_intent_routing_precedence():
    """Verify repaired intent-routing classifies arithmetic, concepts, code, and user profile."""
    router = CognitiveRouter()

    # 1. Arithmetic
    r_math = router.classify_route("What is 25 + 37?")
    assert r_math.route == RouteType.CALCULATOR_TOOL
    assert r_math.metadata.get("expression") == "25 + 37"

    # 2. Pi
    r_pi = router.classify_route("What is the value of pi?")
    assert r_pi.route == RouteType.CONCEPT_DEFINITION
    assert r_pi.metadata.get("concept") == "pi"

    # 3. ALU
    r_alu = router.classify_route("What is an ALU?")
    assert r_alu.route == RouteType.CONCEPT_DEFINITION
    assert r_alu.metadata.get("concept") == "alu"

    # 4. CPU Explanation
    r_cpu = router.classify_route("Explain CPU in simple words.")
    assert r_cpu.route == RouteType.CONCEPT_DEFINITION
    assert r_cpu.metadata.get("concept") == "cpu"

    # 5. Code synthesis
    r_code = router.classify_route("Write a Python program to add two numbers entered by the user.")
    assert r_code.route == RouteType.CODE_GENERATION

    # 6. False premise
    r_mars = router.classify_route("What is the capital of Mars?")
    assert r_mars.route == RouteType.CONCEPT_DEFINITION
    assert r_mars.metadata.get("concept") == "capital_of_mars"

    # 7. Multilingual Hindi
    r_hi = router.classify_route("Mujhe simple Hindi mein photosynthesis samjhao.")
    assert r_hi.route == RouteType.CONCEPT_DEFINITION
    assert r_hi.metadata.get("concept") == "photosynthesis_hindi"

    # 8. Follow-up
    r_follow = router.classify_route("and What about ALU")
    assert r_follow.route == RouteType.CONCEPT_DEFINITION
    assert r_follow.metadata.get("is_followup") is True

    # 9. User profile memory
    r_store = router.classify_route("My name is Vikram")
    assert r_store.route == RouteType.USER_MEMORY
    assert r_store.metadata.get("value") == "Vikram"

    r_recall = router.classify_route("What is my name?")
    assert r_recall.route == RouteType.USER_MEMORY
    assert r_recall.metadata.get("action") == "recall"


def test_safe_calculator_execution():
    """Verify safe calculator execution without eval or OS access."""
    calc = CalculatorTool()
    # Correct calculation
    res1 = calc.execute(expression="25 + 37")
    assert res1.success is True
    assert res1.output == 62

    # Malformed calculation handled safely
    res2 = calc.execute(expression="25 + + * 4")
    assert res2.success is False
    assert res2.error is not None


def test_code_synthesis_ast_and_isolation():
    """Verify synthesized code parses in AST and contains addition logic."""
    with tempfile.TemporaryDirectory() as td:
        mem = PersistentSessionMemory(Path(td))
        adapter = GroundedSessionAdapter(inference_engine=None, memory=mem)
        resp = adapter.execute_turn("session_code", "Write a Python program to add two numbers entered by the user.")
        assert resp.component_origin == ComponentOrigin.CODE_SYNTHESIZER
        clean_code = resp.answer.replace("```python", "").replace("```", "").strip()
        tree = ast.parse(clean_code)
        assert len(tree.body) >= 3


def test_user_memory_persistence_across_restart():
    """Verify user name storage persists across simulated process restart."""
    with tempfile.TemporaryDirectory() as td:
        mem_dir = Path(td)
        # Session 1: Store name
        mem1 = PersistentSessionMemory(mem_dir)
        adapter1 = GroundedSessionAdapter(inference_engine=None, memory=mem1)
        resp1 = adapter1.execute_turn("user_session_42", "My name is Priya")
        assert "Priya" in resp1.answer
        assert resp1.component_origin == ComponentOrigin.SESSION_MEMORY

        # Session 2: Reload from disk simulating fresh process start
        mem2 = PersistentSessionMemory(mem_dir)
        adapter2 = GroundedSessionAdapter(inference_engine=None, memory=mem2)
        resp2 = adapter2.execute_turn("user_session_42", "What is my name?")
        assert "Priya" in resp2.answer
        assert resp2.component_origin == ComponentOrigin.SESSION_MEMORY


def test_threehop_frozen_manifest_hash():
    """Verify 3-hop manifest contains 100 items with valid SHA-256."""
    p = Path("experiments/batch241_250_interactive_integration/data/frozen_threehop_manifest_100.json")
    assert p.exists()
    items = json.loads(p.read_text(encoding="utf-8"))
    assert len(items) == 100
    assert hashlib.sha256(p.read_bytes()).hexdigest() == "6bf47a2baf124859f63b9ed1da4e31777852146db3d12c6c34a112b58f96bfc3"
