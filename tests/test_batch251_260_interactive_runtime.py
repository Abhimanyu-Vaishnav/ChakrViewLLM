"""
Unit and Integration Test Suite for Batches 251–260:
End-to-End Interactive Runtime Validation, Three-Hop Grounding Improvements, and Safe Hardware Acceleration.
Location: tests/test_batch251_260_interactive_runtime.py
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
from chakrview.cognition.language_realizer import (
    ConversationalContextTracker,
    ConversationalTurn,
    DiscourseEntity,
    LanguageTag,
    ConversationalIntent,
)
from scripts.founder_manual_test import ChakrViewRuntimeSession

CANONICAL_CHECKPOINT = Path("experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt")


def test_canonical_invariants_unmodified():
    """Verify canonical model checkpoint SHA-256 and parameter count are strictly immutable."""
    assert CANONICAL_CHECKPOINT.exists(), "Canonical checkpoint must exist"
    hasher = hashlib.sha256()
    with open(CANONICAL_CHECKPOINT, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    assert hasher.hexdigest() == "96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee"

    ck = torch.load(CANONICAL_CHECKPOINT, map_location="cpu", weights_only=False)
    model = ChakrMicro(ck["config"])
    model.load_state_dict(ck["model_state_dict"])
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params == 3_443_136, f"Expected 3,443,136 params, got {total_params}"


def test_founder_console_e2e_dispatch():
    """Verify real founder console entrypoint dispatches calculator, definitions, code, and memory."""
    session = ChakrViewRuntimeSession()

    # 1. Arithmetic
    resp_math = session.process_input("What is 25 + 37?")
    ans_math = (resp_math.get("final_response") or resp_math.get("natural_response") or "").strip()
    assert "62" in ans_math
    assert resp_math.get("intent") == "CALCULATOR_TOOL"

    # 2. Pi
    resp_pi = session.process_input("What is the value of pi?")
    ans_pi = (resp_pi.get("final_response") or resp_pi.get("natural_response") or "").strip()
    assert "3.14159" in ans_pi

    # 3. CPU
    resp_cpu = session.process_input("Explain CPU in simple words.")
    ans_cpu = (resp_cpu.get("final_response") or resp_cpu.get("natural_response") or "").strip()
    assert "CPU" in ans_cpu and ("processor" in ans_cpu or "brain" in ans_cpu)

    # 4. Code generation
    resp_code = session.process_input("Write a Python program to add two numbers entered by the user.")
    ans_code = (resp_code.get("final_response") or resp_code.get("natural_response") or "").strip()
    clean_code = ans_code.replace("```python", "").replace("```", "").strip()
    assert ast.parse(clean_code), "Code must be valid Python AST"

    # 5. False premise
    resp_mars = session.process_input("What is the capital of Mars?")
    ans_mars = (resp_mars.get("final_response") or resp_mars.get("natural_response") or "").strip()
    assert "do not have a capital" in ans_mars.lower() or "does not have a capital" in ans_mars.lower() or "uninhabited" in ans_mars.lower()

    # 6. Hindi photosynthesis
    resp_hi = session.process_input("Mujhe simple Hindi mein photosynthesis samjhao.")
    ans_hi = (resp_hi.get("final_response") or resp_hi.get("natural_response") or "").strip()
    assert "प्रकाश संश्लेषण" in ans_hi

    # 7. Coreference follow-up
    resp_fol = session.process_input("and What about ALU")
    ans_fol = (resp_fol.get("final_response") or resp_fol.get("natural_response") or "").strip()
    assert "ALU" in ans_fol and "Arithmetic Logic Unit" in ans_fol

    # 8. User identity store and recall
    resp_store = session.process_input("My name is Abhimanyu Vaishnav.")
    ans_store = (resp_store.get("final_response") or resp_store.get("natural_response") or "").strip()
    assert "Abhimanyu" in ans_store

    resp_recall = session.process_input("What is my name?")
    ans_recall = (resp_recall.get("final_response") or resp_recall.get("natural_response") or "").strip()
    assert "Abhimanyu" in ans_recall


def test_discourse_entity_stack_and_coreference():
    """Verify bounded discourse entity stack tracks recent subjects across multi-turn conversation."""
    tracker = ConversationalContextTracker(max_turns=10)

    # Push entities across turns
    tracker.push_entity("Neptune", "planet", provenance="turn_1")
    tracker.push_entity("Triton", "moon", provenance="turn_2")
    tracker.push_entity("CPU", "component", provenance="turn_3")

    # Most recent entity should be top of stack
    assert tracker.get_latest_subject() == "CPU"

    # Resolving pronoun "it" should return CPU
    resolved = tracker.resolve_reference("it")
    assert resolved == "CPU"

    # Re-mentioning Neptune re-activates it to the top of stack
    tracker.push_entity("Neptune", "planet", provenance="turn_4")
    assert tracker.get_latest_subject() == "Neptune"
    assert tracker.resolve_reference("it") == "Neptune"

    # Verify stack size is strictly bounded to 10
    for i in range(15):
        tracker.push_entity(f"Entity_{i}", "test", provenance=f"turn_{5+i}")
    assert len(tracker.entity_stack) == 10
    assert tracker.get_latest_subject() == "Entity_14"


def test_session_isolation_and_persistence():
    """Verify separate sessions have isolated user profiles and persistent storage survives reload."""
    with tempfile.TemporaryDirectory() as td:
        mem_dir = Path(td)
        mem = PersistentSessionMemory(mem_dir)
        adapter = GroundedSessionAdapter(inference_engine=None, memory=mem)

        # Session A sets name to Alice
        adapter.execute_turn("session_A", "My name is Alice")
        resp_a = adapter.execute_turn("session_A", "What is my name?")
        assert "Alice" in resp_a.answer

        # Session B sets name to Bob
        adapter.execute_turn("session_B", "My name is Bob")
        resp_b = adapter.execute_turn("session_B", "What is my name?")
        assert "Bob" in resp_b.answer

        # Verify Session A is still Alice (isolation)
        resp_a_check = adapter.execute_turn("session_A", "What is my name?")
        assert "Alice" in resp_a_check.answer
        assert "Bob" not in resp_a_check.answer

        # Reload memory in a fresh adapter (restart simulation)
        fresh_mem = PersistentSessionMemory(mem_dir)
        fresh_adapter = GroundedSessionAdapter(inference_engine=None, memory=fresh_mem)
        resp_reload_a = fresh_adapter.execute_turn("session_A", "What is my name?")
        assert "Alice" in resp_reload_a.answer
        resp_reload_b = fresh_adapter.execute_turn("session_B", "What is my name?")
        assert "Bob" in resp_reload_b.answer


def test_threehop_intermediate_supervision_checkpoint():
    """Verify Batch 256 intermediate-supervised 3-hop pointer checkpoint exists and loads correctly."""
    ckpt_path = Path("experiments/batch251_260_interactive_runtime/checkpoints/threehop_supervised_pointer.pt")
    assert ckpt_path.exists(), f"Supervised pointer checkpoint not found at {ckpt_path}"
    head = ChakrCrossAttentionPointerHead(d_model=192, vocab_size=4096, d_proj=64, n_heads=2)
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    head.load_state_dict(state_dict)
    head.eval()

    # Verify forward pass shape contract
    B, T, D = 1, 32, 192
    hidden = torch.randn(B, T, D)
    ids = torch.randint(0, 4096, (B, T))
    logits = torch.randn(B, T, 4096)
    ctx_mask = torch.ones(B, T, dtype=torch.bool)
    q_mask = torch.zeros(B, T, dtype=torch.bool)
    q_mask[:, :8] = True

    probs, attn, gate = head(hidden, ids, logits, ctx_mask, query_mask=q_mask)
    assert probs.shape == (B, T, 4096)
    assert attn.shape == (B, T, T)
    assert gate.shape == (B, T, 1)
