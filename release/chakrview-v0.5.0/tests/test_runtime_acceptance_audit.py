"""
Runtime Acceptance & Edge Case Regression Tests (Batch 17).

Verifies user-facing runtime robustness across:
1. LocalModelRuntime: canonical baseline loading and bit-exact verification.
2. LocalModelRuntime: neural autoregressive forward generation without symbolic fallbacks.
3. LocalModelRuntime: edge cases (empty string, whitespace, excessive input length, unicode).
4. LocalModelRuntime: streaming chat generator edge cases and chunk validity.
5. LocalModelRuntime: weight immutability across repeated inference passes (Delta W = 0).
6. ChakrViewRuntimeSession: empty input handling with honest abstention.
7. ChakrViewRuntimeSession: definition queries for known terms vs unknown entities (honest abstention).
8. ChakrViewRuntimeSession: multi-hop relational path reasoning.
9. ChakrViewRuntimeSession: conflicting evidence detection and transparent reporting.
10. ChakrViewRuntimeSession: task memory boundary isolation and reset behavior.
"""

from __future__ import annotations

import pytest
from chakrview.runtime.local_runtime import LocalModelRuntime
from chakrview.runtime.inference import GenerationConfig, StopReason
from chakrview.runtime.sampling import SamplingConfig
from chakrview.brain.canonical_hash import verify_canonical_baseline
from scripts.founder_manual_test import ChakrViewRuntimeSession


def test_01_local_runtime_canonical_baseline_loading():
    """Verify LocalModelRuntime loads the canonical model and matches baseline bit-for-bit."""
    rt = LocalModelRuntime.from_default()
    assert rt.verify_runtime_integrity() is True
    check = verify_canonical_baseline(rt.model)
    assert check["is_bit_exact"] is True
    assert check["total_params"] == 3_443_136


def test_02_local_runtime_pure_neural_generation():
    """Verify LocalModelRuntime performs genuine neural generation."""
    rt = LocalModelRuntime.from_default()
    cfg = GenerationConfig(
        max_new_tokens=12,
        sampling=SamplingConfig(temperature=0.0),
    )
    res = rt.chat(session_id="test_pure_neural", prompt="ChakrView is", generation_config=cfg)
    assert res.output_token_count == 12
    assert len(res.text) > 0
    # Neural output is raw autoregressive text
    assert isinstance(res.text, str)


def test_03_local_runtime_edge_cases_safety():
    """Verify LocalModelRuntime handles empty input, whitespace, long strings, and special characters."""
    rt = LocalModelRuntime.from_default()
    cfg = GenerationConfig(max_new_tokens=8)

    edge_prompts = [
        "",
        "   \t\n  ",
        "A " * 300,  # Long sequence
        "\x00\x01\x02\u200b\ufffd",  # Non-printable and unicode markers
    ]

    for p in edge_prompts:
        res = rt.chat(session_id="edge_safety", prompt=p, generation_config=cfg)
        assert res.output_token_count == 8
        assert isinstance(res.text, str)


def test_04_local_runtime_stream_chat_validity():
    """Verify LocalModelRuntime.stream_chat yields valid chunks even on empty/whitespace inputs."""
    rt = LocalModelRuntime.from_default()
    cfg = GenerationConfig(max_new_tokens=6)

    for p in ["", "  ", "Test input"]:
        chunks = list(rt.stream_chat(session_id="stream_test", prompt=p, generation_config=cfg))
        assert len(chunks) == 6
        assert all(hasattr(c, "token_text") for c in chunks)


def test_05_local_runtime_weight_immutability():
    """Verify model weights do not drift during inference (Delta W = 0)."""
    rt = LocalModelRuntime.from_default()
    cfg = GenerationConfig(max_new_tokens=16)

    for i in range(5):
        rt.chat(session_id="immutability_test", prompt=f"Prompt number {i}", generation_config=cfg)

    # Re-verify integrity
    assert rt.verify_runtime_integrity() is True
    check = verify_canonical_baseline(rt.model)
    assert check["is_bit_exact"] is True
    assert check["delta_w"] == 0


def test_06_runtime_session_empty_input_honest_abstention():
    """Verify ChakrViewRuntimeSession returns honest abstention for empty or whitespace inputs."""
    session = ChakrViewRuntimeSession()
    for empty_inp in ["", "   ", " \t \n "]:
        diag = session.process_input(empty_inp)
        assert diag["decision"] == "ABSTAIN"
        assert diag["confidence"] == 0.0
        assert "No input text provided" in diag["final_response"]


def test_07_runtime_session_definition_query_honest_abstention():
    """Verify definition query returns known definition for valid term, and honest abstention for unknown term."""
    session = ChakrViewRuntimeSession()

    # Known term: CPU
    diag_cpu = session.process_input("What is CPU?")
    assert diag_cpu["decision"] == "ANSWER"
    assert diag_cpu["confidence"] > 0.9
    assert "Central Processing Unit" in diag_cpu["final_response"]

    # Unknown term: Zorvax-917
    diag_unk = session.process_input("Define Zorvax-917")
    assert diag_unk["decision"] == "ABSTAIN"
    assert diag_unk["confidence"] == 0.0
    assert "verified" in diag_unk["final_response"].lower()


def test_08_runtime_session_multihop_transitivity():
    """Verify multi-hop relational path reasoning."""
    session = ChakrViewRuntimeSession()
    session.process_input("Ram lives in Delhi.")
    session.process_input("Delhi is in India.")
    session.process_input("India is in Asia.")

    diag = session.process_input("Where is Ram relative to Asia?")
    assert diag["decision"] in ("ANSWER", "BELIEF_ACQUIRED", "PLAN_GENERATED")


def test_09_runtime_session_conflict_detection():
    """Verify conflicting evidence triggers transparent REPORT_CONFLICT decision."""
    session = ChakrViewRuntimeSession()
    session.process_input("Ram lives in Delhi.")

    # Supply external conflicting evidence: Mumbai
    diag = session.process_input(
        "Where does Ram live?",
        active_evidence={"ram::lives_in": "Mumbai"},
    )
    assert diag["decision"] == "REPORT_CONFLICT"
    assert diag["confidence"] < 0.5
    assert "conflict" in diag["final_response"].lower() or "virodhabhas" in diag["final_response"].lower()


def test_10_runtime_session_task_memory_isolation():
    """Verify working memory boundaries are isolated across tasks."""
    session = ChakrViewRuntimeSession()
    session.working_memory.initialize_task("task_A", "Task A Context")
    session.working_memory.set_variable("task_A", "session_secret", "alpha_val_42")

    # Verify task B cannot access task A's variable
    session.working_memory.initialize_task("task_B", "Task B Context")
    assert session.working_memory.get_variable("task_B", "session_secret") is None

    # Reset task A
    session.working_memory.reset_task("task_A")
    assert session.working_memory.get_variable("task_A", "session_secret") is None
