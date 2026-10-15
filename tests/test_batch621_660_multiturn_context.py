"""
Unit and Integration Tests for Batches 621-660:
Multi-Turn Context Binding, Neural Anaphora Resolution & Persistent Memory Grounding.
Location: tests/test_batch621_660_multiturn_context.py
"""
import tempfile
import pytest

from chakrview.runtime.cognitive_router import ComponentOrigin, RouteType
from chakrview.runtime.context_resolver import ContextReferenceResolver
from chakrview.runtime.persistent_memory import PersistentSessionMemory


@pytest.fixture
def temp_memory():
    with tempfile.TemporaryDirectory() as td:
        yield PersistentSessionMemory(td)


@pytest.fixture
def resolver(temp_memory):
    return ContextReferenceResolver(temp_memory)


def test_missing_context_abstention(resolver):
    """Anaphoric query with no conversation history must abstain and request clarification."""
    resp = resolver.execute_multi_turn_step("empty_sess", "Now multiply that result by 4")
    assert resp.component_origin == ComponentOrigin.FALLBACK_POLICY
    assert "Clarification required" in resp.answer
    assert resp.is_supported is False


def test_two_hop_arithmetic_followup(resolver, temp_memory):
    """Turn 1 calculation followed by Turn 2 anaphoric operation."""
    sess = "two_hop_calc"
    # Turn 1 setup
    temp_memory.append_turn(sess, "user", "Calculate 17 * 23")
    temp_memory.append_turn(sess, "assistant", "391", {"tool_output": "391"})

    # Turn 2 follow-up
    resp = resolver.execute_multi_turn_step(sess, "Now multiply that result by 4")
    assert resp.component_origin == ComponentOrigin.CALCULATOR_TOOL
    assert resp.route_decision.route == RouteType.CALCULATOR_TOOL
    assert resp.answer == "1564"
    assert resp.extracted_slots["expression"] == "391 * 4"


def test_three_hop_arithmetic_chain(resolver, temp_memory):
    """Turn 1 -> Turn 2 -> Turn 3 multi-turn execution chain."""
    sess = "three_hop_chain"
    # Turn 1: 25 * 4 = 100
    temp_memory.append_turn(sess, "user", "Calculate 25 * 4")
    temp_memory.append_turn(sess, "assistant", "100", {"tool_output": "100"})

    # Turn 2: + 50 -> 150
    resp2 = resolver.execute_multi_turn_step(sess, "Now add 50 to that result")
    assert resp2.answer == "150"

    # Turn 3: / 3 -> 50
    resp3 = resolver.execute_multi_turn_step(sess, "Now divide that by 3")
    assert resp3.answer == "50"


def test_text_transformation_anaphora(resolver, temp_memory):
    """Turn 1 mentions text, Turn 2 transforms it using anaphora."""
    sess = "text_anaphora"
    temp_memory.append_turn(sess, "user", "Remember this phrase: 'indigenous neural network'")
    temp_memory.append_turn(sess, "assistant", "Phrase recorded: indigenous neural network", {"context_text": "indigenous neural network"})

    resp = resolver.execute_multi_turn_step(sess, "Convert the previous text to uppercase")
    assert resp.answer == "INDIGENOUS NEURAL NETWORK"
    assert resp.extracted_slots["operation"] == "uppercase"


def test_ordinal_disambiguation(resolver, temp_memory):
    """User provides multiple numbers and specifically refers to 'second value'."""
    sess = "disambig_sess"
    temp_memory.append_turn(sess, "user", "The items cost 45 and 90 dollars respectively.")
    temp_memory.append_turn(sess, "assistant", "Recorded items costing 45 and 90.")

    resp = resolver.execute_multi_turn_step(sess, "Multiply the second value by 2")
    assert resp.answer == "180"


def test_session_boundary_isolation(resolver, temp_memory):
    """Session B cannot access or resolve anaphora from Session A."""
    sess_a = "user_alice_secret"
    sess_b = "user_bob_guest"

    # Alice calculates private value
    temp_memory.append_turn(sess_a, "user", "Calculate 50 * 50")
    temp_memory.append_turn(sess_a, "assistant", "2500", {"tool_output": "2500"})

    # Bob tries anaphoric query in clean session
    resp_b = resolver.execute_multi_turn_step(sess_b, "Now divide that result by 10")
    assert "Clarification required" in resp_b.answer
    assert "250" not in resp_b.answer


def test_intervening_turn_robustness(resolver, temp_memory):
    """Intervening non-numeric turns do not break binding to last computed output."""
    sess = "intervene_sess"
    temp_memory.append_turn(sess, "user", "Calculate 10 * 10")
    temp_memory.append_turn(sess, "assistant", "100", {"tool_output": "100"})
    temp_memory.append_turn(sess, "user", "Thank you, what a wonderful system.")
    temp_memory.append_turn(sess, "assistant", "You are welcome!")

    resp = resolver.execute_multi_turn_step(sess, "Now subtract 25 from that result")
    assert resp.answer == "75"
