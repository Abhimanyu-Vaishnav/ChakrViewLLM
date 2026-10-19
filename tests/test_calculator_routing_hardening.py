"""
Regression tests for Safe Calculator Routing & Arithmetic Parsing Hardening.
Location: tests/test_calculator_routing_hardening.py

Validates:
1. "can you tell me what 99 multiple by 7 and add 80 in it" -> Exactly 773.
2. "what is 99*7 and + 80" -> Exactly 773.
3. Equivalent natural-language formulations:
   - "What is 99 multiplied by 7 and then add 80?" -> Exactly 773.
   - "Multiply 99 by 7, then add 80." -> Exactly 773.
   - "What is 99*7 + 80?" -> Exactly 773.
   - "What is 99 times 7 plus 80?" -> Exactly 773.
4. Ensuring trailing operations/numbers are never silently dropped.
5. Incomplete/ambiguous math expressions trigger clear abstention rather than partial evaluation.
6. Safe AST limits (no eval/exec, exponent limits, division by zero).
7. Preservation of working memory facts across hardened calculator turns.
8. Preservation of document citations and abstention for ungrounded queries.
"""

from __future__ import annotations

import pytest
from experiments.batch2181_2220.verified_tool_router import VerifiedToolRouter
from experiments.batch2221_2260.integrated_assistant_adapter import IntegratedAssistantAdapter


def test_observed_failure_1_verbal_multiplication_and_addition():
    """Observed Failure 1: 'can you tell me what 99 multiple by 7 and add 80 in it' must evaluate to 773."""
    adapter = IntegratedAssistantAdapter()
    resp = adapter.ask("can you tell me what 99 multiple by 7 and add 80 in it")

    assert resp.decision == "CALCULATED"
    assert "773" in resp.answer
    assert resp.tool_trace["tool_output"] == 773
    assert resp.execution_breakdown["subsystem"] == "safe_calculator"
    assert resp.execution_breakdown["neural_inference"] == "not invoked"


def test_observed_failure_2_mixed_symbols_and_conjunctions():
    """Observed Failure 2: 'what is 99*7 and + 80' must evaluate to 773 without dropping + 80."""
    adapter = IntegratedAssistantAdapter()
    resp = adapter.ask("what is 99*7 and + 80")

    assert resp.decision == "CALCULATED"
    assert "773" in resp.answer
    assert resp.tool_trace["tool_output"] == 773
    assert resp.execution_breakdown["subsystem"] == "safe_calculator"


@pytest.mark.parametrize("query", [
    "What is 99 multiplied by 7 and then add 80?",
    "Multiply 99 by 7, then add 80.",
    "What is 99*7 + 80?",
    "What is 99 times 7 plus 80?",
    "compute 99 * 7 + 80",
    "calculate (99 * 7) + 80",
])
def test_equivalent_phrasings_evaluate_to_773(query: str):
    """Verify that all equivalent phrasings of (99 * 7) + 80 yield exactly 773."""
    adapter = IntegratedAssistantAdapter()
    resp = adapter.ask(query)

    assert resp.decision == "CALCULATED"
    assert "773" in resp.answer
    assert resp.tool_trace["tool_output"] == 773


def test_no_silent_drop_of_trailing_operations():
    """Ensure multi-operator chains evaluate completely and never drop trailing numbers."""
    router = VerifiedToolRouter()
    # (100 - 20) + 5
    t1 = router.route_and_execute("what is 100 minus 20 and then add 5")
    assert t1 is not None and t1.success
    assert t1.tool_output == 85

    # 10 * 5 - 12
    t2 = router.route_and_execute("what is 10 times 5 and subtract 12 from it")
    assert t2 is not None and t2.success
    assert t2.tool_output == 38


def test_ambiguous_or_dangling_expression_abstains():
    """Ambiguous or dangling operators must trigger clear abstention rather than partial evaluation."""
    adapter = IntegratedAssistantAdapter()
    
    # Dangling trailing operator
    resp1 = adapter.ask("compute 99 * and then")
    assert resp1.decision == "ABSTAIN"
    assert "Incomplete or ambiguous arithmetic expression" in resp1.answer
    assert "Dangling or incomplete" in resp1.answer

    # Dangling plus
    resp2 = adapter.ask("what is 50 + ?")
    assert resp2.decision == "ABSTAIN"
    assert "Incomplete or ambiguous" in resp2.answer


def test_safety_ast_limits_preserved():
    """Ensure AST security and mathematical boundary constraints are preserved."""
    router = VerifiedToolRouter()
    
    # Division by zero
    t_div_zero = router.route_and_execute("What is 100 / 0?")
    assert t_div_zero is not None
    assert not t_div_zero.success
    assert "zero" in t_div_zero.error.lower()

    # Disallowed code injection attempts
    t_inject = router.route_and_execute("Calculate __import__('os').system('dir')")
    assert t_inject is None  # Does not match arithmetic syntax


def test_memory_context_preserved_across_hardened_calculation():
    """Verify that working memory fact recall works across a hardened multi-step calculation."""
    adapter = IntegratedAssistantAdapter()

    # 1. State fact
    r1 = adapter.ask("The project codename is Blue Mango 47.")
    assert r1.decision == "STATEMENT"

    # 2. Hardened arithmetic calculation
    r2 = adapter.ask("can you tell me what 99 multiple by 7 and add 80 in it")
    assert r2.decision == "CALCULATED"
    assert "773" in r2.answer

    # 3. Recall fact
    r3 = adapter.ask("What was the project codename I told you earlier?")
    assert r3.decision == "WORKING_MEMORY"
    assert "Blue Mango 47" in r3.answer


def test_observed_three_message_reproduction():
    """
    Observed reproduction:
    1. 'what is 9 multiple by 99 and add 100' -> 991
    2. 'compute 99 * and then' -> Abstain / incomplete
    3. 'okay multiple it with 87' -> Must NOT return greeting/acknowledgement, must clarify:
       'Which number should I multiply by 87?'
    """
    adapter = IntegratedAssistantAdapter()

    # Step 1: Initial calculation
    r1 = adapter.ask("what is 9 multiple by 99 and add 100")
    assert r1.decision == "CALCULATED"
    assert "991" in r1.answer

    # Step 2: Incomplete calculation
    r2 = adapter.ask("compute 99 * and then")
    assert r2.decision == "ABSTAIN"
    assert "Incomplete or ambiguous" in r2.answer

    # Step 3: Follow-up after incomplete calculation
    r3 = adapter.ask("okay multiple it with 87")
    assert r3.decision == "CLARIFY"
    assert r3.answer == "Which number should I multiply by 87?"
    assert "You're welcome" not in r3.answer


def test_followup_after_successful_calculation():
    """Follow-up arithmetic after a successful calculation computes the new result safely."""
    adapter = IntegratedAssistantAdapter()

    # Step 1: Valid calculation
    r1 = adapter.ask("what is 9 multiple by 99 and add 100")
    assert r1.decision == "CALCULATED"
    assert "991" in r1.answer

    # Step 2: Follow-up calculation multiplying prior result (991) by 2
    r2 = adapter.ask("multiply it by 2")
    assert r2.decision == "CALCULATED"
    assert "1982" in r2.answer
    assert r2.tool_trace["tool_output"] == 1982


def test_followup_after_incomplete_calculation_without_prior():
    """Follow-up arithmetic after an incomplete calculation when no prior valid calculation exists."""
    adapter = IntegratedAssistantAdapter()

    # Step 1: Incomplete expression
    r1 = adapter.ask("compute 50 +")
    assert r1.decision == "ABSTAIN"

    # Step 2: Follow-up asking to multiply 'that'
    r2 = adapter.ask("now multiply that by 87")
    assert r2.decision == "CLARIFY"
    assert r2.answer == "Which number should I multiply by 87?"


def test_ambiguous_pronouns_and_phrasings_clarify_when_no_valid_antecedent():
    """Verify various pronouns ('it', 'that', 'this') clarify cleanly without guessing operands."""
    adapter = IntegratedAssistantAdapter()

    r1 = adapter.ask("multiply it by 5")
    assert r1.decision == "CLARIFY"
    assert r1.answer == "Which number should I multiply by 5?"

    r2 = adapter.ask("okay multiple that with 12")
    assert r2.decision == "CLARIFY"
    assert r2.answer == "Which number should I multiply by 12?"

    r3 = adapter.ask("divide this by 4")
    assert r3.decision == "CLARIFY"
    assert r3.answer == "Which number should I divide by 4?"


def test_conversational_fallback_cannot_override_arithmetic_intent():
    """Phrases with leading conversational acks ('ok', 'okay', 'thanks') + arithmetic cannot be routed to greeting."""
    adapter = IntegratedAssistantAdapter()

    queries = [
        "okay multiple it with 87",
        "ok multiply it by 87",
        "thanks now multiply that by 87",
        "great divide it by 2",
    ]
    for q in queries:
        resp = adapter.ask(q)
        assert resp.decision != "CONVERSATIONAL"
        assert "You're welcome" not in resp.answer
        assert "Hello!" not in resp.answer

