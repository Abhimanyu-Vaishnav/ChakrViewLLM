"""
Unit and regression tests for Batches 271-280:
- Compound arithmetic multi-step evaluation ("Add 15 and 25 then subtract 8")
- Valid unit conversions (km -> m, Celsius -> Kelvin, GB -> MB)
- Dimensional validity rejection (km -> kg, invalid units)
- Non-physical temperature rejection (< 0 K)
- RobustAnswerScorer numeric token boundary checks (12 not matching 120)
- Sandboxed Python AST execution scoring
- Governed internet research permission gating (consent required before web operations)
"""
import pytest
import sys
from pathlib import Path

sys.path.insert(0, ".")

from chakrview.runtime.compound_arithmetic import CompoundArithmeticEngine, CompoundArithmeticResult
from chakrview.runtime.robust_scoring import RobustAnswerScorer, ScoreResult
from chakrview.runtime.research_workflow import PermissionGatedResearchWorkflow, ResearchState


def test_compound_arithmetic_multistep():
    # Multi-step verbal: "Add 25 and 35 then subtract 10" -> 50.0
    res = CompoundArithmeticEngine.evaluate_query("Add 25 and 35 then subtract 10")
    assert res is not None
    assert res.is_success
    assert res.result_value == 50.0
    assert "50" in res.formatted_output

    # Multi-step verbal: "Multiply 12 by 4 and add 6" -> 54.0
    res2 = CompoundArithmeticEngine.evaluate_query("Multiply 12 by 4 and add 6")
    assert res2 is not None
    assert res2.is_success
    assert res2.result_value == 54.0


def test_unit_conversions_valid():
    # Distance: 5.5 km to meters -> 5500.0 meters
    res = CompoundArithmeticEngine.evaluate_query("Convert 5.5 km to meters")
    assert res is not None
    assert res.is_success
    assert res.result_value == 5500.0
    assert "5500" in res.formatted_output

    # Temperature: 25 C to kelvin -> 298.15 K
    res2 = CompoundArithmeticEngine.evaluate_query("Convert 25 C to kelvin")
    assert res2 is not None
    assert res2.is_success
    assert round(res2.result_value, 2) == 298.15

    # Storage: 2 GB to MB -> 2048.0 MB
    res3 = CompoundArithmeticEngine.evaluate_query("Convert 2 GB to MB")
    assert res3 is not None
    assert res3.is_success
    assert res3.result_value == 2048.0


def test_unit_conversions_dimensional_rejection():
    # Dimensionally invalid: km to kg
    res = CompoundArithmeticEngine.evaluate_query("Convert 10 km to kg")
    assert res is not None
    assert not res.is_success
    assert "dimension" in (res.error or "").lower()

    # Physical violation: below absolute zero
    res2 = CompoundArithmeticEngine.evaluate_query("Convert -300 C to kelvin")
    assert res2 is not None
    assert not res2.is_success
    assert "absolutezero" in (res2.error or "").lower() or "0 kelvin" in res2.formatted_output.lower()


def test_robust_scorer_numeric_boundaries():
    # Expected number 12 must NOT match '120' or '512'
    res_false = RobustAnswerScorer.score(
        query="What is 6 + 6?",
        generated_answer="The speed limit is 120 km/h.",
        expected_answer="12",
        validation_mode="exact_number"
    )
    assert not res_false.is_correct

    # Expected number 12 DOES match 'The answer is 12.'
    res_true = RobustAnswerScorer.score(
        query="What is 6 + 6?",
        generated_answer="The answer is 12.",
        expected_answer="12",
        validation_mode="exact_number"
    )
    assert res_true.is_correct


def test_robust_scorer_code_execution():
    # Valid syntax & execution
    code_pass = "def add(a, b): return a + b\nresult = add(10, 20)\nprint('Done')"
    res = RobustAnswerScorer.score(
        query="Write Python to add two numbers",
        generated_answer=f"```python\n{code_pass}\n```",
        expected_answer="def add",
        validation_mode="runnable_python"
    )
    assert res.is_correct

    # Syntax error
    code_fail = "def add(a, b) return a + b"
    res_fail = RobustAnswerScorer.score(
        query="Write Python to add two numbers",
        generated_answer=code_fail,
        expected_answer="def add",
        validation_mode="runnable_python"
    )
    assert not res_fail.is_correct
    assert "SyntaxError" in (res_fail.failure_reason or "")


def test_research_workflow_governance():
    workflow = PermissionGatedResearchWorkflow()
    session_id = "test_session_001"
    
    # 1. Initiating external query without grant returns PERMISSION_REQUIRED
    resp = workflow.execute_research_flow(session_id, "search the web for quantum computing advances")
    assert resp.state == ResearchState.PERMISSION_REQUIRED
    assert resp.requires_user_action

    # 2. Denying consent via callback returns RESEARCH_DENIED
    resp_denied = workflow.execute_research_flow(
        session_id,
        "search the web for quantum computing advances",
        user_grant_callback=lambda: False
    )
    assert resp_denied.state == ResearchState.RESEARCH_DENIED
    assert not resp_denied.requires_user_action

    # 3. Granting consent via callback returns RESEARCH_COMPLETED
    resp_granted = workflow.execute_research_flow(
        session_id,
        "search the web for quantum computing advances",
        user_grant_callback=lambda: True
    )
    assert resp_granted.state == ResearchState.RESEARCH_COMPLETED
