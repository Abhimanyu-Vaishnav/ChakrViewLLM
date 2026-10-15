import pytest
from chakrview.cognition.grammar_constrained_decoder import PlanDSLGrammarChecker, GrammarState


def test_grammar_checker_goal_lines():
    # Valid goals
    valid_goals = [
        "GOAL: compound_arithmetic\n",
        "GOAL: text_transform_chain\n",
        "GOAL: clarification_needed\n",
        "GOAL: unsupported\n",
        "GOAL: multi_tool_dag\n",
    ]
    for g in valid_goals:
        valid, st = PlanDSLGrammarChecker.is_valid_partial(g)
        assert valid is True
        assert st == GrammarState.STEP_START_OR_CLAR

    # Invalid goal
    valid, st = PlanDSLGrammarChecker.is_valid_partial("GOAL: execute_hack\n")
    assert valid is False
    assert st == GrammarState.INVALID


def test_grammar_checker_step_lines():
    # Valid full step
    step_str = "GOAL: compound_arithmetic\nSTEP 0: tool_call | calculator | args={expr=25*4} | deps=[]\n"
    valid, st = PlanDSLGrammarChecker.is_valid_partial(step_str)
    assert valid is True
    assert st == GrammarState.STEP_START_OR_CLAR

    # Valid step with upstream dependency
    step_str2 = step_str + "STEP 1: tool_call | calculator | args={expr=$0+10} | deps=[0]\n"
    valid, st = PlanDSLGrammarChecker.is_valid_partial(step_str2)
    assert valid is True
    assert st == GrammarState.STEP_START_OR_CLAR

    # Invalid tool in step
    step_invalid_tool = "GOAL: compound_arithmetic\nSTEP 0: tool_call | unauthorized_tool | args={expr=25*4} | deps=[]\n"
    valid, st = PlanDSLGrammarChecker.is_valid_partial(step_invalid_tool)
    assert valid is False
    assert st == GrammarState.INVALID

    # Invalid action in step
    step_invalid_act = "GOAL: compound_arithmetic\nSTEP 0: delete_db | calculator | args={expr=25*4} | deps=[]\n"
    valid, st = PlanDSLGrammarChecker.is_valid_partial(step_invalid_act)
    assert valid is False
    assert st == GrammarState.INVALID


def test_grammar_checker_clarification_and_end_plan():
    # Full valid clarification plan
    clar_plan = "GOAL: clarification_needed\nCLARIFICATION: NEEDED\n<END_PLAN>"
    valid, st = PlanDSLGrammarChecker.is_valid_partial(clar_plan)
    assert valid is True
    assert st == GrammarState.TERMINATED

    # Full valid step plan with NONE clarification
    full_plan = (
        "GOAL: compound_arithmetic\n"
        "STEP 0: tool_call | calculator | args={expr=10+5} | deps=[]\n"
        "CLARIFICATION: NONE\n"
        "<END_PLAN>"
    )
    valid, st = PlanDSLGrammarChecker.is_valid_partial(full_plan)
    assert valid is True
    assert st == GrammarState.TERMINATED


def test_grammar_checker_partial_delimiters():
    # Incomplete step prefix
    valid, st = PlanDSLGrammarChecker.is_valid_partial("GOAL: compound_arithmetic\nSTEP 0")
    assert valid is True
    assert st == GrammarState.STEP_START_OR_CLAR

    # Partial pipe
    valid, st = PlanDSLGrammarChecker.is_valid_partial("GOAL: compound_arithmetic\nSTEP 0: tool_call |")
    assert valid is True

    # Partial args
    valid, st = PlanDSLGrammarChecker.is_valid_partial("GOAL: compound_arithmetic\nSTEP 0: tool_call | calculator | args={")
    assert valid is True
    assert st == GrammarState.STEP_ARGS

    # Malformed prefix
    valid, st = PlanDSLGrammarChecker.is_valid_partial("RANDOM NOISE")
    assert valid is False
    assert st == GrammarState.INVALID
