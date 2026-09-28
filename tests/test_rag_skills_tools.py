"""
Unit tests for Step 11 Skill Resolution & Governed Tools.
"""

import pytest

from chakrview.runtime.skills import (
    SkillDomain,
    SkillPolicy,
    Skill,
    SkillRegistry,
    RuleBasedSkillResolver,
    get_standard_skill_registry,
)
from chakrview.runtime.tools import (
    CalculatorTool,
    TextUtilityTool,
    ToolRegistry,
    ToolExecutor,
    ToolResult,
    get_standard_tool_registry,
)


def test_rule_based_skill_resolver_domains():
    reg = get_standard_skill_registry()
    resolver = RuleBasedSkillResolver()

    # Math
    s_math = resolver.resolve("calculate the sum 45 + 55", reg)
    assert s_math.domain == SkillDomain.MATHEMATICS
    assert "calculator" in s_math.policy.allowed_tools

    # Coding
    s_code = resolver.resolve("write a python def fibonacci(n):", reg)
    assert s_code.domain == SkillDomain.CODING

    # Reasoning
    s_reason = resolver.resolve("why does causal masking prevent lookahead fallacy?", reg)
    assert s_reason.domain == SkillDomain.REASONING

    # Enterprise
    s_corp = resolver.resolve("what is the company revenue policy for Q3?", reg)
    assert s_corp.domain == SkillDomain.ENTERPRISE

    # Writing
    s_write = resolver.resolve("write an essay summarizing the history of computing", reg)
    assert s_write.domain == SkillDomain.WRITING

    # Analysis
    s_ana = resolver.resolve("analyze and compare the benchmark tradeoffs between CPU and GPU", reg)
    assert s_ana.domain == SkillDomain.ANALYSIS

    # System
    s_sys = resolver.resolve("how to optimize cpu ram hardware cache latency?", reg)
    assert s_sys.domain == SkillDomain.SYSTEM

    # General (fallback)
    s_gen = resolver.resolve("hello there, good morning!", reg)
    assert s_gen.domain == SkillDomain.GENERAL


def test_calculator_tool_valid_arithmetic():
    calc = CalculatorTool()

    # Basic arithmetic
    r1 = calc.execute(expression="2 + 3 * 4")
    assert r1.success is True
    assert r1.output == 14

    # Parentheses and negative numbers
    r2 = calc.execute(expression="-(10 - 2) * (5 + 1)")
    assert r2.success is True
    assert r2.output == -48

    # Power and division
    r3 = calc.execute(expression="2 ** 8 / 4")
    assert r3.success is True
    assert r3.output == 64


def test_calculator_tool_zero_division():
    calc = CalculatorTool()
    r = calc.execute(expression="10 / 0")
    assert r.success is False
    assert "division or modulo by zero" in r.error.lower()


def test_calculator_tool_security_rejection():
    calc = CalculatorTool()

    # Disallow function calls
    r1 = calc.execute(expression="abs(-5)")
    assert r1.success is False
    assert "Disallowed AST node" in r1.error

    # Disallow variable names and imports
    r2 = calc.execute(expression="__import__('os').system('dir')")
    assert r2.success is False
    assert "Disallowed AST node" in r2.error

    # Disallow attributes
    r3 = calc.execute(expression="(1).__class__")
    assert r3.success is False

    # Disallow huge exponents (DoS prevention)
    r4 = calc.execute(expression="2 ** 1000")
    assert r4.success is False
    assert "safety limit" in r4.error.lower()


def test_text_utility_tool():
    tool = TextUtilityTool()

    r1 = tool.execute(operation="word_count", text="One two three four five")
    assert r1.success is True
    assert r1.output == 5

    r2 = tool.execute(operation="char_count", text="abcde")
    assert r2.success is True
    assert r2.output == 5

    r3 = tool.execute(operation="uppercase", text="hello")
    assert r3.success is True
    assert r3.output == "HELLO"

    r4 = tool.execute(operation="unknown_op", text="hello")
    assert r4.success is False
    assert "Unsupported operation" in r4.error


def test_tool_executor_policy_permission_check():
    reg = get_standard_tool_registry()
    executor = ToolExecutor(reg)

    policy_with_calc = SkillPolicy(allowed_tools=["calculator"])
    policy_no_tools = SkillPolicy(allowed_tools=[])

    # Authorized tool
    res_auth = executor.execute("calculator", {"expression": "5 * 5"}, skill_policy=policy_with_calc)
    assert res_auth.success is True
    assert res_auth.output == 25

    # Unauthorized tool rejection
    res_unauth = executor.execute("calculator", {"expression": "5 * 5"}, skill_policy=policy_no_tools)
    assert res_unauth.success is False
    assert "Permission Denied" in res_unauth.error

    # Unknown tool rejection
    res_unknown = executor.execute("shell_exec", {"cmd": "ls"}, skill_policy=policy_with_calc)
    assert res_unknown.success is False
    assert "Permission Denied" in res_unknown.error
