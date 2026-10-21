import pytest
from chakrview.runtime.coding_assistant import CodingAssistant


def test_coding_assistant_input_validation():
    ca = CodingAssistant()
    
    # Empty inputs fail gracefully
    res_draft = ca.draft_code("")
    assert "error" in res_draft
    
    res_explain = ca.explain_code("")
    assert "error" in res_explain

    res_fix = ca.fix_bug("")
    assert "error" in res_fix

    res_tests = ca.generate_tests("")
    assert "error" in res_tests

    # Unsupported language
    res_lang = ca.draft_code("sort array", language="rust")
    assert res_lang.get("language_supported") is False


def test_coding_assistant_sandbox_tool():
    ca = CodingAssistant()
    
    # Valid execution
    res = ca.test_in_sandbox(
        code_snippet="def mul(a, b): return a * b",
        test_assertions="assert mul(3, 4) == 12"
    )
    assert res["passed"] is True
    assert res["syntax_valid"] is True
    assert res["tool_assisted"] is True

    # Security policy rejection
    res_bad = ca.test_in_sandbox(
        code_snippet="import os; os.listdir('.')",
        test_assertions=""
    )
    assert res_bad["passed"] is False
    assert res_bad["rejected_by_policy"] is True
