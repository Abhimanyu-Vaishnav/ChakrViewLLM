import pytest
from unittest.mock import MagicMock
from chakrview.runtime.coding_assistant import CodingAssistant


def test_coding_assistant_input_validation():
    ca = CodingAssistant()
    
    # 1. Empty and whitespace-only inputs fail gracefully without initializing model
    assert ca.initialized is False
    res_draft = ca.draft_code("")
    assert "error" in res_draft
    assert ca.initialized is False

    res_draft_ws = ca.draft_code("   \n\t  ")
    assert "error" in res_draft_ws
    assert ca.initialized is False
    
    res_explain = ca.explain_code("")
    assert "error" in res_explain
    assert ca.initialized is False

    res_explain_ws = ca.explain_code("   ")
    assert "error" in res_explain_ws
    assert ca.initialized is False

    res_fix = ca.fix_bug("")
    assert "error" in res_fix
    assert ca.initialized is False

    res_fix_ws = ca.fix_bug("   ")
    assert "error" in res_fix_ws
    assert ca.initialized is False

    res_tests = ca.generate_tests("")
    assert "error" in res_tests
    assert ca.initialized is False

    res_tests_ws = ca.generate_tests("   ")
    assert "error" in res_tests_ws
    assert ca.initialized is False

    # 2. Unsupported language rejection fails gracefully without initializing model
    res_lang = ca.draft_code("sort array", language="rust")
    assert res_lang.get("language_supported") is False
    assert "error" in res_lang
    assert ca.initialized is False


def test_coding_assistant_valid_execution_with_mock():
    # 3. Valid input follows intended execution path using mock adapter
    mock_adapter = MagicMock()
    mock_adapter.generate.return_value = {
        "generated_text": "def add_one(x):\n    return x + 1\n",
        "output_token_count": 10,
        "input_token_count": 10
    }
    
    ca = CodingAssistant(adapter=mock_adapter)
    assert ca.initialized is False

    res = ca.draft_code("Add one to x", language="python")
    assert ca.initialized is True
    mock_adapter.initialize.assert_called_once()
    mock_adapter.generate.assert_called_once()
    assert res["operation"] == "draft_code"
    assert "return x + 1" in res["native_output"]
    assert res["syntax_valid"] is True
    assert res["tool_assisted"] is False

    # Test explain with mock
    mock_adapter.reset_mock()
    res_exp = ca.explain_code("def foo(): pass")
    mock_adapter.generate.assert_called_once()
    assert res_exp["operation"] == "explain_code"

    # Test fix bug with mock
    mock_adapter.reset_mock()
    res_fix = ca.fix_bug("def foo(): pase", error_message="NameError")
    mock_adapter.generate.assert_called_once()
    assert res_fix["operation"] == "fix_bug"

    # Test generate tests with mock
    mock_adapter.reset_mock()
    res_tests = ca.generate_tests("def add(a, b): return a + b", function_name="add")
    mock_adapter.generate.assert_called_once()
    assert res_tests["operation"] == "generate_tests"


def test_coding_assistant_sandbox_tool():
    # 4. Existing sandbox functionality and security restrictions continue to work directly
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