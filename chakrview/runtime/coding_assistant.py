"""
chakrview/runtime/coding_assistant.py

Practical, testable coding assistant interface for ChakrView v0.7.
Reuses the existing NeuralRuntimeAdapter with the canonical ChakrMicro model.
Supports:
1. Drafting code from a natural language specification.
2. Explaining a supplied code snippet.
3. Suggesting a bug fix for an error and code snippet.
4. Generating candidate unit test cases.
5. Optional isolated AST syntax checking and sandboxed execution.

Strictly distinguishes native model output from tool-assisted execution.
"""

import sys
import os
import ast
import time
from pathlib import Path
from typing import Dict, Any, Optional, List

from chakrview.runtime.neural_runtime_adapter import (
    NeuralRuntimeAdapter,
    CANONICAL_CHECKPOINTS_ALLOWLIST,
    DEFAULT_CHECKPOINT_KEY,
)
from evaluation.coding_suite.isolated_executor import run_process_isolated_test, audit_code_safety


class CodingAssistant:
    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        checkpoint_key: str = DEFAULT_CHECKPOINT_KEY,
        max_new_tokens: int = 48,
        decoding: str = "greedy",
        temperature: float = 0.0,
        repetition_penalty: float = 1.0,
        adapter: Optional[Any] = None,
    ):
        if adapter is not None:
            self.adapter = adapter
        else:
            self.adapter = NeuralRuntimeAdapter(
                checkpoint_path=checkpoint_path,
                max_new_tokens=max_new_tokens,
                decoding=decoding,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
            )
        self.initialized = False

    def initialize(self):
        if not self.initialized:
            self.adapter.initialize()
            self.initialized = True

    def draft_code(self, spec: str, language: str = "python") -> Dict[str, Any]:
        """Drafts a code implementation from a natural language specification."""
        if not spec or not spec.strip():
            return {"error": "Specification string cannot be empty", "native_output": ""}

        if language.lower() != "python":
            return {
                "error": f"Language '{language}' is currently experimental/unsupported. Native evaluation is verified for Python.",
                "native_output": "",
                "language_supported": False
            }

        self.initialize()
        prompt = f"# Specification: {spec.strip()}\n# Python implementation:\n"
        out = self.adapter.generate(prompt)
        gen = out["generated_text"]
        full_code = prompt + gen

        # Optional AST audit
        syntax_valid = self._check_syntax(full_code)
        safety_error = audit_code_safety(full_code)

        return {
            "operation": "draft_code",
            "prompt": prompt,
            "native_output": gen,
            "full_code": full_code,
            "syntax_valid": syntax_valid,
            "security_audit": "PASS" if not safety_error else safety_error,
            "generation_metadata": out,
            "tool_assisted": False
        }

    def explain_code(self, code_snippet: str) -> Dict[str, Any]:
        """Explains a supplied code snippet."""
        if not code_snippet or not code_snippet.strip():
            return {"error": "Code snippet cannot be empty", "native_output": ""}

        self.initialize()
        prompt = f"# Code:\n{code_snippet.strip()}\n# Explanation: What this code does:\n"
        out = self.adapter.generate(prompt)
        return {
            "operation": "explain_code",
            "prompt": prompt,
            "native_output": out["generated_text"],
            "generation_metadata": out,
            "tool_assisted": False
        }

    def fix_bug(self, code_snippet: str, error_message: str = "") -> Dict[str, Any]:
        """Suggests a fix given a buggy snippet and error message."""
        if not code_snippet or not code_snippet.strip():
            return {"error": "Code snippet cannot be empty", "native_output": ""}

        self.initialize()
        err_clause = f"# Error encountered: {error_message.strip()}\n" if error_message else ""
        prompt = f"# Buggy code:\n{code_snippet.strip()}\n{err_clause}# Fixed code:\n"
        out = self.adapter.generate(prompt)
        gen = out["generated_text"]
        full_code = prompt + gen

        return {
            "operation": "fix_bug",
            "prompt": prompt,
            "native_output": gen,
            "syntax_valid": self._check_syntax(gen),
            "generation_metadata": out,
            "tool_assisted": False
        }

    def generate_tests(self, code_snippet: str, function_name: str = "") -> Dict[str, Any]:
        """Generates unit test assertions for the supplied code."""
        if not code_snippet or not code_snippet.strip():
            return {"error": "Code snippet cannot be empty", "native_output": ""}

        self.initialize()
        fn_target = f"for function {function_name}" if function_name else ""
        prompt = f"# Given code:\n{code_snippet.strip()}\n# Unit test assertions {fn_target}:\nassert "
        out = self.adapter.generate(prompt)
        gen = "assert " + out["generated_text"]

        return {
            "operation": "generate_tests",
            "prompt": prompt,
            "native_output": gen,
            "generation_metadata": out,
            "tool_assisted": False
        }

    def test_in_sandbox(self, code_snippet: str, test_assertions: str, timeout_seconds: float = 2.0) -> Dict[str, Any]:
        """Tool-assisted process execution in an isolated sandbox. Clearly labeled as external tooling."""
        exec_res = run_process_isolated_test(code_snippet, test_assertions, timeout_seconds=timeout_seconds)
        return {
            "operation": "sandboxed_execution",
            "tool_assisted": True,
            "tool_name": "isolated_process_executor",
            "passed": exec_res["passed"],
            "syntax_valid": exec_res["syntax_valid"],
            "error": exec_res["error"],
            "latency_ms": exec_res["latency_ms"],
            "timed_out": exec_res.get("timed_out", False),
            "rejected_by_policy": exec_res.get("rejected_by_policy", False)
        }

    @staticmethod
    def _check_syntax(code_str: str) -> bool:
        try:
            ast.parse(code_str)
            return True
        except (SyntaxError, ValueError, TypeError):
            return False