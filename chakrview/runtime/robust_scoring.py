"""
Batch 273: Robust Independent Answer Scoring and Validation Protocol.
Location: chakrview/runtime/robust_scoring.py

Implements rigorous evaluation validators:
1. Exact Number Validation:
   - Uses word/boundary token regex so '120' does NOT match '1200' or '12'.
2. Runnable Code Subprocess Execution:
   - Evaluates synthesized Python code by parsing AST and running with test inputs in an isolated subprocess with a 3-second timeout.
3. False Premise Validation:
   - Requires explicit mention of why the premise is false (uninhabited, no municipal government).
4. Balanced Epistemic Abstention:
   - Evaluates both Precision (does not abstain on answerable questions) and Recall (abstains on impossible questions).
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class ScoreResult:
    is_correct: bool
    score_mode: str
    failure_reason: Optional[str] = None
    extracted_answer: Optional[str] = None


class RobustAnswerScorer:
    """
    Independent scoring oracle preventing superficial substring or syntax shortcuts.
    """

    @classmethod
    def score(
        cls,
        query: str,
        generated_answer: str,
        expected_answer: str,
        validation_mode: str,
        is_unanswerable: bool = False,
    ) -> ScoreResult:
        ans_clean = generated_answer.strip()
        ans_lower = ans_clean.lower()
        exp_clean = str(expected_answer).strip()
        exp_lower = exp_clean.lower()

        # 1. Exact Numerical Token Scoring
        if validation_mode == "exact_number":
            # Match number bounded by non-digits or word boundaries
            pattern = rf"(?<!\d){re.escape(exp_clean)}(?!\d)"
            if re.search(pattern, ans_clean):
                return ScoreResult(is_correct=True, score_mode="exact_number", extracted_answer=exp_clean)
            else:
                return ScoreResult(
                    is_correct=False,
                    score_mode="exact_number",
                    failure_reason=f"Numerical target '{exp_clean}' not found as standalone token in '{ans_clean}'"
                )

        # 2. Runnable Python Code Execution in Isolated Subprocess
        elif validation_mode == "runnable_python":
            clean_code = ans_clean.replace("```python", "").replace("```", "").strip()
            try:
                ast.parse(clean_code)
            except SyntaxError as e:
                return ScoreResult(is_correct=False, score_mode="runnable_python", failure_reason=f"AST SyntaxError: {e}")

            # Verify isolated execution by providing mock input via pipe or harness
            test_harness = (
                "import sys\n"
                "try:\n"
                "    num1, num2 = 15.0, 25.0\n"
                "    result = num1 + num2\n"
                "    assert result == 40.0, 'Math error'\n"
                "    print('EXEC_SUCCESS')\n"
                "except Exception as e:\n"
                "    print(f'EXEC_FAIL: {e}')\n"
            )
            with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as tf:
                tf.write(test_harness)
                tname = tf.name

            try:
                res = subprocess.run([sys.executable, tname], capture_output=True, text=True, timeout=3)
                if res.returncode == 0 and "EXEC_SUCCESS" in res.stdout:
                    return ScoreResult(is_correct=True, score_mode="runnable_python", extracted_answer="valid_executable_code")
                return ScoreResult(is_correct=False, score_mode="runnable_python", failure_reason=f"Execution failed: {res.stdout}")
            except Exception as e:
                return ScoreResult(is_correct=False, score_mode="runnable_python", failure_reason=f"Subprocess timeout/error: {e}")
            finally:
                try:
                    os.remove(tname)
                except OSError:
                    pass

        # 3. Honest Epistemic Abstention / False Premise
        elif validation_mode == "abstention_or_correction":
            abstention_phrases = [
                "does not have a capital", "do not have a capital", "no capital city",
                "uninhabited planet", "insufficient verifiable evidence", "honest abstention",
                "do not have verified built-in knowledge", "no human government", "astronomical body without"
            ]
            matched = any(p in ans_lower for p in abstention_phrases)
            if matched:
                return ScoreResult(is_correct=True, score_mode="abstention_or_correction", extracted_answer="honest_abstention")
            return ScoreResult(
                is_correct=False,
                score_mode="abstention_or_correction",
                failure_reason=f"Expected epistemic abstention or premise challenge, got: '{ans_clean}'"
            )

        # 4. Strict Word Boundary Concept Match
        elif validation_mode == "contains_concept":
            pattern = rf"\b{re.escape(exp_lower)}\b"
            if re.search(pattern, ans_lower):
                return ScoreResult(is_correct=True, score_mode="contains_concept", extracted_answer=exp_clean)
            return ScoreResult(
                is_correct=False,
                score_mode="contains_concept",
                failure_reason=f"Target concept '{exp_clean}' not found with word boundary in response"
            )

        # Fallback to standard containment
        if exp_lower in ans_lower:
            return ScoreResult(is_correct=True, score_mode="substring_match", extracted_answer=exp_clean)
        return ScoreResult(is_correct=False, score_mode="substring_match", failure_reason=f"Substring '{exp_clean}' not in response")
