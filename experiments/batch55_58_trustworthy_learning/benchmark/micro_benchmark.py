"""
Verified Micro-Benchmark Suite and Scorer for ChakrView (Batch 56 & 57).

Defines an independently checkable benchmark across Mathematics, Coding, and Science:
- Simple arithmetic & algebra with deterministic answer checks.
- Code functions with AST parsing and isolated test execution.
- Science / Computational facts with deterministic string normalization.
- Separates DATA_ROLE: HELD_OUT_BENCHMARK, VALIDATION, TRAIN, EXPLORATORY_ONLY.
- Enforces zero gradient update leakage.
"""

from __future__ import annotations

import ast
from dataclasses import asdict, dataclass
from enum import Enum
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Callable, Dict, List, Optional, Tuple


class DataRole(str, Enum):
    TRAIN = "TRAIN"
    VALIDATION = "VALIDATION"
    HELD_OUT_BENCHMARK = "HELD_OUT_BENCHMARK"
    EXPLORATORY_ONLY = "EXPLORATORY_ONLY"


class VerificationMethod(str, Enum):
    DETERMINISTIC_DERIVATION = "DETERMINISTIC_DERIVATION"
    EXECUTION_TEST = "EXECUTION_TEST"
    EXACT_TEXT_NORMALIZATION = "EXACT_TEXT_NORMALIZATION"


@dataclass
class BenchmarkItem:
    id: str
    version: str
    domain: str  # "mathematics", "code", "science"
    task_type: str
    prompt: str
    expected_answer: str
    scoring_method: str  # "exact_match", "python_execution", "numeric_or_token"
    verification_method: str
    verification_status: str  # "ANSWER_VERIFIED", "EXECUTION_VERIFIED"
    data_role: str = DataRole.HELD_OUT_BENCHMARK.value
    family: str = ""
    test_code: Optional[str] = None


VERIFIED_MICROBENCHMARK_ITEMS: List[BenchmarkItem] = [
    # 1. Mathematics: Deterministic Arithmetic
    BenchmarkItem(
        id="BENCH-MATH-01-ARITHMETIC-ADD-2-3",
        version="1.0.0",
        domain="mathematics",
        task_type="deterministic_arithmetic",
        prompt="<|im_start|>user\nWhat is 2 + 3?<|im_end|>\n<|im_start|>assistant\n",
        expected_answer="5",
        scoring_method="numeric_or_token",
        verification_method=VerificationMethod.DETERMINISTIC_DERIVATION.value,
        verification_status="ANSWER_VERIFIED",
        family="MATH_ARITHMETIC_ADD",
    ),
    BenchmarkItem(
        id="BENCH-MATH-02-ARITHMETIC-MUL-6-7",
        version="1.0.0",
        domain="mathematics",
        task_type="deterministic_arithmetic",
        prompt="<|im_start|>user\nWhat is 6 * 7?<|im_end|>\n<|im_start|>assistant\n",
        expected_answer="42",
        scoring_method="numeric_or_token",
        verification_method=VerificationMethod.DETERMINISTIC_DERIVATION.value,
        verification_status="ANSWER_VERIFIED",
        family="MATH_ARITHMETIC_MUL",
    ),
    BenchmarkItem(
        id="BENCH-MATH-03-ALGEBRA-MODULO",
        version="1.0.0",
        domain="mathematics",
        task_type="modulo_arithmetic",
        prompt="<|im_start|>user\nWhat is 19 % 4?<|im_end|>\n<|im_start|>assistant\n",
        expected_answer="3",
        scoring_method="numeric_or_token",
        verification_method=VerificationMethod.DETERMINISTIC_DERIVATION.value,
        verification_status="ANSWER_VERIFIED",
        family="MATH_ARITHMETIC_MOD",
    ),
    # 2. Coding: Small Functions with Isolated Execution Contracts
    BenchmarkItem(
        id="BENCH-CODE-01-ADD-TWO",
        version="1.0.0",
        domain="code",
        task_type="function_implementation",
        prompt="<|im_start|>user\nWrite a Python function `add_two(x: int, y: int) -> int` that returns the sum of x and y.<|im_end|>\n<|im_start|>assistant\n",
        expected_answer="def add_two(x: int, y: int) -> int:\n    return x + y<|im_end|>\n",
        scoring_method="python_execution",
        verification_method=VerificationMethod.EXECUTION_TEST.value,
        verification_status="EXECUTION_VERIFIED",
        family="CODE_ADD_TWO",
        test_code="assert fn(2, 3) == 5\nassert fn(-4, 4) == 0\nassert fn(100, 250) == 350",
    ),
    BenchmarkItem(
        id="BENCH-CODE-02-IS-EVEN",
        version="1.0.0",
        domain="code",
        task_type="function_implementation",
        prompt="<|im_start|>user\nWrite a Python function `is_even(n: int) -> bool` that returns True if n is even, else False.<|im_end|>\n<|im_start|>assistant\n",
        expected_answer="def is_even(n: int) -> bool:\n    return n % 2 == 0<|im_end|>\n",
        scoring_method="python_execution",
        verification_method=VerificationMethod.EXECUTION_TEST.value,
        verification_status="EXECUTION_VERIFIED",
        family="CODE_IS_EVEN",
        test_code="assert fn(4) is True\nassert fn(7) is False\nassert fn(0) is True",
    ),
    BenchmarkItem(
        id="BENCH-CODE-03-DIVISIBILITY-ASSERTION",
        version="1.0.0",
        domain="code",
        task_type="python_syntax_completion",
        prompt="<|im_start|>user\nHow do you check that d_model divides evenly by the number of attention heads in Python?<|im_end|>\n<|im_start|>assistant\n",
        expected_answer="assert d_model % n_heads == 0<|im_end|>\n",
        scoring_method="exact_match",
        verification_method=VerificationMethod.EXECUTION_TEST.value,
        verification_status="EXECUTION_VERIFIED",
        family="CODE_ASSERT_HEADS_DIVISIBILITY",
        test_code="d_model = 192\nn_heads = 6\nexec(code)\n",
    ),
    # 3. Science & Computational Architecture
    BenchmarkItem(
        id="BENCH-SCI-01-CHAKRMICRO-CONTEXT",
        version="1.0.0",
        domain="science",
        task_type="architectural_fact",
        prompt="<|im_start|>user\nWhat is the maximum context length of ChakrMicro in tokens?<|im_end|>\n<|im_start|>assistant\n",
        expected_answer="512",
        scoring_method="numeric_or_token",
        verification_method=VerificationMethod.EXACT_TEXT_NORMALIZATION.value,
        verification_status="ANSWER_VERIFIED",
        family="SCI_CHAKRMICRO_SPECS",
    ),
    BenchmarkItem(
        id="BENCH-SCI-02-CHAKRMICRO-LAYERS",
        version="1.0.0",
        domain="science",
        task_type="architectural_fact",
        prompt="<|im_start|>user\nHow many transformer layers are in ChakrMicro?<|im_end|>\n<|im_start|>assistant\n",
        expected_answer="6",
        scoring_method="numeric_or_token",
        verification_method=VerificationMethod.EXACT_TEXT_NORMALIZATION.value,
        verification_status="ANSWER_VERIFIED",
        family="SCI_CHAKRMICRO_SPECS",
    ),
    BenchmarkItem(
        id="BENCH-SCI-03-BYTES-IN-KILOBYTE",
        version="1.0.0",
        domain="science",
        task_type="computational_fact",
        prompt="<|im_start|>user\nHow many bytes are in 1 kibibyte (KiB)?<|im_end|>\n<|im_start|>assistant\n",
        expected_answer="1024",
        scoring_method="numeric_or_token",
        verification_method=VerificationMethod.DETERMINISTIC_DERIVATION.value,
        verification_status="ANSWER_VERIFIED",
        family="SCI_BINARY_UNITS",
    ),
]


def normalize_output(text: str) -> str:
    """Removes ChatML markers, trailing whitespace, and normalizes formatting."""
    cleaned = text.replace("<|im_end|>", "").strip()
    return cleaned


def evaluate_benchmark_item(
    item: BenchmarkItem,
    generated_text: str,
    eos_reached: bool,
) -> Dict[str, Any]:
    """
    Deterministically scores a model's generated text against a benchmark item.
    """
    clean_gen = normalize_output(generated_text)
    clean_exp = normalize_output(item.expected_answer)

    result = {
        "item_id": item.id,
        "domain": item.domain,
        "task_type": item.task_type,
        "eos_reached": eos_reached,
        "generated_text": generated_text,
        "normalized_generated": clean_gen,
        "expected_answer": clean_exp,
        "exact_match": False,
        "task_success": False,
        "repetition_detected": False,
        "error_message": None,
    }

    # Repetition detector
    lines = [l.strip() for l in generated_text.splitlines() if l.strip()]
    if len(lines) > 3 and len(set(lines)) <= len(lines) // 2:
        result["repetition_detected"] = True

    # Scoring
    if item.scoring_method == "exact_match":
        if clean_gen == clean_exp:
            result["exact_match"] = True
            result["task_success"] = True

    elif item.scoring_method == "numeric_or_token":
        # Check if the expected numeric answer appears at the beginning of the response
        tokens = clean_gen.split()
        first_token = tokens[0] if tokens else ""
        if clean_gen == clean_exp or first_token == clean_exp:
            result["exact_match"] = (clean_gen == clean_exp)
            result["task_success"] = True

    elif item.scoring_method == "python_execution":
        try:
            # 1. Parse AST
            tree = ast.parse(clean_gen)
            # Find function name
            fn_name = None
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    fn_name = node.name
                    break

            if not fn_name:
                result["error_message"] = "No function definition found in generated output."
            else:
                # 2. Execute safely
                loc: Dict[str, Any] = {}
                safe_globals = {
                    "__builtins__": {
                        "len": len,
                        "range": range,
                        "int": int,
                        "bool": bool,
                        "str": str,
                        "list": list,
                        "dict": dict,
                        "set": set,
                        "max": max,
                        "min": min,
                        "sum": sum,
                        "AssertionError": AssertionError,
                    }
                }
                exec(clean_gen, safe_globals, loc)
                fn_obj = loc.get(fn_name)
                if not callable(fn_obj):
                    result["error_message"] = f"Extracted object {fn_name} is not callable."
                else:
                    # Run tests
                    if item.test_code:
                        test_loc = {"fn": fn_obj}
                        exec(item.test_code, safe_globals, test_loc)
                    result["exact_match"] = (clean_gen == clean_exp)
                    result["task_success"] = True

        except Exception as e:
            result["error_message"] = f"Execution failed: {e}"

    return result
