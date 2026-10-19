"""
Verified Tool Router and Neuro-Symbolic Evaluator for ChakrView.
Integrates SafeCalculator and deterministic tools to enable exact mathematical and symbolic
capabilities while strictly preserving safety gates and raw model baseline immutability.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional, Tuple

from chakrview.runtime.safe_calculator import SafeCalculator
from chakrview.runtime.tools import ToolResult


@dataclass
class ToolExecutionTrace:
    query: str
    tool_name: str
    tool_input: Dict[str, Any]
    tool_output: Any
    success: bool
    error: Optional[str] = None
    execution_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class VerifiedToolRouter:
    """
    Constrained, allowlisted tool router for ChakrView.
    Detects when a task requires exact symbolic or mathematical computation and routes
    to safe, AST-governed tools without unrestricted shell or Python eval.
    """
    def __init__(self):
        self.calculator = SafeCalculator()
        self.math_pattern = re.compile(
            r"(\b\d+(?:\.\d+)?\s*(?:[\+\-\*\/x\^]|(?:\*\*))\s*\d+(?:\.\d+)?(?:\s*(?:[\+\-\*\/x\^]|(?:\*\*))\s*\d+(?:\.\d+)?)*\b)"
        )

    def extract_arithmetic_followup(self, user_query: str) -> Optional[Tuple[str, str, str]]:
        """
        Detects follow-up arithmetic operations referring to a prior antecedent or pronoun ('it', 'that', 'this', 'result').
        Examples:
          - "okay multiple it with 87" -> ("*", "87", "multiply")
          - "multiply it by 87" -> ("*", "87", "multiply")
          - "now multiply that by 87" -> ("*", "87", "multiply")
          - "divide it by 5" -> ("/", "5", "divide")
          - "add 10 to it" -> ("+", "10", "add")
          - "subtract 4 from that" -> ("-", "4", "subtract")
        Returns:
          Tuple of (operator_symbol, operand_str, op_verb) or None if not an arithmetic follow-up.
        """
        clean_q = user_query.strip()
        s = re.sub(r"[?!.,;]+$", "", clean_q)
        s_lower = s.lower()

        # Remove leading conversational filler
        s_lower = re.sub(r"^(?:ok|okay|now|then|and|so|please|well)\b\s*", "", s_lower).strip()

        # Check if single number present
        nums = re.findall(r"\b\d+(?:\.\d+)?\b", s_lower)
        if len(nums) != 1:
            return None
        operand = nums[0]

        # Multiplication patterns
        if re.search(r"\b(?:multiply|multiplied|multiple|times)\s+(?:it|that|this|the\s+result)?\s*(?:by|with)\s+" + re.escape(operand) + r"\b", s_lower) or \
           re.search(r"\b(?:multiply|multiplied|multiple|times)\s+(?:it|that|this|the\s+result)\b", s_lower):
            return ("*", operand, "multiply")

        # Division patterns
        if re.search(r"\b(?:divide|divided)\s+(?:it|that|this|the\s+result)?\s*(?:by|with)\s+" + re.escape(operand) + r"\b", s_lower) or \
           re.search(r"\b(?:divide|divided)\s+(?:it|that|this|the\s+result)\b", s_lower):
            return ("/", operand, "divide")

        # Addition patterns: "add 10 to it", "add 10 to that", "add it with 10", "add 10"
        if re.search(r"\b(?:add|plus)\s+" + re.escape(operand) + r"\s+(?:to|in|with)\s+(?:it|that|this|the\s+result)\b", s_lower) or \
           re.search(r"\b(?:add|plus)\s+(?:it|that|this|the\s+result)\s+(?:with|and|to)\s+" + re.escape(operand) + r"\b", s_lower) or \
           re.search(r"\b(?:add|plus)\s+(?:to\s+)?(?:it|that|this|the\s+result)\b", s_lower):
            return ("+", operand, "add")

        # Subtraction patterns: "subtract 10 from it", "subtract it by 10", "minus 10"
        if re.search(r"\b(?:subtract|minus)\s+" + re.escape(operand) + r"\s+from\s+(?:it|that|this|the\s+result)\b", s_lower) or \
           re.search(r"\b(?:subtract|minus)\s+(?:from\s+)?(?:it|that|this|the\s+result)\b", s_lower):
            return ("-", operand, "subtract")

        return None

    def extract_arithmetic_expression(self, user_query: str) -> Optional[Tuple[str, Optional[str]]]:
        """
        Parses arithmetic expressions from natural language or formulaic text.
        Returns:
            (expression_string, None) if cleanly extracted.
            (None, error_reason) if ambiguous or partial expression detected.
            None if not an arithmetic query.
        """
        clean_q = user_query.strip()
        s = re.sub(r"[?!.,;]+$", "", clean_q)
        s_lower = s.lower()

        # Find all numbers in the query
        nums = re.findall(r"\b\d+(?:\.\d+)?\b", s_lower)
        if not nums:
            return None

        # Check for arithmetic operational cues or symbols
        has_math_symbols = bool(re.search(r"[\+\-\*\/x\^]|(?:\*\*)", s_lower))
        has_math_words = bool(re.search(
            r"\b(?:add|plus|sum|subtract|minus|difference|multiply|multiplied|multiple|times|product|divide|divided|over|quotient|calculate|compute)\b",
            s_lower
        ))

        if not has_math_symbols and not has_math_words:
            return None

        # If only 1 number and no arithmetic operators/cues, it's not arithmetic
        if len(nums) == 1 and not has_math_symbols:
            return None

        # Handle verbal prefix patterns like "multiply X by Y" or "divide X by Y"
        s_lower = re.sub(r"\bmultiply\s+(\d+(?:\.\d+)?)\s+by\s+(\d+(?:\.\d+)?)\b", r"\1 * \2", s_lower)
        s_lower = re.sub(r"\bdivide\s+(\d+(?:\.\d+)?)\s+by\s+(\d+(?:\.\d+)?)\b", r"\1 / \2", s_lower)

        # Phrasal operator normalizations
        subs = [
            (r"\b(?:multiplied\s+by|multiple\s+by|times)\b", " * "),
            (r"\b(?:divided\s+by|over)\b", " / "),
            (r"\b(?:plus)\b", " + "),
            (r"\b(?:minus)\b", " - "),
            (r"\b(?:and\s+then\s+add|and\s+add|then\s+add|add)\s+(\d+(?:\.\d+)?)(?:\s+(?:in|to)\s+(?:it|that|this))?\b", r" + \1 "),
            (r"\b(?:and\s+then\s+subtract|and\s+subtract|then\s+subtract|subtract)\s+(\d+(?:\.\d+)?)(?:\s+(?:from)\s+(?:it|that|this))?\b", r" - \1 "),
            (r"\b(?:and\s+then\s+multiply(?:\s+by)?|and\s+multiply(?:\s+by)?|then\s+multiply(?:\s+by)?)\s+(\d+(?:\.\d+)?)\b", r" * \1 "),
            (r"\b(?:and\s+then\s+divide(?:\s+by)?|and\s+divide(?:\s+by)?|then\s+divide(?:\s+by)?)\s+(\d+(?:\.\d+)?)\b", r" / \1 "),
            (r"\band\s*\+", " + "),
            (r"\band\s*\-", " - "),
            (r"\band\s*\*|\band\s*[xX]", " * "),
            (r"\band\s*\/", " / "),
        ]
        for pat, rep in subs:
            s_lower = re.sub(pat, rep, s_lower)

        # Remove commas, semicolons that might have separated clauses
        s_lower = re.sub(r"[,;]+", " ", s_lower)

        # Extract mathematical expression slice from first digit/paren to last digit/paren
        match = re.search(r"([\(\d][\d\s\+\-\*\/\(\)\^\.]*[\)\d])", s_lower)
        if not match:
            return None

        candidate_expr = match.group(1).strip()

        # Invariant: Check for dangling trailing operator or dangling leading operator
        # e.g., '99 * and then' -> if after the extracted expression there was an operator like '*', '+', '-', '/'
        # that was discarded, reject as incomplete!
        post_expr_slice = s_lower[match.end():].strip()
        pre_expr_slice = s_lower[:match.start()].strip()
        if re.search(r"[\+\-\*\/x\^]", post_expr_slice):
            return None, f"Dangling or incomplete trailing operator detected in '{user_query}'."

        # Invariant: Ensure the candidate expression accounts for all numbers present in the arithmetic query.
        # If numbers in the query were dropped, do NOT evaluate a partial expression!
        expr_nums = re.findall(r"\b\d+(?:\.\d+)?\b", candidate_expr)
        if expr_nums != nums:
            return None, f"Incomplete expression parsed: expected numbers {nums}, but captured {expr_nums}."

        return candidate_expr, None

    def route_and_execute(self, user_query: str) -> Optional[ToolExecutionTrace]:
        # 1. Parse expression with natural language and full-coverage validation
        extraction = self.extract_arithmetic_expression(user_query)
        if extraction is None:
            return None

        expr, error_reason = extraction
        if error_reason is not None:
            # Ambiguous or partial expression: Return safe failure trace rather than dropping trailing operations
            return ToolExecutionTrace(
                query=user_query,
                tool_name=self.calculator.tool_id,
                tool_input={"raw_query": user_query},
                tool_output=None,
                success=False,
                error=error_reason,
                execution_time_ms=0.0
            )

        t0 = time.perf_counter()
        res: ToolResult = self.calculator.execute(expression=expr)
        elapsed = (time.perf_counter() - t0) * 1000.0

        return ToolExecutionTrace(
            query=user_query,
            tool_name=self.calculator.tool_id,
            tool_input={"expression": expr},
            tool_output=res.output if res.success else None,
            success=res.success,
            error=res.error if not res.success else None,
            execution_time_ms=elapsed
        )


def evaluate_arithmetic_probes_with_tool(
    probes: List[str],
    ground_truths: List[int | float]
) -> Dict[str, Any]:
    router = VerifiedToolRouter()
    results = []
    correct = 0

    for query, expected in zip(probes, ground_truths):
        trace = router.route_and_execute(query)
        is_correct = False
        if trace and trace.success:
            if trace.tool_output == expected:
                is_correct = True
                correct += 1

        results.append({
            "query": query,
            "expected": expected,
            "tool_used": trace.tool_name if trace else None,
            "tool_output": trace.tool_output if trace else None,
            "correct": is_correct
        })

    return {
        "total_probes": len(probes),
        "correct": correct,
        "accuracy": round(correct / len(probes), 4),
        "probes_evaluated": results
    }
