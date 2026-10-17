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

    def route_and_execute(self, user_query: str) -> Optional[ToolExecutionTrace]:
        # 1. Inspect query for deterministic arithmetic expressions
        clean_q = user_query.strip()
        
        # Match common arithmetic queries e.g. "Calculate 15 * 4", "What is 100 - 37?", "25 + 14"
        match = self.math_pattern.search(clean_q)
        if not match:
            return None

        expr = match.group(1).strip()
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
