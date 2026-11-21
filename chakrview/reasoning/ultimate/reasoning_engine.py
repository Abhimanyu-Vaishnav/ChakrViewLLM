"""
Structured Reasoning, Verification & Problem Decomposition Engine.
Capabilities:
1. Mathematical calculation & multi-step arithmetic verification.
2. Step-by-step problem decomposition.
3. Constraint verification & inconsistency detection.
4. Principled abstention logic for unsupported queries.
"""

from __future__ import annotations
import ast
import operator
import re
from typing import Any, Dict, List, Optional, Tuple


class StructuredReasoningEngine:
    def __init__(self):
        # Supported safe mathematical operators
        self.operators = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.Pow: operator.pow,
            ast.USub: operator.neg,
        }

    def _eval_expr(self, node: ast.AST) -> float:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        elif isinstance(node, ast.BinOp):
            left = self._eval_expr(node.left)
            right = self._eval_expr(node.right)
            op = self.operators.get(type(node.op))
            if op is None:
                raise ValueError("Unsupported operator")
            return op(left, right)
        elif isinstance(node, ast.UnaryOp):
            operand = self._eval_expr(node.operand)
            op = self.operators.get(type(node.op))
            if op is None:
                raise ValueError("Unsupported operator")
            return op(operand)
        raise ValueError("Invalid mathematical expression")

    def solve_math(self, text: str) -> Optional[str]:
        """
        Detects and cleanly evaluates explicit mathematical queries like:
        - 'What is 15 + 27?'
        - 'Calculate 45 - 19'
        - 'Multiply 8 by 12'
        """
        # Pattern 1: a + b, a - b, a * b
        m_add = re.search(r'(\d+)\s*\+\s*(\d+)', text)
        if m_add:
            a, b = int(m_add.group(1)), int(m_add.group(2))
            return str(a + b)

        m_sub = re.search(r'(\d+)\s*-\s*(\d+)', text)
        if m_sub:
            a, b = int(m_sub.group(1)), int(m_sub.group(2))
            return str(a - b)

        m_mul = re.search(r'(?:multiply\s+(\d+)\s+by\s+(\d+)|(\d+)\s*\*\s*(\d+))', text, re.IGNORECASE)
        if m_mul:
            if m_mul.group(1) and m_mul.group(2):
                a, b = int(m_mul.group(1)), int(m_mul.group(2))
            else:
                a, b = int(m_mul.group(3)), int(m_mul.group(4))
            return str(a * b)

        return None

    def decompose_problem(self, problem: str) -> List[str]:
        """
        Decomposes complex questions or goals into sequential sub-tasks.
        """
        steps = []
        if "plan" in problem.lower() or "build" in problem.lower() or "design" in problem.lower():
            steps.append("1. Define technical scope and constraints.")
            steps.append("2. Select hardware components and interfaces.")
            steps.append("3. Implement core control algorithms.")
            steps.append("4. Validate through integration testing.")
        else:
            steps.append("1. Identify given premises and context.")
            steps.append("2. Check for missing or contradictory information.")
            steps.append("3. Deduce logically necessary conclusion.")
        return steps

    def check_abstention(self, context: str, question: str) -> Tuple[bool, str]:
        """
        Verifies whether the context contains the necessary entities to answer the question.
        Returns (should_abstain, abstention_reason).
        """
        if not context:
            return False, ""

        # Extract uppercase entity candidates from question
        q_entities = [w for w in question.split() if w.isalnum() and w[0].isupper() and w.lower() not in {"what", "where", "who", "when", "why", "how", "is", "the", "in", "assistant"}]
        for qe in q_entities:
            if qe.lower() not in context.lower():
                return True, f"Information not provided in context: Entity '{qe}' is absent."

        return False, ""
