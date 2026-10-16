"""
Spoken-Number Arithmetic Preprocessor and Evaluation Module (Batches 1741–1780).
Location: experiments/batch1741_1780/spoken_number_parser.py

Implements:
1. SpokenNumberParser: Pure deterministic mapping of English spoken-number words to canonical digits.
   Supported numbers: zero through nine hundred ninety-nine, compound numbers ('twenty-five', 'one hundred twenty').
   Supported operations: 'plus', 'minus', 'times', 'divided by', 'over', 'multiplied by', 'sum of', 'product of'.
2. Strict Rejection of Unsupported Input (Decimals, negative word numbers, arbitrary English).
3. Test Suite benchmarking parser accuracy and integration with SafeCalculator.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import re
from typing import Optional, Tuple
from chakrview.runtime.safe_calculator import SafeCalculator

SMALL_NUMBERS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
}

TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}

SCALES = {
    "hundred": 100,
    "thousand": 1000,
}

OPERATORS = {
    "plus": "+",
    "added to": "+",
    "and": "+",
    "minus": "-",
    "subtracted from": "-",
    "times": "*",
    "multiplied by": "*",
    "divided by": "/",
    "over": "/",
}


class SpokenNumberParser:
    """
    Deterministic tokenizer and parser converting spoken arithmetic phrases into canonical math expressions.
    Never uses eval() or exec().
    """

    @classmethod
    def parse_number_phrase(cls, words: list[str]) -> Optional[int]:
        """Parses a list of English words into an integer between 0 and 9999."""
        if not words:
            return None

        total = 0
        current = 0

        for w in words:
            w = w.lower().strip("-")
            if w in SMALL_NUMBERS:
                current += SMALL_NUMBERS[w]
            elif w in TENS:
                current += TENS[w]
            elif w == "hundred":
                if current == 0:
                    current = 1
                current *= 100
            elif w == "thousand":
                if current == 0:
                    current = 1
                total += current * 1000
                current = 0
            elif w == "and":
                continue
            elif w.isdigit():
                current += int(w)
            else:
                return None  # Unrecognized number word

        total += current
        return total

    @classmethod
    def parse_spoken_expression(cls, query: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Parses phrases like:
          'twenty times fifteen' -> '20 * 15'
          'one hundred divided by four' -> '100 / 4'
          'sum of fifty and twenty-five' -> '50 + 25'
          'subtract forty from ninety' -> '90 - 40'
        Returns: (canonical_expression, error_reason)
        """
        q = query.lower().strip().rstrip(".?!")

        # 1. Pattern: 'subtract X from Y'
        q_norm = q.replace("-", " ")
        m_sub = re.match(r"(?:subtract|take)\s+([a-z0-9\s]+?)\s+(?:from|off)\s+([a-z0-9\s]+)", q_norm)
        if m_sub:
            n1 = cls.parse_number_phrase(m_sub.group(1).split())
            n2 = cls.parse_number_phrase(m_sub.group(2).split())
            if n1 is not None and n2 is not None:
                return f"{n2} - {n1}", None
            return None, "Failed to parse operands in subtraction"

        # 2. Pattern: 'sum of X and Y' / 'product of X and Y'
        m_of = re.match(r"(?:sum|total|product|quotient)\s+of\s+([a-z0-9\s]+?)\s+and\s+([a-z0-9\s]+)", q_norm)
        if m_of:
            op = "*" if "product" in q_norm else "/" if "quotient" in q_norm else "+"
            n1 = cls.parse_number_phrase(m_of.group(1).split())
            n2 = cls.parse_number_phrase(m_of.group(2).split())
            if n1 is not None and n2 is not None:
                return f"{n1} {op} {n2}", None
            return None, "Failed to parse operands in binary phrase"

        # 3. Pattern: 'find the quotient of X over Y'
        m_over = re.match(r".*quotient of\s+([a-z0-9\s]+?)\s+over\s+([a-z0-9\s]+)", q_norm)
        if m_over:
            n1 = cls.parse_number_phrase(m_over.group(1).split())
            n2 = cls.parse_number_phrase(m_over.group(2).split())
            if n1 is not None and n2 is not None:
                return f"{n1} / {n2}", None

        # 4. Pattern: 'X <operator> Y'
        op_pattern = r"\s+(times|multiplied by|divided by|over|plus|minus)\s+"
        parts = re.split(op_pattern, q_norm)
        if len(parts) == 3:
            left_words = re.sub(r"^(?:work out|calculate|what is|find|compute)\s+", "", parts[0]).split()
            op_word = parts[1]
            right_words = parts[2].replace("please", "").strip().split()

            n1 = cls.parse_number_phrase(left_words)
            n2 = cls.parse_number_phrase(right_words)
            op = OPERATORS.get(op_word)

            if n1 is not None and n2 is not None and op is not None:
                return f"{n1} {op} {n2}", None

        return None, "No supported spoken arithmetic pattern found"


def evaluate_spoken_parser_benchmark() -> dict:
    calc = SafeCalculator()
    test_cases = [
        ("twenty times fifteen", "20 * 15", 300),
        ("one hundred divided by four", "100 / 4", 25),
        ("fifty plus seventy-five", "50 + 75", 125),
        ("ninety minus thirty-five", "90 - 35", 55),
        ("subtract forty from ninety", "90 - 40", 50),
        ("sum of fifty and twenty-five", "50 + 25", 75),
        ("product of twelve and eight", "12 * 8", 96),
        ("two hundred over five", "200 / 5", 40),
        ("fourteen times six", "14 * 6", 84),
        ("eighty minus forty-five", "80 - 45", 35),
        ("what is thirty-three plus sixty-seven", "33 + 67", 100),
        ("compute sixty divided by five", "60 / 5", 12),
        ("take twenty from one hundred", "100 - 20", 80),
        ("fifteen multiplied by eight", "15 * 8", 120),
        ("find the quotient of one hundred forty-four over twelve", "144 / 12", 12),
    ]

    unsupported_cases = [
        "negative five times ten",      # Negative numbers not supported
        "three point five plus two",    # Decimals unsupported
        "write a poem about math",      # Non-math query
        "who won the championship",     # Out-of-scope query
        "calculate: '; DROP TABLE'",    # Injection
    ]

    parsed_correct = 0
    calc_correct = 0

    for query, exp_expr, exp_val in test_cases:
        expr, err = SpokenNumberParser.parse_spoken_expression(query)
        if expr == exp_expr:
            parsed_correct += 1
            res = calc.evaluate(expr)
            if res.success and res.result == exp_val:
                calc_correct += 1

    rejected_correct = 0
    for query in unsupported_cases:
        expr, err = SpokenNumberParser.parse_spoken_expression(query)
        if expr is None:
            rejected_correct += 1

    return {
        "total_supported": len(test_cases),
        "parser_accuracy": round(parsed_correct / len(test_cases), 4),
        "safe_calc_accuracy": round(calc_correct / len(test_cases), 4),
        "total_unsupported": len(unsupported_cases),
        "rejection_accuracy": round(rejected_correct / len(unsupported_cases), 4)
    }


if __name__ == "__main__":
    res = evaluate_spoken_parser_benchmark()
    print("Spoken Number Parser Benchmark Results:")
    print(json.dumps(res, indent=2))
