"""
Batch 1781-1820: Work Package 7 Spoken-Number Parser Expanded Regression Suite.
Tests units, tens, compounds, hundreds, multiple operators, ambiguous phrasing,
malformed expressions, negatives, decimals, division by zero, and prompt injection strings.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.batch1741_1780.spoken_number_parser import SpokenNumberParser
from chakrview.runtime.safe_calculator import SafeCalculator


def run_spoken_number_regression() -> dict:
    calc = SafeCalculator()

    # Category 1: Units, Tens, Compound, Hundreds with Supported Operators
    valid_cases = [
        # Units
        ("five plus four", "5 + 4", 9.0),
        ("nine minus two", "9 - 2", 7.0),
        ("three times six", "3 * 6", 18.0),
        ("eight divided by two", "8 / 2", 4.0),
        # Tens & Compound Numbers
        ("twenty five times four", "25 * 4", 100.0),
        ("ninety nine minus thirty three", "99 - 33", 66.0),
        ("seventy plus twenty eight", "70 + 28", 98.0),
        ("forty eight divided by six", "48 / 6", 8.0),
        # Hundreds
        ("one hundred times five", "100 * 5", 500.0),
        ("two hundred fifty plus seventy five", "250 + 75", 325.0),
        ("nine hundred minus four hundred fifty", "900 - 450", 450.0),
        ("six hundred divided by twenty", "600 / 20", 30.0),
        # Phrase Patterns
        ("sum of fifty and twenty five", "50 + 25", 75.0),
        ("product of twelve and eight", "12 * 8", 96.0),
        ("subtract forty from ninety", "90 - 40", 50.0),
        ("take twenty from one hundred", "100 - 20", 80.0),
        ("fifteen multiplied by eight", "15 * 8", 120.0),
        ("fourteen over two", "14 / 2", 7.0),
    ]

    # Category 2: Ambiguous, Malformed, Unsupported Negatives, Decimals, Injections & Division by Zero
    rejection_cases = [
        ("negative fifteen plus eight", "unsupported_negative_word"),
        ("three point five times two", "unsupported_decimal"),
        ("twenty point five plus one", "unsupported_decimal"),
        ("divide five by zero", "malformed_or_zero_division"),
        ("hundred divided by zero", "zero_division"),
        ("what is love", "non_math_conversational"),
        ("calculate __import__('os').system('id')", "code_injection"),
        ("'; DROP TABLE calculations; --", "sql_injection"),
        ("plus plus minus", "malformed_operators"),
        ("twenty times", "missing_second_operand"),
        ("times fifteen", "missing_first_operand"),
    ]

    results = {
        "valid_cases_total": len(valid_cases),
        "valid_cases_passed": 0,
        "rejection_cases_total": len(rejection_cases),
        "rejection_cases_passed": 0,
        "failures": []
    }

    # Evaluate Valid Cases
    for query, expected_expr, expected_result in valid_cases:
        expr, err = SpokenNumberParser.parse_spoken_expression(query)
        if err is not None:
            results["failures"].append({"query": query, "type": "valid_parse_failed", "error": err})
            continue
        if expr != expected_expr:
            results["failures"].append({
                "query": query, "type": "expression_mismatch", "expected": expected_expr, "got": expr
            })
            continue
        res = calc.evaluate(expr)
        if not res.success or res.result != expected_result:
            results["failures"].append({
                "query": query, "type": "calculation_mismatch", "expected": expected_result, "got": res.result
            })
            continue
        results["valid_cases_passed"] += 1

    # Evaluate Rejection Cases
    for query, reason in rejection_cases:
        expr, err = SpokenNumberParser.parse_spoken_expression(query)
        if expr is not None:
            # Check if calc safely fails or if it was inappropriately parsed
            res = calc.evaluate(expr)
            if res.success and "zero" in query:
                results["failures"].append({"query": query, "type": "zero_division_not_blocked", "reason": reason})
                continue
            if res.success:
                results["failures"].append({"query": query, "type": "unauthorized_acceptance", "reason": reason})
                continue
        results["rejection_cases_passed"] += 1

    results["valid_accuracy"] = round(results["valid_cases_passed"] / len(valid_cases), 4)
    results["rejection_accuracy"] = round(results["rejection_cases_passed"] / len(rejection_cases), 4)

    out_file = "experiments/batch1781_1820/results/SPOKEN_NUMBER_REGRESSION_RESULTS.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Spoken Number Regression Suite Completed:")
    print(f"  Valid Cases: {results['valid_cases_passed']}/{results['valid_cases_total']} ({results['valid_accuracy']*100:.1f}%)")
    print(f"  Rejection Cases: {results['rejection_cases_passed']}/{results['rejection_cases_total']} ({results['rejection_accuracy']*100:.1f}%)")
    return results


if __name__ == "__main__":
    run_spoken_number_regression()
