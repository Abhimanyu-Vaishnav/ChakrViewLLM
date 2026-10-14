"""
Batch 254: Founder Console End-to-End Evaluation Runner.
Location: experiments/batch251_260_interactive_runtime/run_console_evaluation.py

Runs the real interactive entrypoint (scripts/founder_manual_test.py / ChakrViewRuntimeSession)
across the 21 frozen regression cases, plus 9 extended failure/paraphrase archetypes (30 total):
1. Arithmetic variants
2. Constants & mathematical formulas
3. Core architecture definitions
4. Code synthesis & AST verification
5. Multi-turn discourse continuity & coreference
6. False premise & honest abstention
7. Hindi & Hinglish multilingual comprehension
8. Isolated profile memory store & recall

Produces: reports/interactive_console_results.json
"""
from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, ".")

from scripts.founder_manual_test import ChakrViewRuntimeSession

SUITE_PATH = Path("experiments/batch241_250_interactive_integration/data/frozen_interactive_regression_suite.json")
REPORTS_DIR = Path("experiments/batch251_260_interactive_runtime/reports")


def evaluate_python_code(code_str: str) -> bool:
    clean = code_str.replace("```python", "").replace("```", "").strip()
    try:
        ast.parse(clean)
        return True
    except SyntaxError:
        return False


def main():
    random.seed(42)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    base_suite = json.loads(SUITE_PATH.read_text(encoding="utf-8"))

    # Add 9 extended paraphrase and negative control cases
    extended_cases = [
        {
            "id": "ext01_arithmetic_product",
            "query": "What is 15 * 8?",
            "expected_answer": "120",
            "validation_mode": "contains",
            "category": "arithmetic_extended"
        },
        {
            "id": "ext02_euler_constant",
            "query": "What is the value of e?",
            "expected_answer": "2.718",
            "validation_mode": "contains",
            "category": "constants_extended"
        },
        {
            "id": "ext03_ram_definition",
            "query": "Explain RAM in simple words.",
            "expected_answer": "memory",
            "validation_mode": "contains",
            "category": "concept_definition_extended"
        },
        {
            "id": "ext04_python_subtraction",
            "query": "Write a Python script to subtract two numbers.",
            "expected_answer": "def",
            "validation_mode": "runnable_python",
            "category": "code_generation_extended"
        },
        {
            "id": "ext05_false_premise_sun",
            "query": "What is the capital of the Sun?",
            "expected_answer": "does not have",
            "validation_mode": "abstention_or_correction",
            "category": "false_premise_extended"
        },
        {
            "id": "ext06_hinglish_water_cycle",
            "query": "Mujhe simple Hindi mein evaporation samjhao.",
            "expected_answer": "वाष्पीकरण",
            "validation_mode": "contains",
            "category": "multilingual_extended"
        },
        {
            "id": "ext07_coreference_step1",
            "query": "What is a CPU?",
            "expected_answer": "processor",
            "validation_mode": "contains",
            "category": "coreference_extended"
        },
        {
            "id": "ext08_coreference_step2",
            "query": "and What about ALU",
            "expected_answer": "arithmetic logic unit",
            "validation_mode": "contains",
            "category": "coreference_extended"
        },
        {
            "id": "ext09_unsupported_quantum",
            "query": "Predict tomorrow's lottery numbers with quantum certainty.",
            "expected_answer": "do not have sufficient",
            "validation_mode": "abstention_or_correction",
            "category": "honest_abstention_negative_control"
        }
    ]

    all_test_cases = base_suite + extended_cases

    # Initialize Real Founder Console Session
    session = ChakrViewRuntimeSession()

    results = []
    passed_count = 0
    total = len(all_test_cases)

    print(f"Executing Real Founder Console Evaluation ({total} test cases)...")

    for item in all_test_cases:
        item_id = item["id"]
        query = item["query"]
        exp_ans = item["expected_answer"]
        val_mode = item["validation_mode"]
        cat = item["category"]

        t0 = time.perf_counter()
        resp_payload = session.process_input(query)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        ans_clean = (resp_payload.get("final_response") or resp_payload.get("natural_response") or "").strip()
        route = resp_payload.get("intent", "UNKNOWN")
        origin = resp_payload.get("component_origin") or resp_payload.get("strategy", "UNKNOWN")

        is_pass = False
        notes = ""

        if val_mode == "contains":
            is_pass = exp_ans.lower() in ans_clean.lower()
        elif val_mode == "runnable_python":
            is_pass = evaluate_python_code(ans_clean)
            notes = "AST parsed successfully"
        elif val_mode == "abstention_or_correction":
            keywords = ["does not have", "no capital", "not have a capital", "uninhabited", "insufficient", "do not have sufficient", "not mentioned", "honest abstention", "do not have verified"]
            is_pass = any(k in ans_clean.lower() for k in keywords)
            notes = "Honest abstention or false premise correction"
        elif val_mode == "exact_recall":
            is_pass = exp_ans.lower() in ans_clean.lower()
            notes = "Memory recall verified"

        if is_pass:
            passed_count += 1

        results.append({
            "id": item_id,
            "query": query,
            "category": cat,
            "route": route,
            "component_origin": origin,
            "expected_answer": exp_ans,
            "generated_answer": ans_clean,
            "passed": is_pass,
            "validation_mode": val_mode,
            "latency_ms": round(elapsed_ms, 2),
            "notes": notes,
        })

        status_str = "PASS" if is_pass else "FAIL"
        print(f"[{status_str}] {item_id:<34} | Route: {route:<18} | Origin: {origin:<22} | {elapsed_ms:>6.2f}ms")

    pass_rate = round(passed_count / total, 4)
    print(f"\nFounder Console Results: {passed_count}/{total} Passed ({pass_rate*100:.1f}%)")

    report_payload = {
        "milestone": "Batch 254: Actual Founder Console End-to-End Evaluation",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "total": total,
        "passed": passed_count,
        "pass_rate": pass_rate,
        "results": results
    }

    out_file = REPORTS_DIR / "interactive_console_results.json"
    out_file.write_text(json.dumps(report_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved console evaluation report to: {out_file}")


if __name__ == "__main__":
    main()
