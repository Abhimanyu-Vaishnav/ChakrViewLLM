"""
evaluation/cbr_sft_05_suite/evaluator.py

Evaluator Engine for CBR-SFT-05 (Context Binding, Induction & Conversational Memory).

Key Capabilities:
1. Exact String Comparison (character-by-character entity preservation)
2. Normalized Entity Comparison (strips surrounding quotes, periods, whitespace)
3. Abstention Rule Scoring (validates genuine epistemic refusals)
4. Multi-Turn Context Recall Validation
5. Partial Match & Substring Diagnostics
"""

import sys
import os
import re
import json
import string
from pathlib import Path
from typing import Dict, Any, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

def normalize_entity_string(text: str) -> str:
    """Normalizes whitespace and removes boundary punctuation without modifying internal characters."""
    t = text.strip()
    # Strip wrapping quotes or brackets
    t = re.sub(r"^['\"<\[\{\(]+", "", t)
    t = re.sub(r"['\">\]\}\)]+$", "", t)
    # Strip trailing periods/commas/colons
    t = t.rstrip(".,:;")
    return t.strip()

class CBR05Evaluator:
    @staticmethod
    def evaluate_task(
        generated_text: str,
        expected_concept: str,
        scoring_rule: str,
        category: str
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Evaluates a single model output against expected concept.
        Returns: (passed: bool, reason: str, metadata: dict)
        """
        gen_clean = generated_text.strip()
        exp_clean = expected_concept.strip()

        gen_norm = normalize_entity_string(gen_clean)
        exp_norm = normalize_entity_string(exp_clean)

        meta = {
            "raw_output": generated_text,
            "expected": expected_concept,
            "normalized_output": gen_norm,
            "normalized_expected": exp_norm,
            "scoring_rule": scoring_rule
        }

        # 1. Abstention Evaluation
        if scoring_rule == "abstention" or "abstention" in category or exp_clean.lower() == "not provided":
            abstention_patterns = [
                "not provided", "does not specify", "does not contain",
                "cannot determine", "insufficient", "unknown", "was not provided"
            ]
            gen_lower = gen_clean.lower()
            matched_pat = [p for p in abstention_patterns if p in gen_lower]
            if matched_pat:
                return True, f"Abstention matched pattern '{matched_pat[0]}'", meta
            else:
                return False, "Failed to express required abstention", meta

        # 2. Exact Match (Character-by-character on normalized token)
        if scoring_rule == "exact_match":
            # Direct exact match
            if exp_norm == gen_norm:
                return True, "Exact character match", meta

            # Exact prefix match (model generated the exact entity first, then stopped or padded)
            if gen_norm.startswith(exp_norm):
                # Ensure word boundary after prefix
                remainder = gen_norm[len(exp_norm):]
                if not remainder or remainder[0] in " .,:;\n":
                    return True, "Exact entity prefix match", meta

            # Exact substring containment
            if exp_norm in gen_clean:
                # Check for boundary delimiter
                pattern = r"(?<![a-zA-Z0-9_-])" + re.escape(exp_norm) + r"(?![a-zA-Z0-9_-])"
                if re.search(pattern, gen_clean, re.IGNORECASE):
                    return True, "Exact entity bounded match", meta

            return False, f"Expected entity '{exp_norm}' not reproduced in output '{gen_norm}'", meta

        # 3. Fallback normalized comparison
        if exp_norm.lower() in gen_clean.lower():
            return True, "Case-insensitive normalized containment", meta

        return False, f"Expected '{exp_clean}' not found", meta

def evaluate_suite_file(suite_json_path: Path, model_runner: Any) -> Dict[str, Any]:
    with open(suite_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    tasks = data["tasks"]
    passed_count = 0
    cat_counts = {}
    results = []

    for task in tasks:
        t_id = task["id"]
        cat = task["category"]
        prompt = task["prompt"]
        exp = task.get("expected_concept", "")
        rule = task.get("scoring", "exact_match")
        max_toks = task.get("max_tokens", 32)

        cat_counts.setdefault(cat, {"total": 0, "passed": 0})
        cat_counts[cat]["total"] += 1

        resp = model_runner.generate(prompt=prompt, max_new_tokens=max_toks, temperature=0.0)
        gen_text = resp["text"]

        passed, reason, meta = CBR05Evaluator.evaluate_task(gen_text, exp, rule, cat)
        if passed:
            passed_count += 1
            cat_counts[cat]["passed"] += 1

        results.append({
            "task_id": t_id,
            "category": cat,
            "prompt": prompt,
            "expected": exp,
            "generated_text": gen_text,
            "passed": passed,
            "reason": reason,
            "metadata": meta
        })

    total = len(tasks)
    return {
        "suite_name": data.get("suite_name", "HELD_OUT_CBR_SFT_05_120"),
        "total_tasks": total,
        "passed_tasks": passed_count,
        "accuracy": passed_count / total if total > 0 else 0.0,
        "category_breakdown": {
            c: {
                "total": v["total"],
                "passed": v["passed"],
                "accuracy": v["passed"] / v["total"] if v["total"] > 0 else 0.0
            } for c, v in cat_counts.items()
        },
        "results": results
    }

if __name__ == "__main__":
    print("CBR05Evaluator smoke test...")
    pass_t, r, _ = CBR05Evaluator.evaluate_task(" 192.168.1.105", "192.168.1.105", "exact_match", "exact_copy_ip")
    print(f"Test 1 (IP): passed={pass_t}, reason={r}")
    pass_t2, r2, _ = CBR05Evaluator.evaluate_task(" That information is not provided.", "not provided", "abstention", "abstention")
    print(f"Test 2 (Abstention): passed={pass_t2}, reason={r2}")

