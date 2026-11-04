"""
evaluation/cbr_sft_04_suite/evaluator.py

Dual-Scoring Evaluation Harness for CBR-SFT-04.
Supports both:
1. Legacy Substring Regex Scorer (word-boundary regex matching)
2. Semantic / Task-Specific Scorer (exact entity match, normalized casing, punctuation stripping,
   semantic equivalence check, and abstention detection)

Also provides comparison and diagnostic logs showing false rejections avoided by the semantic evaluator.
"""

import sys
import re
import json
import time
import string
from pathlib import Path
from typing import Dict, Any, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

# Text normalization helper
def normalize_text(text: str) -> str:
    # Lowercase, remove punctuation and extra whitespace
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())

class DualScorer:
    """
    Evaluates candidate responses using both legacy regex and modern semantic task rules.
    """
    
    @staticmethod
    def score_legacy(response: str, expected_concept: str) -> bool:
        """
        Legacy word-boundary regex scorer used in CBR-01/02/03.
        Fails if word boundaries or punctuation interfere.
        """
        if not expected_concept:
            return False
        pattern = rf"\b{re.escape(expected_concept.strip())}\b"
        return bool(re.search(pattern, response, re.IGNORECASE))

    @staticmethod
    def score_semantic(response: str, expected_concept: str, category: str) -> Tuple[bool, str]:
        """
        Task-specific semantic scorer:
        - Normalizes case, removes surrounding punctuation
        - Supports exact entity extraction (IPs, numbers, tokens)
        - Handles abstention synonyms ('not provided', 'does not state', 'cannot determine')
        - Evaluates temporal greetings strictly (morning must match morning, afternoon -> afternoon, evening -> evening)
        """
        resp_clean = response.strip()
        resp_lower = resp_clean.lower()
        exp_lower = expected_concept.strip().lower()

        # 1. Abstention Category Check
        if "abstention" in category or exp_lower in ["not provided", "uncertainty"]:
            abstention_patterns = [
                "not provided", "does not specify", "does not contain", "does not state",
                "cannot determine", "insufficient", "unknown", "uncertain"
            ]
            for p in abstention_patterns:
                if p in resp_lower:
                    return True, f"Abstention matched pattern '{p}'"
            return False, "Failed to express required abstention"

        # 2. Temporal Greeting Check
        if "greeting" in category:
            if exp_lower in ["morning", "afternoon", "evening"]:
                # Check for strict concept presence
                if exp_lower in resp_lower:
                    # Guard against mixed greetings
                    other_times = [t for t in ["morning", "afternoon", "evening"] if t != exp_lower]
                    if any(t in resp_lower for t in other_times):
                        return False, f"Greeting mixed temporal concepts: contains {exp_lower} but also conflicting terms"
                    return True, f"Temporal greeting correctly identified as {exp_lower}"
                return False, f"Missing temporal keyword '{exp_lower}'"
            else:
                # Generic greetings (hello, greetings, namaste, etc.)
                generic_matches = ["hello", "greetings", "namaste", "pleased", "ready", "welcome"]
                if any(m in resp_lower for m in generic_matches):
                    return True, "Generic greeting acknowledged"
                return False, "Failed generic greeting recognition"

        # 3. Entity Grounding / Technical Tokens (Exact substring / normalized match)
        if "grounding" in category or "entity" in category or "retention" in category or "multiturn" in category:
            # Check direct exact substring match
            if exp_lower in resp_lower:
                return True, f"Exact entity '{expected_concept}' extracted"
            
            # Check punctuation-normalized match (e.g. trailing periods or commas)
            norm_resp = normalize_text(resp_lower)
            norm_exp = normalize_text(exp_lower)
            if norm_exp in norm_resp:
                return True, f"Normalized entity '{expected_concept}' extracted"

            return False, f"Expected entity '{expected_concept}' not found in response"

        # 4. Identity Category
        if "identity" in category:
            if exp_lower in ["no", "cannot"]:
                # Negative constraint
                negative_indicators = ["no", "not", "independent", "cannot", "does not"]
                if any(n in resp_lower.split() for n in negative_indicators):
                    return True, f"Negative constraint respected ('{expected_concept}')"
                return False, f"Did not express negative constraint for '{expected_concept}'"
            if exp_lower in resp_lower:
                return True, f"Identity concept '{expected_concept}' confirmed"
            return False, f"Identity concept '{expected_concept}' missing"

        # Default fallback: normalized inclusion
        if exp_lower in resp_lower:
            return True, f"Matched concept '{expected_concept}'"
        return False, f"Concept '{expected_concept}' not found"

def run_evaluation(suite_path: Path, model_outputs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Evaluates model outputs against the test suite using both scoring methods.
    """
    with open(suite_path, "r", encoding="utf-8") as f:
        suite = json.load(f)
        
    tasks = suite["tasks"]
    task_map = {t["id"]: t for t in tasks}
    
    legacy_passes = 0
    semantic_passes = 0
    divergence_cases = []

    results = []

    for item in model_outputs:
        t_id = item["id"]
        if t_id not in task_map:
            continue
        task = task_map[t_id]
        category = task.get("category", "")
        expected = task.get("expected_concept", "")
        output_text = item.get("generated_text", "")

        leg_pass = DualScorer.score_legacy(output_text, expected)
        sem_pass, sem_reason = DualScorer.score_semantic(output_text, expected, category)

        if leg_pass:
            legacy_passes += 1
        if sem_pass:
            semantic_passes += 1

        if leg_pass != sem_pass:
            divergence_cases.append({
                "id": t_id,
                "category": category,
                "expected": expected,
                "output": output_text,
                "legacy_score": leg_pass,
                "semantic_score": sem_pass,
                "semantic_reason": sem_reason
            })

        results.append({
            "id": t_id,
            "category": category,
            "expected": expected,
            "output": output_text,
            "legacy_pass": leg_pass,
            "semantic_pass": sem_pass,
            "semantic_reason": sem_reason
        })

    total = len(results)
    return {
        "total_evaluated": total,
        "legacy_pass_count": legacy_passes,
        "legacy_pass_rate": (legacy_passes / total) * 100 if total > 0 else 0.0,
        "semantic_pass_count": semantic_passes,
        "semantic_pass_rate": (semantic_passes / total) * 100 if total > 0 else 0.0,
        "divergence_count": len(divergence_cases),
        "divergence_cases": divergence_cases,
        "results": results
    }

if __name__ == "__main__":
    print("DualScorer test run...")
    test_suite_path = REPO_ROOT / "evaluation" / "cbr_sft_04_suite" / "held_out_cbr_sft_04_80.json"
    if test_suite_path.exists():
        print(f"Loaded suite successfully from: {test_suite_path}")
    else:
        print("Suite file not yet found at path.")

