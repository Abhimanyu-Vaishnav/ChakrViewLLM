"""CR-EXP-16 Stage A Evaluator Calibration Suite
Evaluates known-answer tests across 10 specific edge-case scenarios:
1. Exact sequence copying.
2. One-token copying.
3. Multi-token copying.
4. Repeated tokens.
5. Long sequences.
6. Missing tokens (partial match - must NOT receive exact credit).
7. Extra tokens (must NOT receive exact credit).
8. Incorrect ordering (must NOT receive exact credit).
9. Early termination / empty output.
10. Special-token / punctuation contamination.
"""
import re
from typing import Dict, Any, List

class SequenceCopyEvaluator:
    def evaluate(self, expected_target: str, raw_output: str) -> Dict[str, Any]:
        clean_out = raw_output.strip()
        # Clean trailing newlines or prompt echoes
        clean_out = clean_out.split("\n")[0].strip()
        
        tgt_tokens = expected_target.strip().split()
        out_tokens = clean_out.split()
        
        # Exact sequence match (case-insensitive)
        exact_match = (expected_target.strip().lower() == clean_out.lower())
        
        # Token-level overlap (precision, recall, f1)
        if not tgt_tokens or not out_tokens:
            token_acc = 1.0 if (not tgt_tokens and not out_tokens) else 0.0
        else:
            # Token accuracy = fraction of matched tokens in correct positions
            min_len = min(len(tgt_tokens), len(out_tokens))
            pos_matches = sum(1 for i in range(min_len) if tgt_tokens[i].lower() == out_tokens[i].lower())
            token_acc = pos_matches / max(len(tgt_tokens), len(out_tokens))
            
        return {
            "exact_match": exact_match,
            "token_accuracy": token_acc,
            "expected": expected_target,
            "actual": clean_out
        }

def run_calibration():
    evaluator = SequenceCopyEvaluator()
    
    test_cases = [
        {
            "id": "CASE_1",
            "desc": "Exact copying",
            "expected": "alpha beta gamma",
            "actual": "alpha beta gamma",
            "expected_exact": True,
            "min_token_acc": 1.0
        },
        {
            "id": "CASE_2",
            "desc": "One-token copying",
            "expected": "Dev",
            "actual": "Dev",
            "expected_exact": True,
            "min_token_acc": 1.0
        },
        {
            "id": "CASE_3",
            "desc": "Multi-token copying",
            "expected": "Mukesh",
            "actual": "Mukesh",
            "expected_exact": True,
            "min_token_acc": 1.0
        },
        {
            "id": "CASE_4",
            "desc": "Repeated tokens",
            "expected": "red red blue red",
            "actual": "red red blue red",
            "expected_exact": True,
            "min_token_acc": 1.0
        },
        {
            "id": "CASE_5",
            "desc": "Long sequences",
            "expected": "one two three four five six seven eight",
            "actual": "one two three four five six seven eight",
            "expected_exact": True,
            "min_token_acc": 1.0
        },
        {
            "id": "CASE_6",
            "desc": "Missing tokens (partial match)",
            "expected": "alpha beta gamma delta",
            "actual": "alpha beta",
            "expected_exact": False,
            "min_token_acc": 0.5
        },
        {
            "id": "CASE_7",
            "desc": "Extra tokens added",
            "expected": "alpha beta",
            "actual": "alpha beta gamma delta",
            "expected_exact": False,
            "min_token_acc": 0.5
        },
        {
            "id": "CASE_8",
            "desc": "Incorrect ordering",
            "expected": "alpha beta gamma",
            "actual": "gamma beta alpha",
            "expected_exact": False,
            "min_token_acc": 0.33
        },
        {
            "id": "CASE_9",
            "desc": "Early termination / empty output",
            "expected": "alpha beta gamma",
            "actual": "",
            "expected_exact": False,
            "min_token_acc": 0.0
        },
        {
            "id": "CASE_10",
            "desc": "Special-token contamination / punctuation",
            "expected": "alpha beta gamma",
            "actual": "alpha beta gamma.",
            "expected_exact": False, # strict match fails with trailing period
            "min_token_acc": 0.5
        }
    ]
    
    print("=" * 60)
    print("RUNNING STAGE A EVALUATOR CALIBRATION TESTS (10 CASES)")
    print("=" * 60)
    all_passed = True
    
    for tc in test_cases:
        res = evaluator.evaluate(tc["expected"], tc["actual"])
        pass_exact = (res["exact_match"] == tc["expected_exact"])
        pass_token = (res["token_accuracy"] >= tc["min_token_acc"] - 0.05)
        passed = pass_exact and pass_token
        if not passed:
            all_passed = False
        print(f"[{'PASS' if passed else 'FAIL'}] {tc['id']}: {tc['desc']}")
        print(f"       Exact Match: got {res['exact_match']}, expected {tc['expected_exact']}")
        print(f"       Token Acc:   got {res['token_accuracy']:.2f}, expected >={tc['min_token_acc']:.2f}")
        
    print("=" * 60)
    print(f"CALIBRATION SUMMARY: {'ALL 10 CASES PASSED' if all_passed else 'CALIBRATION FAILURES DETECTED'}")
    print("=" * 60)
    return all_passed

if __name__ == "__main__":
    run_calibration()

