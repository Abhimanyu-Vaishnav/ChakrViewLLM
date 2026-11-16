"""
Unit tests for UCIP-03R Independent Evaluator.
Verifies that:
- Incoherent word salad is REJECTED (FAIL)
- Meaningful responses are ACCEPTED (PASS)
- Script detection and semantic matching work accurately.
"""

import sys
from pathlib import Path
REPO_ROOT = Path("D:/Project/ChakrView")
sys.path.insert(0, str(REPO_ROOT))

from experiments.cognition.ucip_03r.independent_evaluation.independent_evaluator import evaluate_response

def run_unit_tests():
    test_cases = [
        # Incoherent outputs observed by owner must FAIL
        {
            "category": "copying",
            "prompt": "User: Say exactly: The sky is blue.\nAssistant:",
            "gen": "The planned of the bold.",
            "target": "The sky is blue.",
            "expected_verdict": "FAIL"
        },
        {
            "category": "english",
            "prompt": "User: What does a CPU do?\nAssistant:",
            "gen": "The planet is the bold.",
            "target": "CPU",
            "expected_verdict": "FAIL"
        },
        {
            "category": "english",
            "prompt": "User: Who are you?\nAssistant:",
            "gen": "A real is the bold.",
            "target": "ChakrView",
            "expected_verdict": "FAIL"
        },
        {
            "category": "devanagari_hindi",
            "prompt": "User: नमस्ते! आप क्या कर सकते हैं?\nAssistant:",
            "gen": "Your name is Aarav.",
            "target": "मदद",
            "expected_verdict": "FAIL"
        },
        {
            "category": "hinglish",
            "prompt": "User: Bhai, mujhe machine learning simple language mein samjhao.\nAssistant:",
            "gen": "makind plangnornever is simp.",
            "target": "machine learning",
            "expected_verdict": "FAIL"
        },
        # Correct samples must PASS
        {
            "category": "copying",
            "prompt": "User: Say exactly: The sky is blue.\nAssistant:",
            "gen": "The sky is blue.",
            "target": "The sky is blue.",
            "expected_verdict": "PASS"
        },
        {
            "category": "arithmetic",
            "prompt": "User: What is 2 + 2?\nAssistant:",
            "gen": "2 + 2 is 4.",
            "target": "4",
            "expected_verdict": "PASS"
        },
        {
            "category": "arithmetic",
            "prompt": "User: What is 2 + 2?\nAssistant:",
            "gen": "2 + 2 is 5.",
            "target": "4",
            "expected_verdict": "FAIL"
        },
        {
            "category": "devanagari_hindi",
            "prompt": "User: नमस्ते, आप कैसे हैं?\nAssistant:",
            "gen": "नमस्ते! मैं आपकी सहायता के लिए तैयार हूँ।",
            "target": "नमस्ते",
            "expected_verdict": "PASS"
        },
        {
            "category": "roman_hindi",
            "prompt": "User: Aapka naam kya hai?\nAssistant:",
            "gen": "Mera naam ChakrView hai, main aapki madad kar sakta hoon.",
            "target": "ChakrView",
            "expected_verdict": "PASS"
        },
        {
            "category": "hinglish",
            "prompt": "User: Machine learning kya hai?\nAssistant:",
            "gen": "Machine learning ek branch hai jisme models data se learn karte hain.",
            "target": "models",
            "expected_verdict": "PASS"
        },
        # Repetition loop must FAIL
        {
            "category": "english",
            "prompt": "User: Hello\nAssistant:",
            "gen": "the problem is the problem is the problem is the problem is",
            "target": "",
            "expected_verdict": "FAIL"
        },
        # Empty string must be UNSCORABLE
        {
            "category": "english",
            "prompt": "User: Hello\nAssistant:",
            "gen": "",
            "target": "",
            "expected_verdict": "UNSCORABLE"
        }
    ]

    passed = 0
    for idx, tc in enumerate(test_cases, 1):
        res = evaluate_response(tc["category"], tc["prompt"], tc["gen"], tc["target"])
        match = (res["verdict"] == tc["expected_verdict"])
        status = "PASS" if match else "FAIL"
        print(f"[{status}] Test {idx:02d} ({tc['category']}): Expected {tc['expected_verdict']}, Got {res['verdict']} - Reason: {res['reasoning']}")
        if match:
            passed += 1

    print(f"\nUnit test summary: {passed}/{len(test_cases)} passed.")
    assert passed == len(test_cases), f"Failed {len(test_cases) - passed} tests!"

if __name__ == "__main__":
    run_unit_tests()
