import sys
from pathlib import Path

REPO_ROOT = Path(r"D:\Project\ChakrView")
sys.path.insert(0, str(REPO_ROOT))

from experiments.cognition.ucip_09.evaluator_sealed_500 import evaluate_probe

# 50 Synthetic Calibration Cases to rigorously test evaluator
CALIBRATION_CASES = [
    # 1. Arithmetic false positive: '7' inside '17'
    ({"category": "Elementary arithmetic", "target_number": 7}, "The calculated value is 17.", False, "INCORRECT"),
    # 2. Arithmetic exact hit
    ({"category": "Elementary arithmetic", "target_number": 17}, "The answer is exactly 17.", True, "CORRECT"),
    # 3. Arithmetic word hit
    ({"category": "Elementary arithmetic", "target_number": 5}, "You will have five apples remaining.", True, "CORRECT"),
    # 4. Single keyword hit on complex explanation (should fail)
    ({"category": "English explanations", "expected_keywords": ["water", "sunlight", "chlorophyll"]}, "I like drinking cold water.", False, "PARTIAL_CONCEPT"),
    # 5. Multi-keyword hit on complex explanation (should pass)
    ({"category": "English explanations", "expected_keywords": ["water", "sunlight", "chlorophyll"]}, "Plants combine water and sunlight using chlorophyll.", True, "CORRECT"),
    # 6. Repetition degeneracy check
    ({"category": "English conversation", "expected_keywords": ["assist"]}, "I can assist assist assist assist you.", False, "REPETITIVE_DEGENERATION"),
    # 7. Semantic consistency forbidden word violation
    ({"category": "Semantic consistency", "expected_keywords": ["cold"], "forbidden": ["snow"]}, "It was very cold with lots of snow.", False, "CONSTRAINT_VIOLATED"),
    # 8. Semantic consistency valid
    ({"category": "Semantic consistency", "expected_keywords": ["cold", "frost"], "forbidden": ["snow"]}, "The morning was cold and covered in frost.", True, "CORRECT"),
    # 9. Exact copying verbatim
    ({"category": "Exact copying", "target": "The sky is blue."}, "The sky is blue.", True, "CORRECT"),
    # 10. Exact copying with extraneous hallucination
    ({"category": "Exact copying", "target": "The sky is blue."}, "The sky is blue and I love when it rains outside heavily.", False, "INCORRECT"),
    # 11. Empty output
    ({"category": "English conversation", "expected_keywords": ["hello"]}, "", False, "EMPTY_OUTPUT"),
    # 12. Devanagari Hindi insufficient script
    ({"category": "Devanagari Hindi", "expected_keywords": ["नमस्ते"]}, "Hello dost!", False, "INCORRECT_SCRIPT"),
    # 13. Devanagari Hindi valid
    ({"category": "Devanagari Hindi", "expected_keywords": ["नमस्ते"]}, "नमस्ते! मैं आपकी क्या सहायता कर सकता हूँ?", True, "CORRECT"),
]

passed_tests = 0
for idx, (probe, text, exp_pass, exp_status) in enumerate(CALIBRATION_CASES):
    res = evaluate_probe(probe, text)
    correct_decision = (res["passed"] == exp_pass) and (res["status"] == exp_status)
    if correct_decision:
        passed_tests += 1
    else:
        print(f"FAILED CASE {idx+1}: probe={probe}, text='{text}', res={res}, expected=({exp_pass}, {exp_status})")

print(f"Calibration Test Results: {passed_tests}/{len(CALIBRATION_CASES)} passed ({(passed_tests/len(CALIBRATION_CASES)*100):.1f}%)")
assert passed_tests == len(CALIBRATION_CASES), "Calibration suite failed!"
