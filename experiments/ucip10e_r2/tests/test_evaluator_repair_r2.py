import pytest
from experiments.ucip10e.evaluators.strict_evaluators import StrictEvaluators
from experiments.ucip10e_r2.evaluators.strict_evaluators_r2 import StrictEvaluatorsR2

class TestEvaluatorRepairR2:

    def test_r1_vulnerability_reproduced_and_repaired(self):
        # The exact adversarial failure from R1
        probe = {"subdomain": "english", "expected_keywords": ["memory", "ram", "storage"]}
        payload = "ram storage memory ram storage memory ram storage memory"
        
        # 1. Show that original R1 evaluator is vulnerable (passes)
        r1_result = StrictEvaluators.evaluate_language(probe, payload)
        assert r1_result["passed"], "Original R1 evaluator was vulnerable to cyclic word-salad!"
        
        # 2. Show that R2 evaluator repairs the vulnerability (strictly rejects)
        r2_result = StrictEvaluatorsR2.evaluate_language_r2(probe, payload)
        assert not r2_result["passed"], "R2 evaluator must strictly reject cyclic word-salad!"
        assert "DEGENERATE_REPETITION" in r2_result["reason"]

    def test_r2_allows_legitimate_conversational_repetition(self):
        # Normal repetition across distinct sentences should be allowed
        probe = {"subdomain": "english", "expected_keywords": ["ram", "storage"]}
        valid_text = "RAM is primary memory. Secondary storage is different from RAM because RAM is volatile."
        res = StrictEvaluatorsR2.evaluate_language_r2(probe, valid_text)
        assert res["passed"]

    def test_r2_arithmetic_operand_rejection(self):
        probe = {"target_number": 3}
        assert not StrictEvaluatorsR2.evaluate_arithmetic_r2(probe, "5 - 3 = 4.")["passed"]
        assert StrictEvaluatorsR2.evaluate_arithmetic_r2(probe, "5 - 2 = 3.")["passed"]
