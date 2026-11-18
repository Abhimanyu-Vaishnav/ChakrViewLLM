import pytest
from experiments.ucip10e.evaluators.strict_evaluators import StrictEvaluators

class TestStrictEvaluatorsIntegrity:

    def test_arithmetic_rejects_operand_presence(self):
        # The critical bug from UCIP-09S: "5 - 3 = 4." for probe 5 - 2 = 3
        probe = {"target_number": 3}
        res = StrictEvaluators.evaluate_arithmetic(probe, "5 - 3 = 4.")
        assert not res["passed"], "Must reject when target 3 is merely an operand in a bogus equation!"

    def test_arithmetic_accepts_valid_equation(self):
        probe = {"target_number": 4}
        res = StrictEvaluators.evaluate_arithmetic(probe, "2 + 2 = 4.")
        assert res["passed"]
        assert res["extracted"] == 4

    def test_arithmetic_accepts_word_problem_phrasing(self):
        probe = {"target_number": 3}
        res = StrictEvaluators.evaluate_arithmetic(probe, "You will have 3 apples remaining.")
        assert res["passed"]
        assert res["extracted"] == 3

    def test_arithmetic_rejects_single_digit_frequency_bias(self):
        # UCIP-09R number 21 collapse
        probe = {"target_number": 4}
        res = StrictEvaluators.evaluate_arithmetic(probe, "21")
        assert not res["passed"]

    def test_copying_verbatim_vs_normalized(self):
        probe = {"target_text": "The sky is blue.", "evaluation_type": "verbatim_copy"}
        # Truncated output fails
        assert not StrictEvaluators.evaluate_copying(probe, "The sky.")["passed"]
        # Exact output passes
        assert StrictEvaluators.evaluate_copying(probe, "The sky is blue.")["passed"]
        # Output with missing trailing period fails verbatim
        res_norm = StrictEvaluators.evaluate_copying(probe, "The sky is blue")
        assert not res_norm["exact_match"]
        assert res_norm["norm_match"]

    def test_language_rejects_gibberish_and_length_only_passes(self):
        # UCIP-09 gibberish sentence
        probe = {"subdomain": "english", "expected_keywords": ["memory", "ram", "storage"]}
        res = StrictEvaluators.evaluate_language(probe, "The que wlean a are curves of rest, vols hostant for the lastorgerer.")
        assert not res["passed"], "Must reject corrupted tokens!"

    def test_language_rejects_truncated_fragments(self):
        probe = {"subdomain": "english", "expected_keywords": ["fetch", "decode", "execute"]}
        res = StrictEvaluators.evaluate_language(probe, "A CPU operates through the fetch-decode-")
        assert not res["passed"], "Must reject truncated fragment ending with hyphen!"

    def test_language_devanagari_script_requirement(self):
        probe = {"subdomain": "hindi", "expected_keywords": ["नमस्ते"]}
        assert not StrictEvaluators.evaluate_language(probe, "Namaste aap kaise hain")["passed"]

    def test_uncertainty_abstention_detection(self):
        probe = {"metadata": {"should_abstain": True}}
        assert StrictEvaluators.evaluate_uncertainty(probe, "There is not enough information to answer.")["passed"]
        assert not StrictEvaluators.evaluate_uncertainty(probe, "The answer is definitely 42.")["passed"]

    def test_coding_sandboxed_execution(self):
        probe = {
            "code_test": {
                "test_code": "assert add(2, 3) == 5\nassert add(-1, 1) == 0"
            }
        }
        valid_code = "def add(a, b):\n    return a + b"
        invalid_code = "def add(a, b):\n    return a - b"
        assert StrictEvaluators.evaluate_coding(probe, valid_code)["passed"]
        assert not StrictEvaluators.evaluate_coding(probe, invalid_code)["passed"]
