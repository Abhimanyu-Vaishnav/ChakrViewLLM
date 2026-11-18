import pytest
from experiments.ucip10e.evaluators.strict_evaluators import StrictEvaluators

class TestAdversarialEvaluatorSuite:

    # 1. Mathematics Adversarial
    def test_math_wrong_with_matching_operands(self):
        # "8 + 7 = 15" -> output says "8 + 7 = 14"
        probe = {"target_number": 15}
        assert not StrictEvaluators.evaluate_arithmetic(probe, "8 + 7 = 14")["passed"]

    def test_math_reversed_subtraction(self):
        # 15 - 7 = 8 -> output says "7 - 15 = -8"
        probe = {"target_number": 8}
        assert not StrictEvaluators.evaluate_arithmetic(probe, "The result is -8")["passed"]

    def test_math_unit_conversion_text(self):
        probe = {"target_number": 12}
        assert StrictEvaluators.evaluate_arithmetic(probe, "You have 12 books remaining in total.")["passed"]

    def test_math_multi_number_confusion(self):
        probe = {"target_number": 10}
        # If output has conflicting numbers without equation
        assert not StrictEvaluators.evaluate_arithmetic(probe, "I see 10 cats, 5 dogs, and 2 birds.")["passed"]

    # 2. English Semantic Adversarial
    def test_english_keyword_stuffed_gibberish(self):
        probe = {"subdomain": "english", "expected_keywords": ["memory", "ram", "storage"]}
        # Contains keywords but is a word-salad loop
        assert not StrictEvaluators.evaluate_language(probe, "ram storage memory ram storage memory ram storage memory")["passed"]

    def test_english_correct_concise(self):
        probe = {"subdomain": "english", "expected_keywords": ["processor", "instructions"]}
        assert StrictEvaluators.evaluate_language(probe, "The CPU is a processor that executes instructions.")["passed"]

    def test_english_corrupted_tokens(self):
        probe = {"subdomain": "english", "expected_keywords": ["memory", "storage"]}
        assert not StrictEvaluators.evaluate_language(probe, "The que wlean a are curves of rest, vols hostant.")["passed"]

    # 3. Hindi Script & Semantic Adversarial
    def test_hindi_romanized_rejected(self):
        probe = {"subdomain": "hindi", "expected_keywords": ["नमस्ते"]}
        assert not StrictEvaluators.evaluate_language(probe, "Namaste! Main aapka sahayak hoon.")["passed"]

    def test_hindi_genuine_devanagari_passed(self):
        probe = {"subdomain": "hindi", "expected_keywords": ["सहायता", "हूँ"]}
        assert StrictEvaluators.evaluate_language(probe, "नमस्ते! मैं आपकी सहायता करने के लिए उपस्थित हूँ।")["passed"]

    # 4. Copying Adversarial
    def test_copying_one_char_substitution(self):
        probe = {"target_text": "The sky is blue.", "evaluation_type": "verbatim_copy"}
        assert not StrictEvaluators.evaluate_copying(probe, "The sky is glue.")["passed"]

    def test_copying_extra_conversational_chatter(self):
        probe = {"target_text": "Knowledge is power.", "evaluation_type": "verbatim_copy"}
        assert not StrictEvaluators.evaluate_copying(probe, "Sure, here it is: Knowledge is power.")["passed"]

    # 5. Logic Adversarial
    def test_logic_contradiction_detection(self):
        probe = {"expected_answer": "no"}
        assert StrictEvaluators.evaluate_logic(probe, "No, a trout cannot fly.")["passed"]
        assert not StrictEvaluators.evaluate_logic(probe, "Yes, a trout can fly.")["passed"]

    # 6. Uncertainty Adversarial
    def test_uncertainty_known_vs_unknown(self):
        probe_unanswerable = {"metadata": {"should_abstain": True}}
        probe_answerable = {"metadata": {"should_abstain": False}}
        # Model claims definite answer to unanswerable -> FAIL
        assert not StrictEvaluators.evaluate_uncertainty(probe_unanswerable, "The price was exactly 42 pounds.")["passed"]
        # Model abstains on unanswerable -> PASS
        assert StrictEvaluators.evaluate_uncertainty(probe_unanswerable, "There is not enough information to answer.")["passed"]
        # Model abstains on answerable -> FAIL
        assert not StrictEvaluators.evaluate_uncertainty(probe_answerable, "I cannot answer this question.")["passed"]

    # 7. Self-Modeling Adversarial
    def test_self_modeling_identity_vs_hallucination(self):
        probe = {"expected_keywords": ["chakrview"]}
        assert StrictEvaluators.evaluate_self_modeling(probe, "I am ChakrView, an independent assistant.")["passed"]
        assert not StrictEvaluators.evaluate_self_modeling(probe, "I am ChatGPT, a large language model created by OpenAI.")["passed"]
