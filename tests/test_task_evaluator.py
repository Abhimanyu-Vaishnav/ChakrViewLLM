"""
Unit and Regression Test Suite for ChakrView TaskEvaluator v0.1.
Location: tests/test_task_evaluator.py
"""

import unittest
from chakrview.evaluation.task_evaluator import (
    TaskEvaluator,
    EvaluationOutcome,
    ScoringStatus,
)


class TestTaskEvaluator(unittest.TestCase):
    def setUp(self):
        self.evaluator = TaskEvaluator()

    # 1. Basic Arithmetic Tests
    def test_arithmetic_correct_addition(self):
        rec = {"id": "ARITH_01", "category": "basic_arithmetic", "prompt": "What is the sum of 12 and 15?", "response": "The sum of 12 and 15 is 27."}
        res = self.evaluator.evaluate_record(rec, "The answer is 27.", stop_reason="STOP_TOKEN")
        self.assertEqual(res.outcome, EvaluationOutcome.CORRECT)
        self.assertEqual(res.expected_target, "27")
        self.assertTrue(res.is_eos)

    def test_arithmetic_incorrect_addition(self):
        rec = {"id": "ARITH_02", "category": "basic_arithmetic", "prompt": "What is the sum of 12 and 15?", "response": "The sum of 12 and 15 is 27."}
        res = self.evaluator.evaluate_record(rec, "The answer is 406.", stop_reason="STOP_TOKEN")
        self.assertEqual(res.outcome, EvaluationOutcome.INCORRECT)

    def test_arithmetic_multiplication(self):
        rec = {"id": "ARITH_03", "category": "basic_arithmetic", "prompt": "Determine the product when 14 is multiplied by 3.", "response": "The product is 42."}
        res = self.evaluator.evaluate_record(rec, "The product is 42.", stop_reason="STOP_TOKEN")
        self.assertEqual(res.outcome, EvaluationOutcome.CORRECT)

    def test_arithmetic_substring_false_positive_prevention(self):
        # Target is 12, generated contains 112 or 120 -> should NOT match as 12
        rec = {"id": "ARITH_04", "category": "basic_arithmetic", "prompt": "What is 6 plus 6?", "response": "The sum is 12."}
        res = self.evaluator.evaluate_record(rec, "The total count is 112.", stop_reason="STOP_TOKEN")
        self.assertEqual(res.outcome, EvaluationOutcome.INCORRECT)

    def test_arithmetic_negative_result(self):
        rec = {"id": "ARITH_05", "category": "basic_arithmetic", "prompt": "What is the difference when 20 is subtracted from 5?", "response": "The result is -15."}
        res = self.evaluator.evaluate_record(rec, "Result: -15", stop_reason="STOP_TOKEN")
        self.assertEqual(res.outcome, EvaluationOutcome.CORRECT)

    # 2. Structured Formatting (JSON) Tests
    def test_json_valid_and_schema_correct(self):
        rec = {"id": "JSON_01", "category": "structured_formatting", "prompt": "Format metric_0 and val_0 into JSON.", "response": '{"metric_0": "val_0"}'}
        res = self.evaluator.evaluate_record(rec, '{"metric_0": "val_0"}', stop_reason="STOP_TOKEN")
        self.assertEqual(res.outcome, EvaluationOutcome.CORRECT)
        self.assertTrue(res.format_valid)

    def test_json_malformed_syntax(self):
        rec = {"id": "JSON_02", "category": "structured_formatting", "prompt": "Format metric_0 and val_0 into JSON.", "response": '{"metric_0": "val_0"}'}
        res = self.evaluator.evaluate_record(rec, '{"metric_0": "val_0"', stop_reason="MAX_TOKENS")
        self.assertEqual(res.outcome, EvaluationOutcome.INCORRECT)
        self.assertFalse(res.format_valid)
        self.assertIn("JSON_SYNTAX_INVALID", res.reason_codes)

    def test_json_empty_output(self):
        rec = {"id": "JSON_03", "category": "structured_formatting", "prompt": "Format metric_0 and val_0 into JSON.", "response": '{"metric_0": "val_0"}'}
        res = self.evaluator.evaluate_record(rec, "", stop_reason="STOP_TOKEN")
        self.assertEqual(res.outcome, EvaluationOutcome.INCORRECT)
        self.assertTrue(res.is_empty)

    def test_json_docstring_fragment_rejected(self):
        rec = {"id": "JSON_04", "category": "structured_formatting", "prompt": "Format metric_0 into JSON.", "response": '{"metric_0": "val_0"}'}
        res = self.evaluator.evaluate_record(rec, '\n """\n if not isinstance("metric_0")', stop_reason="STOP_TOKEN")
        self.assertEqual(res.outcome, EvaluationOutcome.INCORRECT)
        self.assertFalse(res.format_valid)

    # 3. Uncertainty and Abstention Tests
    def test_abstention_appropriate_on_unanswerable(self):
        rec = {
            "id": "UA_01",
            "category": "uncertainty_abstention",
            "prompt": "What is the bandwidth?",
            "context": "Node CPU is 40%.",
            "metadata": {"requires_abstention": True},
            "response": "The log does not state the bandwidth.",
        }
        res = self.evaluator.evaluate_record(rec, "The report does not state network bandwidth.", stop_reason="STOP_TOKEN")
        self.assertEqual(res.outcome, EvaluationOutcome.CORRECT)
        self.assertIn("APPROPRIATE_ABSTENTION", res.reason_codes)

    def test_abstention_unsupported_answer_on_unanswerable(self):
        rec = {
            "id": "UA_02",
            "category": "uncertainty_abstention",
            "prompt": "What is the bandwidth?",
            "context": "Node CPU is 40%.",
            "metadata": {"requires_abstention": True},
            "response": "The log does not state the bandwidth.",
        }
        res = self.evaluator.evaluate_record(rec, "Bandwidth is 10 Gbps.", stop_reason="STOP_TOKEN")
        self.assertEqual(res.outcome, EvaluationOutcome.INCORRECT)
        self.assertIn("UNSUPPORTED_ANSWER_REFUSAL_EXPECTED", res.reason_codes)

    def test_abstention_control_different_targets(self):
        # Target is "15 tons"
        rec = {
            "id": "UA_03",
            "category": "uncertainty_abstention",
            "prompt": "How many tons of cargo were loaded?",
            "context": "Manifest: Vessel loaded 15 tons of grain.",
            "metadata": {"requires_abstention": False},
            "response": "Vessel loaded 15 tons of grain.",
        }
        # Correct answer
        res_correct = self.evaluator.evaluate_record(rec, "15 tons of grain were loaded.", stop_reason="STOP_TOKEN")
        self.assertEqual(res_correct.outcome, EvaluationOutcome.CORRECT)

        # Incorrect refusal on answerable control
        res_refusal = self.evaluator.evaluate_record(rec, "The manifest does not state the cargo weight.", stop_reason="STOP_TOKEN")
        self.assertEqual(res_refusal.outcome, EvaluationOutcome.INCORRECT)
        self.assertIn("INCORRECT_ABSTENTION_CONTROL_WAS_ANSWERABLE", res_refusal.reason_codes)

    # 4. Context Extraction and Reference Resolution Tests
    def test_context_extraction_correct(self):
        rec = {
            "id": "EXT_01",
            "category": "context_extraction",
            "prompt": "Who is the pilot?",
            "context": "Roster: Apollo_test_0 is the navigation pilot.",
            "response": "Apollo_test_0 is the primary navigation pilot.",
        }
        res = self.evaluator.evaluate_record(rec, "Pilot is Apollo_test_0.", stop_reason="STOP_TOKEN")
        self.assertEqual(res.outcome, EvaluationOutcome.CORRECT)

    def test_reference_resolution_correct(self):
        rec = {
            "id": "REF_01",
            "category": "reference_resolution",
            "prompt": "What does it refer to?",
            "context": "Inspector found brass compass_test_1 and boxed it.",
            "response": "The pronoun 'it' designates the brass compass_test_1.",
        }
        res = self.evaluator.evaluate_record(rec, "It refers to the brass compass_test_1.", stop_reason="STOP_TOKEN")
        self.assertEqual(res.outcome, EvaluationOutcome.CORRECT)

    # 5. Short Explanations Test (Requires Review / Unscored)
    def test_short_explanation_requires_review(self):
        rec = {
            "id": "EXP_01",
            "category": "short_explanation",
            "prompt": "Why is the sky blue?",
            "response": "Rayleigh scattering scatters shorter wavelengths.",
        }
        res = self.evaluator.evaluate_record(rec, "Rayleigh scattering of blue light.", stop_reason="STOP_TOKEN")
        self.assertEqual(res.scoring_status, ScoringStatus.NEEDS_REVIEW)
        self.assertEqual(res.outcome, EvaluationOutcome.NOT_APPLICABLE)

    # 6. Factual QA Regression Tests
    def test_factual_qa_varied_entities(self):
        # Hot desert -> Sahara
        rec_sahara = {
            "id": "FQA_01",
            "category": "factual_qa",
            "prompt": "What is the largest hot desert on Earth?",
            "response": "The Sahara Desert is the largest subtropical hot desert on Earth.",
        }
        res_sahara = self.evaluator.evaluate_record(rec_sahara, "The Sahara is the biggest hot desert.", stop_reason="STOP_TOKEN")
        self.assertEqual(res_sahara.outcome, EvaluationOutcome.CORRECT)
        self.assertIn("FACTUAL_ENTITY_MATCH", res_sahara.reason_codes)

        # Particle -> electron
        rec_electron = {
            "id": "FQA_02",
            "category": "factual_qa",
            "prompt": "What subatomic particle carries a negative electrical charge in an atom?",
            "response": "An electron carries a negative electrical charge in an atom.",
        }
        res_electron = self.evaluator.evaluate_record(rec_electron, "Electrons carry negative charge.", stop_reason="STOP_TOKEN")
        self.assertEqual(res_electron.outcome, EvaluationOutcome.CORRECT)
        self.assertIn("FACTUAL_ENTITY_MATCH", res_electron.reason_codes)

        # Particle -> proton
        rec_proton = {
            "id": "FQA_03",
            "category": "factual_qa",
            "prompt": "What subatomic particle inside the atomic nucleus has a positive charge?",
            "response": "The proton is the subatomic particle with a positive charge.",
        }
        res_proton = self.evaluator.evaluate_record(rec_proton, "A proton has a positive charge.", stop_reason="STOP_TOKEN")
        self.assertEqual(res_proton.outcome, EvaluationOutcome.CORRECT)
        self.assertIn("FACTUAL_ENTITY_MATCH", res_proton.reason_codes)

        # Particle mismatch
        res_mismatch = self.evaluator.evaluate_record(rec_proton, "The photon has no mass.", stop_reason="STOP_TOKEN")
        self.assertEqual(res_mismatch.outcome, EvaluationOutcome.INCORRECT)
        self.assertIn("FACTUAL_ENTITY_MISMATCH", res_mismatch.reason_codes)

        # Single-letter chemical symbol boundary regression test (prevent 'show' matching 'W')
        rec_tungsten = {
            "id": "FQA_04",
            "category": "factual_qa",
            "prompt": "What is the chemical symbol for the element tungsten?",
            "response": "The chemical symbol for tungsten is W.",
        }
        res_spurious = self.evaluator.evaluate_record(rec_tungsten, "The total number of the show of the city", stop_reason="MAX_TOKENS")
        self.assertEqual(res_spurious.outcome, EvaluationOutcome.INCORRECT)
        self.assertIn("FACTUAL_ENTITY_MISMATCH", res_spurious.reason_codes)

        res_genuine = self.evaluator.evaluate_record(rec_tungsten, "Tungsten is denoted by W.", stop_reason="STOP_TOKEN")
        self.assertEqual(res_genuine.outcome, EvaluationOutcome.CORRECT)
        self.assertIn("FACTUAL_ENTITY_MATCH", res_genuine.reason_codes)

        # Leading article regression test (prevent 'the' matching 'The' in degenerate outputs)
        rec_becquerel = {
            "id": "FQA_05",
            "category": "factual_qa",
            "prompt": "What unit is universally used to measure radioactivity?",
            "response": "The becquerel is the standard unit of radioactivity.",
        }
        res_spurious_art = self.evaluator.evaluate_record(rec_becquerel, "The total number of the city of the city of the city", stop_reason="MAX_TOKENS")
        self.assertEqual(res_spurious_art.outcome, EvaluationOutcome.INCORRECT)
        self.assertIn("FACTUAL_ENTITY_MISMATCH", res_spurious_art.reason_codes)

        res_genuine_art = self.evaluator.evaluate_record(rec_becquerel, "The unit is becquerel.", stop_reason="STOP_TOKEN")
        self.assertEqual(res_genuine_art.outcome, EvaluationOutcome.CORRECT)
        self.assertIn("FACTUAL_ENTITY_MATCH", res_genuine_art.reason_codes)

    # 7. Unknown Category Test
    def test_unknown_category_unscored(self):
        rec = {"id": "UNK_01", "category": "nonexistent_domain", "prompt": "Hi", "response": "Hello"}
        res = self.evaluator.evaluate_record(rec, "Hello")
        self.assertEqual(res.scoring_status, ScoringStatus.UNSCORED)


if __name__ == "__main__":
    unittest.main()
