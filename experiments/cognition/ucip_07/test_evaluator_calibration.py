"""Evaluator Calibration Test Suite for UCIP-07.

Ensures that empty outputs, repetitive loops, wrong numbers, and non-Devanagari
responses are strictly rejected by the evaluation rubric.
"""

from __future__ import annotations

import os
import sys
import unittest

DIR = os.path.dirname(os.path.abspath(__file__))
if DIR not in sys.path:
    sys.path.insert(0, DIR)

from evaluator_800 import evaluate_probe


class TestEvaluatorCalibration(unittest.TestCase):
    def test_empty_string_rejected(self):
        res = evaluate_probe(
            category="English Conversation",
            prompt="Hello ChakrView",
            output="",
            expected_keywords=["well", "good"],
            rubric="greeting",
            exact_match=False,
        )
        self.assertFalse(res["success"])
        self.assertTrue(res["is_empty"])

    def test_repetitive_loop_rejected(self):
        res = evaluate_probe(
            category="English Conversation",
            prompt="Hello ChakrView",
            output="the the the the the the the the",
            expected_keywords=["the"],
            rubric="greeting",
            exact_match=False,
        )
        self.assertFalse(res["success"])
        self.assertTrue(res["is_repetitive"])

    def test_wrong_arithmetic_rejected(self):
        res = evaluate_probe(
            category="Elementary Arithmetic",
            prompt="What is 4 + 3?",
            output="4 + 3 = 11.",
            expected_keywords=["7", "seven"],
            rubric="4 + 3 = 7",
            exact_match=False,
        )
        self.assertFalse(res["success"])

    def test_correct_arithmetic_accepted(self):
        res = evaluate_probe(
            category="Elementary Arithmetic",
            prompt="What is 4 + 3?",
            output="The answer is 7.",
            expected_keywords=["7", "seven"],
            rubric="4 + 3 = 7",
            exact_match=False,
        )
        self.assertTrue(res["success"])

    def test_non_devanagari_rejected_for_hindi(self):
        res = evaluate_probe(
            category="Devanagari Hindi",
            prompt="नमस्ते!",
            output="Hello I am ChakrView software.",
            expected_keywords=["नमस्ते"],
            rubric="Devanagari test",
            exact_match=False,
        )
        self.assertFalse(res["success"])

    def test_exact_copying_mismatch_rejected(self):
        res = evaluate_probe(
            category="Exact Copying",
            prompt="Say exactly: The sky is blue.",
            output="The sky is green.",
            expected_keywords=["The sky is blue."],
            rubric="Exact copy",
            exact_match=True,
        )
        self.assertFalse(res["success"])

    def test_exact_copying_match_accepted(self):
        res = evaluate_probe(
            category="Exact Copying",
            prompt="Say exactly: The sky is blue.",
            output="The sky is blue.",
            expected_keywords=["The sky is blue."],
            rubric="Exact copy",
            exact_match=True,
        )
        self.assertTrue(res["success"])


if __name__ == "__main__":
    unittest.main()
