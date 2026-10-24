"""
Real-Data Correctness and Regression Test Suite for Exp19 Response Diversifier.
Tests across actual eligible records from the Exp18 generator and synthetic edge cases.
"""

import re
import unittest
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from experiments.research_v09_conversation_sft.exp18_scaled_instruction_format_v1.generate_exp18_datasets import Exp18DatasetGenerator
from experiments.overnight_runs.exp19_final_integration_audit_20261024_122100.diversification_transform import Exp19ResponseDiversifier


class TestRealDataDiversification(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gen = Exp18DatasetGenerator()
        cls.splits = cls.gen.generate_all_splits()
        cls.diversifier = Exp19ResponseDiversifier(num_variants=4)
        cls.trans_train, cls.train_stats = cls.diversifier.transform_split("train", cls.splits["train"])
        cls.trans_val, cls.val_stats = cls.diversifier.transform_split("val", cls.splits["val"])
        cls.trans_test, cls.test_stats = cls.diversifier.transform_split("test", cls.splits["test"])

    # 1. Dataset Counts and Membership Preservation
    def test_record_counts_preserved(self):
        self.assertEqual(len(self.trans_train), 1200)
        self.assertEqual(len(self.trans_val), 160)
        self.assertEqual(len(self.trans_test), 160)

    def test_record_ids_and_metadata_preserved(self):
        for orig, trans in zip(self.splits["train"], self.trans_train):
            self.assertEqual(orig["id"], trans["id"])
            self.assertEqual(orig["category"], trans["category"])
            self.assertEqual(orig["prompt"], trans["prompt"])
            self.assertEqual(orig["prompt_mode"], trans["prompt_mode"])
            self.assertEqual(orig.get("context", ""), trans.get("context", ""))

    # 2. Strict Split Isolation
    def test_validation_split_untouched(self):
        self.assertEqual(self.splits["val"], self.trans_val)

    def test_test_split_untouched(self):
        self.assertEqual(self.splits["test"], self.trans_test)

    # 3. Arithmetic Semantic Answer Preservation on Actual Eligible Records
    def test_arithmetic_numerical_preservation(self):
        arith_orig = [r for r in self.splits["train"] if r["category"] == "basic_arithmetic"]
        arith_trans = [r for r in self.trans_train if r["category"] == "basic_arithmetic"]

        for orig, trans in zip(arith_orig, arith_trans):
            # Extract numerical result from original (last token before period)
            m_orig = re.search(r"(\d+)\.$", orig["response"])
            self.assertIsNotNone(m_orig, f"Failed to extract number from orig: {orig['response']}")
            expected_num = m_orig.group(1)

            # Transformed response must contain the exact same number
            m_trans = re.search(r"(\d+)\.?$", trans["response"])
            self.assertIsNotNone(m_trans, f"Failed to find number in trans: {trans['response']}")
            self.assertEqual(m_trans.group(1), expected_num)

    # 4. Reference Resolution Entity Preservation on Actual Eligible Records
    def test_reference_resolution_entity_preservation(self):
        ref_orig = [r for r in self.splits["train"] if r["category"] == "reference_resolution"]
        ref_trans = [r for r in self.trans_train if r["category"] == "reference_resolution"]

        for orig, trans in zip(ref_orig, ref_trans):
            m_orig = re.search(r"designates (the .*?)\.$", orig["response"])
            self.assertIsNotNone(m_orig)
            entity = m_orig.group(1)

            self.assertIn(entity, trans["response"])
            self.assertTrue(trans["response"].endswith("."))

    # 5. Factual QA Entity Preservation
    def test_factual_qa_entity_preservation(self):
        fqa_orig = [r for r in self.splits["train"] if r["category"] == "factual_qa"]
        fqa_trans = [r for r in self.trans_train if r["category"] == "factual_qa"]

        for orig, trans in zip(fqa_orig, fqa_trans):
            # If it was a chemical symbol question
            if "chemical symbol" in orig["response"].lower():
                m_sym = re.search(r"is ([A-Z][a-z]?)\.$", orig["response"])
                if m_sym:
                    sym = m_sym.group(1)
                    # Symbol must be present as a distinct word in trans
                    self.assertTrue(re.search(r"\b" + sym + r"\b", trans["response"]))

    # 6. Edge Cases & Unsupported Patterns
    def test_edge_cases_unmodified(self):
        edge_cases = [
            {"id": "E1", "category": "basic_arithmetic", "response": ""},
            {"id": "E2", "category": "basic_arithmetic", "response": "Invalid response without numbers."},
            {"id": "E3", "category": "unknown_cat", "response": "Some text."},
            {"id": "E4", "category": "factual_qa", "response": "Normal sentence."},
        ]
        for ec in edge_cases:
            rec, changed = self.diversifier.transform_record(ec, 0)
            self.assertFalse(changed)
            self.assertEqual(rec["response"], ec["response"])

    # 7. Determinism across Repeated Executions
    def test_determinism_across_runs(self):
        trans_1, _ = self.diversifier.transform_split("train", self.splits["train"])
        trans_2, _ = self.diversifier.transform_split("train", self.splits["train"])
        self.assertEqual(trans_1, trans_2)


if __name__ == "__main__":
    unittest.main()
