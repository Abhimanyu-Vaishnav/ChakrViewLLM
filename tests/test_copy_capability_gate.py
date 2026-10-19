import unittest
import json
import os
from pathlib import Path

class TestCopyCapabilityGate(unittest.TestCase):
    def setUp(self):
        self.repo_root = Path(__file__).resolve().parent.parent
        self.results_path = self.repo_root / "experiments" / "copy_capability_gate" / "COPY_CAPABILITY_RESULTS.json"
        self.runner_path = self.repo_root / "experiments" / "copy_capability_gate" / "run_copy_capability_gate.py"
        self.report_path = self.repo_root / "experiments" / "copy_capability_gate" / "COPY_CAPABILITY_GATE_REPORT.md"
        self.readme_path = self.repo_root / "experiments" / "copy_capability_gate" / "README.md"
        self.session_path = self.repo_root / "chakrview" / "runtime" / "session.py"
        
        # Load results
        with open(self.results_path, "r", encoding="utf-8") as f:
            self.results = json.load(f)

    def test_artifacts_exist(self):
        self.assertTrue(self.results_path.exists(), "COPY_CAPABILITY_RESULTS.json must exist")
        self.assertTrue(self.runner_path.exists(), "run_copy_capability_gate.py must exist")
        self.assertTrue(self.report_path.exists(), "COPY_CAPABILITY_GATE_REPORT.md must exist")
        self.assertTrue(self.readme_path.exists(), "README.md must exist")
        self.assertTrue(self.session_path.exists(), "session.py must exist")

    def test_metadata_and_hashes(self):
        meta = self.results.get("metadata", {})
        self.assertEqual(meta.get("experiment"), "COPY_CAPABILITY_GATE")
        self.assertEqual(meta.get("total_records"), 20)
        self.assertEqual(meta.get("base_checkpoint_hash"), "96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee")
        self.assertEqual(meta.get("candidate_checkpoint_hash"), "bfde007db1a90dde5ccc75d14a9ffe59f57b2393b87ac00b19422b8d088d1879")
        self.assertEqual(meta.get("vocab_hash"), "db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a")
        self.assertEqual(meta.get("merges_hash"), "ada4b1dcfbb3d962a2e60203089881872a5063de73b2fef9eeff81ce0016efa7")

    def test_summary_rates_and_structure(self):
        summary = self.results.get("summary", {})
        expected_formats = ["format_1_cir", "format_2_dialog", "format_3_cqa"]
        for fmt in expected_formats:
            self.assertIn(fmt, summary)
            fmt_data = summary[fmt]
            self.assertIn("base", fmt_data)
            self.assertIn("candidate", fmt_data)
            
            # Base exact copy match is 0/20
            self.assertEqual(fmt_data["base"]["exact_match_count"], 0)
            self.assertEqual(fmt_data["base"]["norm_match_count"], 0)
            
            # Candidate exact copy match is 0/20
            self.assertEqual(fmt_data["candidate"]["exact_match_count"], 0)
            self.assertEqual(fmt_data["candidate"]["norm_match_count"], 0)

        # Attractor rate drop on Candidate: format_1_cir (90.0%) -> format_2_dialog (15.0%)
        cir_cand_attr = summary["format_1_cir"]["candidate"]["attractor_count"]
        dialog_cand_attr = summary["format_2_dialog"]["candidate"]["attractor_count"]
        self.assertEqual(cir_cand_attr, 18)
        self.assertEqual(dialog_cand_attr, 3)
        self.assertLess(dialog_cand_attr, cir_cand_attr)

    def test_passage_contains_ground_truth(self):
        examples = self.results.get("per_example_results", [])
        self.assertEqual(len(examples), 20)
        for ex in examples:
            expected = ex["expected_value"]
            context = ex["context"]
            self.assertIn(expected.lower(), context.lower(), f"Expected value '{expected}' must exist in passage '{context}'")

    def test_boundary_token_tracking(self):
        examples = self.results.get("per_example_results", [])
        for ex in examples:
            for fmt_key, fmt_val in ex["formats"].items():
                self.assertIn("boundary_token_ids", fmt_val)
                self.assertIn("boundary_text", fmt_val)
                self.assertTrue(len(fmt_val["boundary_token_ids"]) > 0)


if __name__ == "__main__":
    unittest.main()
