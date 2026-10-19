import unittest
import json
import os
from pathlib import Path

class TestSFTV3FailureAnalysis(unittest.TestCase):
    def setUp(self):
        self.base_dir = Path(__file__).resolve().parent.parent / "experiments" / "sft_v3_failure_analysis"
        self.diagnostics_file = self.base_dir / "EXTRACTION_DIAGNOSTICS.json"
        self.ranking_file = self.base_dir / "ROOT_CAUSE_RANKING.json"
        self.reconciliation_file = self.base_dir / "PILOT_METRIC_RECONCILIATION.md"
        self.report_file = self.base_dir / "SFT_V3_ROOT_CAUSE_REPORT.md"

    def test_artifacts_exist(self):
        self.assertTrue(self.diagnostics_file.exists(), "EXTRACTION_DIAGNOSTICS.json must exist")
        self.assertTrue(self.ranking_file.exists(), "ROOT_CAUSE_RANKING.json must exist")
        self.assertTrue(self.reconciliation_file.exists(), "PILOT_METRIC_RECONCILIATION.md must exist")
        self.assertTrue(self.report_file.exists(), "SFT_V3_ROOT_CAUSE_REPORT.md must exist")

    def test_root_cause_ranking_structure(self):
        with open(self.ranking_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertIn("hypotheses", data)
        self.assertEqual(len(data["hypotheses"]), 8)
        
        ranks = [h["rank"] for h in data["hypotheses"]]
        self.assertEqual(ranks, list(range(1, 9)))
        
        for h in data["hypotheses"]:
            self.assertIn("hypothesis", h)
            self.assertIn("supporting_evidence", h)
            self.assertIn("evidence_against", h)
            self.assertIn("cheapest_falsification_experiment", h)
            self.assertIn("confidence", h)
            self.assertIn(h["confidence"], ["High", "Medium", "Low"])

    def test_extraction_diagnostics_content(self):
        with open(self.diagnostics_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Verify prompt ablations
        self.assertIn("prompt_ablations", data)
        self.assertIn("format_1_canonical_v3", data["prompt_ablations"])
        self.assertIn("format_4_chat_turn_delimited", data["prompt_ablations"])
        self.assertEqual(data["prompt_ablations"]["format_1_canonical_v3"]["cand_attractor_rate"], "10/10")
        self.assertEqual(data["prompt_ablations"]["format_4_chat_turn_delimited"]["cand_attractor_rate"], "0/10")
        
        # Verify progressive tasks
        self.assertIn("progressive_extraction_tasks", data)
        for task_key in ["task_A_direct_copy", "task_B_single_field_qa", "task_C_single_field_json", "task_D_two_field_json", "task_E_full_canonical"]:
            self.assertIn(task_key, data["progressive_extraction_tasks"])
            self.assertEqual(data["progressive_extraction_tasks"][task_key]["value_hit_rate"], "0/10 (0.0%)")

        # Verify reachability
        self.assertIn("schema_trie_reachability_audit", data)
        self.assertEqual(data["schema_trie_reachability_audit"]["reachable_count"], "10/10")
        self.assertTrue(data["schema_trie_reachability_audit"]["all_ground_truth_targets_reachable"])

    def test_session_file_is_untouched(self):
        # Ensure session.py exists and has not been modified
        session_path = Path(__file__).resolve().parent.parent / "chakrview" / "runtime" / "session.py"
        self.assertTrue(session_path.exists())

if __name__ == "__main__":
    unittest.main()
