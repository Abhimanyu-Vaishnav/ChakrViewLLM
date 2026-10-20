import unittest
import json
from pathlib import Path
from evaluation.intelligence_suite.run_intelligence_benchmark import evaluate_intelligence_suite

class TestIntelligenceSuite(unittest.TestCase):
    def setUp(self):
        self.suite_dir = Path(__file__).resolve().parent.parent / "evaluation" / "intelligence_suite"
        self.tasks_file = self.suite_dir / "intelligence_tasks.json"

    def test_tasks_file_schema_and_coverage(self):
        self.assertTrue(self.tasks_file.is_file(), "intelligence_tasks.json must exist")
        with open(self.tasks_file, "r", encoding="utf-8-sig") as f:
            data = json.load(f)
        self.assertIn("tasks", data)
        self.assertEqual(len(data["tasks"]), 32)
        self.assertEqual(len(data["categories"]), 10)

        seen_ids = set()
        for t in data["tasks"]:
            self.assertIn("id", t)
            self.assertIn("category", t)
            self.assertIn("prompt", t)
            self.assertIn("expected_concept", t)
            self.assertIn("scoring", t)
            self.assertNotIn(t["id"], seen_ids, f"Duplicate task ID: {t['id']}")
            seen_ids.add(t["id"])

    def test_smoke_mode_execution_and_schema(self):
        out_tmp = Path(__file__).resolve().parent.parent / "evaluation" / "intelligence_suite" / "test_smoke_out.json"
        res = evaluate_intelligence_suite(smoke_mode=True, output_path=str(out_tmp))
        self.assertEqual(res["total_tasks"], 32)
        self.assertEqual(res["mode"], "smoke_verification")
        self.assertFalse(res["checkpoint_present"])
        self.assertTrue(out_tmp.is_file())
        if out_tmp.is_file():
            out_tmp.unlink()

if __name__ == "__main__":
    unittest.main()