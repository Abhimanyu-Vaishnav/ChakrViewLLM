import unittest
import json
from pathlib import Path
from evaluation.coding_suite.run_coding_evaluation import check_python_syntax, evaluate_coding_suite
from evaluation.coding_suite.isolated_executor import run_process_isolated_test, audit_code_safety


class TestCodingSuiteHardened(unittest.TestCase):
    def setUp(self):
        self.suite_dir = Path(__file__).resolve().parent.parent / "evaluation" / "coding_suite"
        self.tasks_path = self.suite_dir / "coding_tasks.json"
        self.baseline_results_path = self.suite_dir / "coding_baseline_results.json"

    def test_tasks_file_structure_and_invariants(self):
        self.assertTrue(self.tasks_path.is_file(), "coding_tasks.json must exist")
        with open(self.tasks_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertIn("tasks", data)
        self.assertEqual(len(data["tasks"]), 12)
        
        # Check unique IDs and required schema fields
        seen_ids = set()
        for t in data["tasks"]:
            self.assertIn("id", t)
            self.assertIn("category", t)
            self.assertIn("prompt", t)
            self.assertIn("validation_type", t)
            self.assertNotIn(t["id"], seen_ids, f"Duplicate task id: {t['id']}")
            seen_ids.add(t["id"])

        categories = sorted(list({t["category"] for t in data["tasks"]}))
        expected_categories = [
            "bug_identification_and_repair",
            "code_continuation",
            "code_explanation",
            "code_summarization",
            "small_function_implementation",
            "test_case_generation"
        ]
        self.assertEqual(categories, expected_categories)

    def test_process_isolation_and_safety_rejection(self):
        # 1. Benign valid code passes in subprocess
        res_pass = run_process_isolated_test("def add(a, b): return a + b", "assert add(2, 3) == 5")
        self.assertTrue(res_pass["passed"])
        self.assertTrue(res_pass["syntax_valid"])
        self.assertIsNone(res_pass["error"])

        # 2. Syntax error caught safely
        res_syn = run_process_isolated_test("def broken(: pass", "assert True")
        self.assertFalse(res_syn["passed"])
        self.assertFalse(res_syn["syntax_valid"])
        self.assertIn("SyntaxError", res_syn["error"])

        # 3. Failing assertion caught safely
        res_fail = run_process_isolated_test("def add(a, b): return a - b", "assert add(2, 3) == 5")
        self.assertFalse(res_fail["passed"])
        self.assertTrue(res_fail["syntax_valid"])
        self.assertIn("AssertionError", res_fail["error"])

        # 4. Prohibited module import rejected by policy before execution
        res_os = run_process_isolated_test("import os; os.system('echo 1')", "assert True")
        self.assertFalse(res_os["passed"])
        self.assertTrue(res_os["rejected_by_policy"])
        self.assertIn("SecurityRejection", res_os["error"])

        res_subp = run_process_isolated_test("import subprocess; subprocess.run(['ls'])", "assert True")
        self.assertFalse(res_subp["passed"])
        self.assertTrue(res_subp["rejected_by_policy"])
        self.assertIn("SecurityRejection", res_subp["error"])

        # 5. Prohibited function call rejected by policy
        res_open = run_process_isolated_test("f = open('secret.txt', 'w')", "assert True")
        self.assertFalse(res_open["passed"])
        self.assertTrue(res_open["rejected_by_policy"])
        self.assertIn("SecurityRejection", res_open["error"])

    def test_timeout_enforcement(self):
        # Infinite loop terminates cleanly via timeout
        loop_code = "def hang():\n    while True:\n        pass\nhang()"
        res_timeout = run_process_isolated_test(loop_code, "assert True", timeout_seconds=1.0)
        self.assertFalse(res_timeout["passed"])
        self.assertTrue(res_timeout["timed_out"])
        self.assertIn("TimeoutError", res_timeout["error"])

    def test_baseline_results_presence_and_truthfulness(self):
        self.assertTrue(self.baseline_results_path.is_file(), "coding_baseline_results.json must exist")
        with open(self.baseline_results_path, "r", encoding="utf-8") as f:
            res = json.load(f)
        self.assertEqual(res["total_tasks"], 12)
        self.assertEqual(res["checkpoint_sha256"], "96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee")
        self.assertEqual(res["test_passed_count"], 2)
        self.assertIn("timeout_count", res)
        self.assertIn("policy_rejected_count", res)

    def test_benchmark_smoke_mode_execution(self):
        # Verify benchmark executes in deterministic smoke mode without neural weights
        out_tmp = Path(__file__).resolve().parent.parent / "scratch" / "test_smoke_eval.json"
        res = evaluate_coding_suite(smoke_mode=True, output_path=str(out_tmp))
        self.assertEqual(res["total_tasks"], 12)
        self.assertEqual(res["mode"], "smoke_verification")
        self.assertFalse(res["checkpoint_present"])
        self.assertTrue(out_tmp.is_file())
        if out_tmp.is_file():
            out_tmp.unlink()


if __name__ == "__main__":
    unittest.main()