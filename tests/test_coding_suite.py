import unittest
import json
from pathlib import Path
from evaluation.coding_suite.run_coding_evaluation import check_python_syntax
from evaluation.coding_suite.isolated_executor import run_process_isolated_test, audit_code_safety


class TestCodingSuiteHardened(unittest.TestCase):
    def setUp(self):
        self.suite_dir = Path(__file__).resolve().parent.parent / "evaluation" / "coding_suite"
        self.tasks_path = self.suite_dir / "coding_tasks.json"
        self.baseline_results_path = self.suite_dir / "coding_baseline_results.json"

    def test_tasks_file_structure(self):
        self.assertTrue(self.tasks_path.is_file(), "coding_tasks.json must exist")
        with open(self.tasks_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertIn("tasks", data)
        self.assertEqual(len(data["tasks"]), 6)
        
        categories = [t["category"] for t in data["tasks"]]
        expected_categories = [
            "code_completion",
            "simple_function_implementation",
            "bug_identification_and_correction",
            "basic_unit_test_generation",
            "following_explicit_requirements",
            "syntax_validity"
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
        self.assertEqual(res["total_tasks"], 6)
        self.assertEqual(res["checkpoint_sha256"], "96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee")
        # Ensure results truthfully reflect current Step 10,000 behavior (2/6 syntax valid, 1/6 passed)
        self.assertEqual(res["syntax_valid_count"], 2)
        self.assertEqual(res["test_passed_count"], 1)
        self.assertIn("timeout_count", res)
        self.assertIn("policy_rejected_count", res)


if __name__ == "__main__":
    unittest.main()
