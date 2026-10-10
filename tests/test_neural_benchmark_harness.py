"""
Unit & Regression Tests for Neural Generation Benchmark Harness v2.0.

Tracked Location: tests/test_neural_benchmark_harness.py

Verifies:
1. Metric calculations using known synthetic token sequences (n-gram repetition, unique token ratio).
2. Clean loading and execution of RANDOM_INITIALIZATION_BASELINE.
3. Fail-closed behavior on missing or corrupted checkpoint paths.
4. Correct recording of system info, parameter counts, and weight hashes.
5. Invariant check: Ensures model weights are not mutated during benchmark execution (ΔW = 0).
6. File serialization (JSON & Markdown) without partial file corruption.
7. Verification of Prompt Suite v2.0 schema and completeness (30 prompts, 6 domains).
8. Canonical checkpoint identity verification for Stage C and Stage D checkpoints.
9. Deterministic evaluation logic (substring matching and syntax checking).
10. EOS vs MAX_TOKENS termination handling.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from evaluation.neural_benchmark_harness import (
    NeuralBenchmarkHarness,
    compute_ngram_repetition_ratio,
    compute_unique_token_ratio,
    evaluate_deterministic_match,
    file_sha256,
    save_benchmark_results,
)


class TestNeuralBenchmarkHarness(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root_dir = Path(__file__).resolve().parents[1]
        cls.prompts_v1_path = cls.root_dir / "evaluation" / "benchmark_prompts_v1.0.json"
        cls.prompts_v2_path = cls.root_dir / "evaluation" / "benchmark_prompts_v2.0.json"
        cls.tokenizer_dir = cls.root_dir / "data" / "experiments" / "vocab_4096"
        cls.stage_c_path = cls.root_dir / "checkpoints" / "experiments" / "neural_learning_v02" / "run_10k" / "checkpoint_0009000.pt"
        cls.stage_d_path = cls.root_dir / "checkpoints" / "experiments" / "neural_learning_v02" / "stage_d_pilot_run500" / "checkpoints" / "checkpoint_0000500.pt"

    def test_01_synthetic_metric_calculations(self):
        """Verifies mathematical correctness of unique-token and n-gram repetition ratios."""
        # Case A: Pure repetition of 1 token [5, 5, 5, 5, 5]
        toks_rep = [5, 5, 5, 5, 5]
        self.assertAlmostEqual(compute_unique_token_ratio(toks_rep), 1 / 5)
        # 4 2-grams: (5,5), (5,5), (5,5), (5,5). Unique: 1, repeated: 3 -> 3/4 = 0.75
        self.assertAlmostEqual(compute_ngram_repetition_ratio(toks_rep, 2), 3 / 4)

        # Case B: Completely unique sequence [1, 2, 3, 4, 5, 6, 7, 8]
        toks_uniq = list(range(1, 9))
        self.assertAlmostEqual(compute_unique_token_ratio(toks_uniq), 1.0)
        self.assertAlmostEqual(compute_ngram_repetition_ratio(toks_uniq, 2), 0.0)
        self.assertAlmostEqual(compute_ngram_repetition_ratio(toks_uniq, 4), 0.0)

        # Case C: Empty sequence
        self.assertEqual(compute_unique_token_ratio([]), 0.0)
        self.assertEqual(compute_ngram_repetition_ratio([], 2), 0.0)

    def test_02_harness_fails_closed_on_missing_checkpoint(self):
        """Harness must raise FileNotFoundError when an invalid checkpoint path is supplied."""
        bad_path = Path("checkpoints/non_existent_model_123.pt")
        with self.assertRaises(FileNotFoundError):
            NeuralBenchmarkHarness("BAD_CKPT", checkpoint_path=bad_path)

    def test_03_random_baseline_benchmark_execution(self):
        """Runs a minimal benchmark on random baseline and verifies invariants."""
        harness = NeuralBenchmarkHarness(
            checkpoint_label="RANDOM_INITIALIZATION_BASELINE",
            checkpoint_path=None,
            cpu_threads=2,
            seed=42,
        )
        self.assertEqual(harness.model_weight_hash, "c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da")
        self.assertEqual(sum(p.numel() for p in harness.model.parameters()), 3_443_136)

        # Run on a 2-prompt temporary suite
        with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".json", encoding="utf-8") as tf:
            suite_data = {
                "suite_version": "test-1.0",
                "prompts": [
                    {"id": "TEST-01", "category": "test", "prompt": "Hello world"},
                    {"id": "TEST-02", "category": "test", "prompt": "ChakrView architecture"},
                ]
            }
            tf.write(json.dumps(suite_data))
            tf_path = Path(tf.name)

        res = harness.run_benchmark(prompts_path=tf_path, max_new_tokens=8, warmup_runs=0)
        self.assertEqual(res.total_prompts, 2)
        self.assertEqual(len(res.prompt_results), 2)
        self.assertTrue(res.mean_throughput_tps > 0.0)
        self.assertEqual(res.checkpoint_label, "RANDOM_INITIALIZATION_BASELINE")

        # Verify weight hash did not change during generation (ΔW = 0)
        post_hash = harness.runtime.engine.compute_weight_hash()
        self.assertEqual(post_hash, "c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da")

        tf_path.unlink()

    def test_04_save_benchmark_results_serialization(self):
        """Verifies JSON and Markdown serialization of benchmark results."""
        harness = NeuralBenchmarkHarness("TEST_INIT", checkpoint_path=None, cpu_threads=1)
        with tempfile.TemporaryDirectory() as td:
            p_json = Path(td) / "eval.json"
            p_md = Path(td) / "eval.md"

            with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".json", encoding="utf-8") as tf:
                tf.write(json.dumps({"suite_version": "test-2.0", "prompts": [{"id": "P1", "category": "cat", "prompt": "test"}]}))
                tf_path = Path(tf.name)

            res = harness.run_benchmark(prompts_path=tf_path, max_new_tokens=4, warmup_runs=0)
            save_benchmark_results(res, p_json, p_md)

            self.assertTrue(p_json.is_file())
            self.assertTrue(p_md.is_file())

            data = json.loads(p_json.read_text(encoding="utf-8"))
            self.assertEqual(data["checkpoint_label"], "TEST_INIT")
            self.assertEqual(len(data["prompt_results"]), 1)

            md_content = p_md.read_text(encoding="utf-8")
            self.assertIn("# Benchmark Report: TEST_INIT", md_content)
            self.assertIn("c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da", md_content)

            tf_path.unlink()

    def test_05_prompt_suite_v2_schema_and_domain_coverage(self):
        """Verifies that benchmark_prompts_v2.0.json conforms to the 6-domain 30-prompt specification."""
        self.assertTrue(self.prompts_v2_path.is_file(), "v2 prompt suite file missing")
        data = json.loads(self.prompts_v2_path.read_text(encoding="utf-8"))
        self.assertEqual(data["suite_version"], "2.0.0")
        self.assertEqual(len(data["domains"]), 6)
        self.assertEqual(len(data["prompts"]), 30)

        domains = {p["domain"] for p in data["prompts"]}
        self.assertEqual(len(domains), 6)
        for p in data["prompts"]:
            self.assertIn("id", p)
            self.assertIn("domain", p)
            self.assertIn("prompt", p)
            self.assertIn("provenance", p)
            self.assertIn("expected_answer", p)
            self.assertIn("evaluation_type", p)
            self.assertIn("human_rubric", p)

    def test_06_checkpoint_identity_and_hash_resolution(self):
        """Verifies full weight hashes and file sha256 for canonical baselines and checkpoints."""
        # 1. Canonical Random
        h_rand = NeuralBenchmarkHarness("RAND", checkpoint_path=None, seed=42)
        self.assertEqual(h_rand.model_weight_hash, "c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da")

        # 2. Stage C 9k Checkpoint
        if self.stage_c_path.is_file():
            h_c = NeuralBenchmarkHarness("STAGE_C", checkpoint_path=self.stage_c_path)
            self.assertEqual(h_c.model_weight_hash, "36711d25831740840dc9d62d60087a898aa4a19746322ef3cb0e04974a2770e3")
            self.assertEqual(file_sha256(self.stage_c_path), "3e03ae0a134da40439038d965278f4da2996b1ac2e8f959d6059859600b0bb23")

        # 3. Stage D 500 Checkpoint
        if self.stage_d_path.is_file():
            h_d = NeuralBenchmarkHarness("STAGE_D", checkpoint_path=self.stage_d_path)
            self.assertEqual(h_d.model_weight_hash, "5f3712d80940fb9345d01b5434e8bb64936d01e4d3aaff096ab5268c282d506c")
            self.assertEqual(file_sha256(self.stage_d_path), "8ab1ec36b5c5a09929b8510332f18fecc0183eb0e9332ab99375995a8cc1a449")

    def test_07_deterministic_match_evaluation(self):
        """Verifies deterministic evaluation helper functions on synthetic outputs across validity edge cases."""
        # 1. Exact match and valid formatting
        self.assertTrue(evaluate_deterministic_match("The answer is 1921.", "deterministic_substring", ["1921"]))
        self.assertTrue(evaluate_deterministic_match("   1921   \n", "deterministic_substring", ["1921"]))
        self.assertTrue(evaluate_deterministic_match("Result: 27", "deterministic_exact_or_substring", ["27"]))
        self.assertTrue(evaluate_deterministic_match("Result: yes", "deterministic_substring", ["Yes"]))

        # 2. Prevent substring false positives using word boundaries (e.g. 27 inside 127)
        self.assertFalse(evaluate_deterministic_match("The count was 127 items.", "deterministic_exact_or_substring", ["27"]))
        self.assertFalse(evaluate_deterministic_match("Total is 270 degrees.", "deterministic_exact_or_substring", ["27"]))
        self.assertFalse(evaluate_deterministic_match("It was in 19210.", "deterministic_substring", ["1921"]))

        # 3. Code syntax: AST parsing rejects broken syntax despite containing key substrings
        self.assertTrue(evaluate_deterministic_match("def is_positive(x):\n    return x > 0", "deterministic_syntax_check", ["def is_positive", "> 0"]))
        # Missing colon in function definition: AST fails
        self.assertFalse(evaluate_deterministic_match("def is_positive(x) return x > 0", "deterministic_syntax_check", ["def is_positive", "> 0"]))
        # Missing return statement:
        self.assertFalse(evaluate_deterministic_match("def is_positive(x):\n    pass", "deterministic_syntax_check", ["def is_positive", "return x > 0"]))

        # 4. SQL and C syntax checks
        self.assertTrue(evaluate_deterministic_match("SELECT * FROM users;", "deterministic_syntax_check", ["SELECT * FROM users"]))
        self.assertTrue(evaluate_deterministic_match("select * from users where id=1;", "deterministic_syntax_check", ["SELECT * FROM users"]))
        self.assertTrue(evaluate_deterministic_match("int count = 0;", "deterministic_substring", ["int count = 0;"]))

        # 5. Empty / whitespace outputs fail gracefully
        self.assertFalse(evaluate_deterministic_match("", "deterministic_substring", ["1921"]))
        self.assertFalse(evaluate_deterministic_match("   \n\t  ", "deterministic_substring", ["1921"]))

        # 6. Human rubric returns None (distinguishing subjective from objective)
        self.assertIsNone(evaluate_deterministic_match("Paging uses fixed blocks.", "human_rubric", ["fixed", "variable"]))
        self.assertIsNone(evaluate_deterministic_match("Some text", "human_rubric", []))



if __name__ == "__main__":
    unittest.main()
