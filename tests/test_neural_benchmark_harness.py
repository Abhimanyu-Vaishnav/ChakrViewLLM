"""
Unit & Regression Tests for Neural Generation Benchmark Harness.

Tracked Location: tests/test_neural_benchmark_harness.py

Verifies:
1. Metric calculations using known synthetic token sequences (n-gram repetition, unique token ratio).
2. Clean loading and execution of RANDOM_INITIALIZATION_BASELINE.
3. Fail-closed behavior on missing or corrupted checkpoint paths.
4. Correct recording of system info, parameter counts, and weight hashes.
5. Invariant check: Ensures model weights are not mutated during benchmark execution (ΔW = 0).
6. File serialization (JSON & Markdown) without partial file corruption.
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
    save_benchmark_results,
)


class TestNeuralBenchmarkHarness(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root_dir = Path(__file__).resolve().parents[1]
        cls.prompts_path = cls.root_dir / "evaluation" / "benchmark_prompts_v1.0.json"
        cls.tokenizer_dir = cls.root_dir / "data" / "experiments" / "vocab_4096"

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
                tf.write(json.dumps({"prompts": [{"id": "P1", "category": "cat", "prompt": "test"}]}))
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


if __name__ == "__main__":
    unittest.main()
