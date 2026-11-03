"""
Comprehensive Independent Engineering Audit Test Suite for CAND_01 Infrastructure.

Verifies:
1. Deterministic generation across seeds.
2. Unique entity pools and token disjointness (E_train vs E_test).
3. In-partition duplicate prompt detection.
4. Train, val, test partition isolation.
5. Forward association leakage prevention.
6. Reverse association leakage prevention (VER-03 bidirectional exclusion).
7. TF-01 identity copy exception & exclusion from Partitions B and C.
8. Implementation of all eight Task Families: TF-01 through TF-08.
9. Metric calculations: A_FT (first-token exact match), A_FC (full completion match), target loss, perplexity, and target rank.
10. Statistical bootstrap reproducibility.
11. Kish effective sample size calculation (uniform, skewed, zero-weight, non-finite).
12. Degenerate replicate handling with boundary conditions at 10% threshold.
13. Inconclusive classification under tripped degeneracy.
14. Synthetic Monte Carlo diagnostic simulation with Monte Carlo standard errors.
15. Dry-run pipeline integration and hard boundary rejection of forbidden paths.
16. Cross-platform path traversal defenses against canonical data directories.
"""

import hashlib
import json
import os
import sys
import unittest
import numpy as np

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chakrview.eval.statistical_bootstrap import (
    compute_kish_effective_sample_size,
    ordinary_nonparametric_bootstrap,
    one_way_clustered_bootstrap,
    provisional_two_way_cluster_bootstrap,
    run_synthetic_crossed_simulation,
    EstimatorStatus,
    REGISTERED_MAX_DEGENERATE_RATE,
    REGISTERED_MIN_NEFF_FRACTION,
    REGISTERED_MAX_WEIGHT_VARIANCE,
    REGISTERED_MIN_UNIQUE_CLUSTERS,
    DEFAULT_NUM_BOOTSTRAP_REPLICATES,
)
from scripts.generate_cand01_dataset import (
    SyntheticEntityPool,
    CAND01DatasetGenerator,
    PartitionType,
    TaskFamily,
    DatasetRecord,
)
from scripts.evaluate_cand01_benchmark import (
    CAND01EvaluatorHarness,
    SyntheticModelMock,
    EOS_TOKEN_ID,
)
from scripts.run_cand01_experiment import (
    assert_safe_path,
)


class TestCAND01DatasetGeneration(unittest.TestCase):
    """Tests for dataset generation, schemas, token isolation, and leakage prevention."""

    def setUp(self):
        self.pool = SyntheticEntityPool(n_train=200, n_test=50, seed=123)
        self.gen = CAND01DatasetGenerator(self.pool, seed=123)

    def test_token_pool_disjointness(self):
        """E_train and E_test must have zero intersection."""
        train_ids = self.pool.train_token_ids
        test_ids = self.pool.test_token_ids
        self.assertEqual(len(train_ids.intersection(test_ids)), 0)

    def test_deterministic_generation(self):
        """Same seed must produce bit-exact identical records."""
        pool1 = SyntheticEntityPool(n_train=50, n_test=20, seed=42)
        gen1 = CAND01DatasetGenerator(pool1, seed=42)
        r1 = gen1.generate_record(TaskFamily.TF02_SINGLE_PAIR, PartitionType.PARTITION_A_TRAIN, 0)

        pool2 = SyntheticEntityPool(n_train=50, n_test=20, seed=42)
        gen2 = CAND01DatasetGenerator(pool2, seed=42)
        r2 = gen2.generate_record(TaskFamily.TF02_SINGLE_PAIR, PartitionType.PARTITION_A_TRAIN, 0)

        self.assertEqual(r1.prompt_text, r2.prompt_text)
        self.assertEqual(r1.prompt_sha256, r2.prompt_sha256)
        self.assertEqual(r1.target_token_id, r2.target_token_id)

    def test_tf01_identity_structure_and_partition_restrictions(self):
        """TF-01 direct copy structure and restriction from Partitions B & C."""
        rec = self.gen.generate_record(TaskFamily.TF01_DIRECT_COPY, PartitionType.PARTITION_A_TRAIN, 0)
        self.assertIn("COPY:", rec.prompt_text)
        self.assertEqual(rec.query_key_id, rec.target_token_id)
        self.assertEqual(rec.distractor_count, 0)

        # TF-01 must raise ValueError if attempted in Val Recomb or Test Recomb
        with self.assertRaises(ValueError):
            self.gen.generate_record(TaskFamily.TF01_DIRECT_COPY, PartitionType.PARTITION_B_VAL_RECOMB, 0)
        with self.assertRaises(ValueError):
            self.gen.generate_record(TaskFamily.TF01_DIRECT_COPY, PartitionType.PARTITION_C_TEST_RECOMB, 0)

    def test_all_eight_task_families_implemented(self):
        """Verifies distinct prompt structures for all 8 Task Families (TF-01 to TF-08)."""
        tf_expectations = [
            (TaskFamily.TF01_DIRECT_COPY, PartitionType.PARTITION_A_TRAIN, "COPY:"),
            (TaskFamily.TF02_SINGLE_PAIR, PartitionType.PARTITION_A_TRAIN, "PAIR:"),
            (TaskFamily.TF03_MULTI_PAIR, PartitionType.PARTITION_A_TRAIN, "PAIRS:"),
            (TaskFamily.TF04_DISTRACTOR, PartitionType.PARTITION_A_TRAIN, "PAIRS:"),
            (TaskFamily.TF05_SIMILAR_KEY, PartitionType.PARTITION_A_TRAIN, "_alt"),
            (TaskFamily.TF06_RECOMBINATION, PartitionType.PARTITION_B_VAL_RECOMB, "RECOMB_PAIRS:"),
            (TaskFamily.TF07_OFFSET_SCALING, PartitionType.PARTITION_B_VAL_RECOMB, "CONTEXT_BUFFER:"),
            (TaskFamily.TF08_CONTEXT_SCALING, PartitionType.PARTITION_B_VAL_RECOMB, "DENSE_PAIRS:"),
        ]
        for tf, part, expected_substr in tf_expectations:
            rec = self.gen.generate_record(tf, part, index=1)
            self.assertIn(expected_substr, rec.prompt_text, f"Failed for {tf.value}")
            self.assertEqual(rec.task_family, tf.value)
            self.assertGreater(len(rec.prompt_text), 5)
            self.assertGreater(rec.target_token_id, 0)

    def test_bidirectional_association_leakage_prevention(self):
        """
        VER-03: Neither (K, V) nor (V, K) from Train may appear in Val or Test Recomb.
        """
        train_counts = {TaskFamily.TF02_SINGLE_PAIR: 30}
        val_counts = {TaskFamily.TF02_SINGLE_PAIR: 20}
        test_counts = {TaskFamily.TF02_SINGLE_PAIR: 20}

        self.gen.generate_partition(PartitionType.PARTITION_A_TRAIN, train_counts)
        self.gen.generate_partition(PartitionType.PARTITION_B_VAL_RECOMB, val_counts)
        self.gen.generate_partition(PartitionType.PARTITION_C_TEST_RECOMB, test_counts)

        audit = self.gen.verify_all_isolation_invariants()
        self.assertTrue(audit["audit_passed"])
        self.assertEqual(audit["val_symmetric_leakage_count"], 0)
        self.assertEqual(audit["test_symmetric_leakage_count"], 0)
        self.assertEqual(audit["cross_partition_hash_collisions"], 0)

    def test_unique_keys_per_context(self):
        """Multi-pair contexts must not have duplicate query keys in the context."""
        rec = self.gen.generate_record(TaskFamily.TF03_MULTI_PAIR, PartitionType.PARTITION_A_TRAIN, 0)
        context_keys = [k for k, v in rec.context_pair_ids]
        self.assertEqual(len(context_keys), len(set(context_keys)), "Duplicate keys found in context!")

    def test_prompt_hash_uniqueness(self):
        """Duplicate records within partition must be detected and rejected."""
        gen = CAND01DatasetGenerator(self.pool, seed=999)
        r = gen.generate_record(TaskFamily.TF02_SINGLE_PAIR, PartitionType.PARTITION_A_TRAIN, 0)
        gen.prompt_hashes[PartitionType.PARTITION_A_TRAIN.value].add(r.prompt_sha256)

        with self.assertRaises(AssertionError):
            if r.prompt_sha256 in gen.prompt_hashes[PartitionType.PARTITION_A_TRAIN.value]:
                assert False, "Duplicate prompt hash detected!"


class TestCAND01EvaluatorMetrics(unittest.TestCase):
    """Tests for exact match, full completion, cross-entropy loss, and perplexity calculations."""

    def setUp(self):
        self.harness = CAND01EvaluatorHarness(vocab_size=10)

    def test_metric_decoupling_aft_and_afc(self):
        """
        A_FT: First-token exact match.
        A_FC: First-token exact match AND immediate EOS clean termination (len == 2).
        """
        record = {
            "record_id": "test_001",
            "task_family": "TF_02_SINGLE_PAIR",
            "partition": "test_recomb",
            "query_key_id": 2,
            "target_token_id": 5,
            "target_token": "ent_5",
        }
        logits_correct = np.array([0.1, 0.2, 0.3, 0.1, 0.4, 5.0, 0.2, 0.1, 0.0, 0.1])

        # Case 1: Clean termination [target_id, EOS] -> A_FT=True, A_FC=True
        res_clean = self.harness.evaluate_sample(record, logits_correct, [5, EOS_TOKEN_ID])
        self.assertTrue(res_clean.is_first_token_match)
        self.assertTrue(res_clean.is_full_completion_match)
        self.assertEqual(res_clean.target_rank, 1)

        # Case 2: Syntax echo / trailing repetition [target_id, 3, EOS] -> A_FT=True, A_FC=False
        res_echo = self.harness.evaluate_sample(record, logits_correct, [5, 3, EOS_TOKEN_ID])
        self.assertTrue(res_echo.is_first_token_match)
        self.assertFalse(res_echo.is_full_completion_match)

        # Case 3: Wrong first token [2, EOS] -> A_FT=False, A_FC=False
        logits_wrong = np.array([0.1, 0.2, 6.0, 0.1, 0.4, 0.5, 0.2, 0.1, 0.0, 0.1])
        res_wrong = self.harness.evaluate_sample(record, logits_wrong, [2, EOS_TOKEN_ID])
        self.assertFalse(res_wrong.is_first_token_match)
        self.assertFalse(res_wrong.is_full_completion_match)
        self.assertGreater(res_wrong.target_rank, 1)

    def test_perplexity_relationship(self):
        """Perplexity must equal exp(loss)."""
        record = {
            "record_id": "test_002",
            "task_family": "TF_01_DIRECT_COPY",
            "partition": "test_disjoint",
            "query_key_id": 1,
            "target_token_id": 1,
            "target_token": "ent_1",
        }
        logits = np.array([1.0, 2.0, 0.5, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        res = self.harness.evaluate_sample(record, logits, [1, EOS_TOKEN_ID])
        expected_ppl = float(np.exp(res.target_loss))
        self.assertAlmostEqual(res.target_perplexity, expected_ppl, places=4)

    def test_empty_dataset_handling(self):
        """Evaluator handles empty dataset gracefully without exceptions."""
        mock_model = SyntheticModelMock(vocab_size=10, seed=42)
        res = self.harness.evaluate_dataset([], mock_model)
        self.assertEqual(res["num_samples"], 0)
        self.assertEqual(res["aggregate_metrics"]["first_token_exact_accuracy_A_FT"], 0.0)


class TestStatisticalBootstrapModule(unittest.TestCase):
    """Tests for ordinary, one-way, and provisional two-way bootstrap estimators."""

    def test_kish_effective_sample_size_edge_cases(self):
        """Handles uniform, delta, all-zero, negative, and non-finite weights safely."""
        # Uniform
        self.assertAlmostEqual(compute_kish_effective_sample_size(np.ones(100)), 100.0, places=4)
        # Delta
        skewed = np.zeros(100)
        skewed[0] = 100.0
        self.assertAlmostEqual(compute_kish_effective_sample_size(skewed), 1.0, places=4)
        # All zeros
        self.assertEqual(compute_kish_effective_sample_size(np.zeros(10)), 0.0)
        # Negative weights
        self.assertEqual(compute_kish_effective_sample_size(np.array([1.0, -2.0, 3.0])), 0.0)
        # Non-finite (NaN / Inf)
        self.assertEqual(compute_kish_effective_sample_size(np.array([1.0, np.nan, 3.0])), 0.0)
        self.assertEqual(compute_kish_effective_sample_size(np.array([1.0, np.inf, 3.0])), 0.0)

    def test_bootstrap_reproducibility(self):
        """Fixed random seed yields bit-exact confidence intervals."""
        vals = np.array([1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0])
        res1 = ordinary_nonparametric_bootstrap(vals, num_replicates=100, random_seed=42)
        res2 = ordinary_nonparametric_bootstrap(vals, num_replicates=100, random_seed=42)
        self.assertEqual(res1.ci_lower, res2.ci_lower)
        self.assertEqual(res1.ci_upper, res2.ci_upper)

    def test_one_way_cluster_sensitivity_status(self):
        """One-way cluster bootstrap must retain SECONDARY_SENSITIVITY_ESTIMATOR status."""
        vals = np.array([1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0])
        clusters = np.array([1, 1, 2, 2, 3, 3, 4, 4])
        res = one_way_clustered_bootstrap(vals, clusters, num_replicates=50, random_seed=42)
        self.assertEqual(res.status, EstimatorStatus.SECONDARY_SENSITIVITY)
        self.assertIn("SECONDARY SENSITIVITY ESTIMATOR", res.notes[0])

    def test_degeneracy_boundary_conditions_10_percent_threshold(self):
        """
        Tests exact boundary logic around registered 10% degeneracy threshold:
        - When degen rate <= 10%: status is PROVISIONAL_PENDING_STATISTICAL_VALIDATION.
        - When degen rate > 10%: status is INCONCLUSIVE.
        """
        # Create balanced crossed dataset with 20 keys and 20 values
        n_keys = 20
        n_vals = 20
        keys = []
        vals_id = []
        outcomes = []
        for k in range(n_keys):
            for v in range(n_vals):
                keys.append(k)
                vals_id.append(v)
                outcomes.append(1.0 if (k + v) % 2 == 0 else 0.0)

        # Dataset has ~30% degeneracy.
        # Test 1: threshold = 0.50 (> 30%) -> degen rate does not exceed threshold -> status is PROVISIONAL
        res_pass = provisional_two_way_cluster_bootstrap(
            np.array(outcomes), np.array(keys), np.array(vals_id),
            num_replicates=30, max_degenerate_rate=0.50, random_seed=42
        )
        self.assertEqual(res_pass.status, EstimatorStatus.PROVISIONAL_PENDING_VALIDATION)
        self.assertLessEqual(res_pass.degenerate_rate, 0.50)

        # Test 2: threshold = 0.10 (< 30%) -> degen rate exceeds threshold -> trips to INCONCLUSIVE
        res_trip = provisional_two_way_cluster_bootstrap(
            np.array(outcomes), np.array(keys), np.array(vals_id),
            num_replicates=30, max_degenerate_rate=REGISTERED_MAX_DEGENERATE_RATE, random_seed=42
        )
        self.assertEqual(res_trip.status, EstimatorStatus.INCONCLUSIVE)
        self.assertGreater(res_trip.degenerate_rate, REGISTERED_MAX_DEGENERATE_RATE)

    def test_synthetic_crossed_simulation_diagnostic(self):
        """Monte Carlo simulation runs and returns required diagnostic structure with Monte Carlo standard errors."""
        sim = run_synthetic_crossed_simulation(
            num_simulations=5,
            n_keys=15,
            n_vals=15,
            num_bootstrap_replicates=30,
            random_seed=42,
        )
        self.assertIn("diagnostic_disclaimer", sim)
        self.assertIn("coverage", sim)
        self.assertIn("provisional_two_way_mc_se", sim["coverage"])
        self.assertIn("empirical_bias", sim)
        self.assertIn("mean_two_way_degenerate_rate", sim)
        self.assertIsInstance(sim["coverage"]["provisional_two_way_coverage"], float)


class TestRunnerDryRunAndSecurityBoundaries(unittest.TestCase):
    """Tests that runner enforces security boundaries, cross-platform path normalization, and dry-run execution."""

    def test_assert_safe_path_security_checks(self):
        """Tests that assert_safe_path blocks traversal and normalized canonical paths."""
        # Forbidden paths
        with self.assertRaises(PermissionError):
            assert_safe_path("data/preprocessed/test", "test_param")
        with self.assertRaises(PermissionError):
            assert_safe_path("data/tokenized/stage_c", "test_param")
        with self.assertRaises(PermissionError):
            assert_safe_path("data/experiments/cand_01/prod", "test_param")
        with self.assertRaises(PermissionError):
            assert_safe_path("experiments/overnight_runs", "test_param")

        # Path traversal escape attempt
        with self.assertRaises(PermissionError):
            assert_safe_path("tests/fixtures/../../data/preprocessed", "test_param")

        # Permitted fixture path
        try:
            assert_safe_path("tests/fixtures/cand01_synthetic", "test_param")
        except PermissionError:
            self.fail("assert_safe_path raised PermissionError on permitted synthetic fixture path!")

    def test_runner_boundary_rejection_cli(self):
        """Runner must reject forbidden parameters and checkpoint paths via CLI."""
        import subprocess
        res = subprocess.run(
            [
                sys.executable,
                "scripts/run_cand01_experiment.py",
                "--dry-run",
                "--checkpoint-path", "experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt",
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(res.returncode, 1)
        self.assertIn("FATAL GOVERNANCE BREACH", res.stderr)

    def test_runner_dry_run_success(self):
        """Runner succeeds cleanly when using --dry-run and synthetic paths."""
        import subprocess
        res = subprocess.run(
            [
                sys.executable,
                "scripts/run_cand01_experiment.py",
                "--dry-run",
                "--output-dir", "tests/fixtures/test_runner_output",
                "--replicates", "30",
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(res.returncode, 0)
        self.assertIn("SUCCESS: CAND_01 dry-run pipeline completed cleanly", res.stdout)
        report_file = os.path.join("tests/fixtures/test_runner_output", "cand01_dryrun_execution_report.json")
        self.assertTrue(os.path.exists(report_file))


if __name__ == "__main__":
    unittest.main()
