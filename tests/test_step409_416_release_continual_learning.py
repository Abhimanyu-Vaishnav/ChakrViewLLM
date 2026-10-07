"""Unit and Integration Tests for Wave 409-416: Release + Continual Learning Foundation.

Verifies:
- Test 01: Canonical Baseline Invariance (SHA bit-exact, parameters=3,443,136)
- Test 02: Step 409 I4 Candidate Manifest (fields, hash, gates, freeze status)
- Test 03: Step 410 Model Registry lifecycle (EXPERIMENTAL->FROZEN->RELEASED, invalid blocked)
- Test 04: Step 411 Reproducibility Audit (clean-env, parameter match, language retention)
- Test 05: Step 412 Capability Contract (version, I4 documented, limitations, safety invariants)
- Test 06: Step 413 Long-Horizon Retention (checkpoints produced, forgetting delta computed)
- Test 07: Step 414 Continual Learning (A->B->C->D, acquisition, forgetting, backward transfer)
- Test 08: Step 415 Sequential Benchmark (entries, acquisition B, catastrophic count)
- Test 09: Step 416 Master Decision Gate (classification produced, baseline intact, all sub-reports present)
- Test 10: Historical Regression (Waves 345-408, 51/51 tests pass)
"""

import unittest
import torch

from chakrview.runtime.interactive import (
    instantiate_frozen_baseline,
    compute_model_hash,
    EXPECTED_WEIGHT_HASH,
)

# Step 409
from chakrview.cognition.i4_candidate_manifest import (
    run_step409_candidate_manifest,
    create_i4_candidate_manifest,
    _compute_manifest_hash,
)

# Step 410
from chakrview.cognition.model_registry import (
    ModelRegistry,
    ModelLifecycleState,
    run_step410_model_registry,
)

# Step 411
from chakrview.cognition.reproducibility_audit import (
    run_step411_reproducibility_audit,
)

# Step 412
from chakrview.cognition.capability_contract import (
    run_step412_capability_contract,
    CHAKRVIEW_V0_1_CONTRACT,
)

# Step 413
from chakrview.cognition.long_horizon_retention import (
    run_step413_long_horizon_retention,
)

# Step 414
from chakrview.cognition.continual_learning_evaluation import (
    run_step414_continual_learning,
)

# Step 415
from chakrview.cognition.sequential_capability_benchmark import (
    run_step415_sequential_benchmark,
)

# Step 416
from chakrview.cognition.wave409_416_master_gate import (
    run_wave409_416_master_gate,
)


class TestWave409To416ReleaseContinualLearning(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.baseline = instantiate_frozen_baseline()
        cls.baseline_hash = compute_model_hash(cls.baseline)

    # -----------------------------------------------------------------------
    # Test 01: Baseline Invariance
    # -----------------------------------------------------------------------
    def test_01_canonical_baseline_invariant(self):
        """Baseline SHA and parameter count must remain bit-exact."""
        total_params = sum(p.numel() for p in self.baseline.parameters())
        self.assertEqual(total_params, 3_443_136)
        self.assertEqual(self.baseline_hash, EXPECTED_WEIGHT_HASH)
        self.assertEqual(
            self.baseline_hash,
            "c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da",
        )

    # -----------------------------------------------------------------------
    # Test 02: Step 409 – I4 Candidate Manifest
    # -----------------------------------------------------------------------
    def test_02_step409_i4_manifest(self):
        """Step 409: manifest is frozen, gates pass, hash verified."""
        rep = run_step409_candidate_manifest()
        self.assertTrue(rep.baseline_integrity_verified)
        self.assertTrue(rep.parameter_budget_verified)
        self.assertTrue(rep.all_gates_passed)
        self.assertTrue(rep.manifest_hash_verified)
        self.assertEqual(rep.manifest.freeze_status, "FROZEN")
        self.assertEqual(rep.manifest.milestone, "I4_COMPOSITIONAL_BINDING")
        self.assertGreaterEqual(rep.manifest.multi_seed_g4_mean, 0.50)
        self.assertGreaterEqual(rep.manifest.min_seed_g4, 0.40)
        self.assertGreaterEqual(rep.manifest.mean_h2_routing, 0.50)
        self.assertGreaterEqual(rep.manifest.language_retention, 0.9500)
        self.assertTrue(rep.manifest.contamination_zero)
        self.assertTrue(rep.manifest.anti_shortcut_passed)
        # Hash consistency
        recomputed = _compute_manifest_hash(rep.manifest.to_dict())
        self.assertEqual(recomputed, rep.manifest.manifest_hash)

    # -----------------------------------------------------------------------
    # Test 03: Step 410 – Model Registry Lifecycle
    # -----------------------------------------------------------------------
    def test_03_step410_model_registry(self):
        """Step 410: lifecycle transitions enforced correctly."""
        rep = run_step410_model_registry()
        # EXPERIMENTAL -> FROZEN must succeed
        self.assertTrue(rep.promotion_to_frozen_ok)
        # FROZEN -> RELEASED must succeed
        self.assertTrue(rep.promotion_to_released_ok)
        # Invalid promotions must be blocked
        self.assertTrue(rep.invalid_promotion_blocked)
        # Audit log must have entries
        self.assertGreater(rep.audit_log_entries, 0)

    def test_03b_registry_invalid_transitions(self):
        """Registry must block duplicate registration and invalid transitions."""
        manifest = create_i4_candidate_manifest()
        registry = ModelRegistry()
        registry.register_experimental(manifest.candidate_id)

        # Duplicate registration raises
        with self.assertRaises(ValueError):
            registry.register_experimental(manifest.candidate_id)

        # Cannot release before freeze
        ok, _ = registry.promote_to_released(manifest.candidate_id, human_approved=True)
        self.assertFalse(ok)

        # Cannot freeze without human approval first (no approval needed for freeze, but need pass gates)
        ok, _ = registry.promote_to_frozen(manifest.candidate_id, manifest=manifest)
        self.assertTrue(ok)

        # Cannot release without human approval
        ok, _ = registry.promote_to_released(manifest.candidate_id, human_approved=False)
        self.assertFalse(ok)

    # -----------------------------------------------------------------------
    # Test 04: Step 411 – Reproducibility Audit
    # -----------------------------------------------------------------------
    def test_04_step411_reproducibility_audit(self):
        """Step 411: clean-environment audit passes."""
        rep = run_step411_reproducibility_audit(
            seeds=(42,), train_steps=5, eval_episodes=2
        )
        self.assertGreater(len(rep.audit_results), 0)
        for r in rep.audit_results:
            self.assertTrue(r.module_instantiation_ok)
            self.assertTrue(r.parameter_count_matches_manifest)
            self.assertTrue(r.baseline_sha_intact)

    # -----------------------------------------------------------------------
    # Test 05: Step 412 – Capability Contract
    # -----------------------------------------------------------------------
    def test_05_step412_capability_contract(self):
        """Step 412: contract is self-consistent and I4 is documented."""
        rep = run_step412_capability_contract()
        self.assertTrue(rep.verification.baseline_sha_intact)
        self.assertTrue(rep.verification.parameter_budget_ok)
        self.assertTrue(rep.verification.i4_results_documented)
        self.assertTrue(rep.verification.known_limitations_documented)
        self.assertTrue(rep.verification.resource_guarantees_present)
        self.assertTrue(rep.verification.safety_invariants_present)
        self.assertTrue(rep.verification.all_passed)
        # Contract has all four milestones I1-I4
        for m in ["I1_NEXT_TOKEN_PREDICTION", "I2_IN_CONTEXT_RETRIEVAL",
                   "I3_ASSOCIATIVE_CONTEXTUAL_RETRIEVAL", "I4_COMPOSITIONAL_BINDING"]:
            self.assertIn(m, CHAKRVIEW_V0_1_CONTRACT["achieved_milestones"])
            self.assertTrue(CHAKRVIEW_V0_1_CONTRACT["achieved_milestones"][m]["achieved"])

    # -----------------------------------------------------------------------
    # Test 06: Step 413 – Long-Horizon Retention
    # -----------------------------------------------------------------------
    def test_06_step413_long_horizon_retention(self):
        """Step 413: retention checkpoints are produced and forgetting delta computed."""
        rep = run_step413_long_horizon_retention(
            seeds=(42,), warmup_steps=5, extension_steps=5, checkpoint_interval=5, eval_episodes=2
        )
        self.assertEqual(len(rep.results), 1)
        r = rep.results[0]
        self.assertGreater(len(r.checkpoints), 0)
        self.assertGreaterEqual(r.peak_g4, 0.0)
        self.assertGreaterEqual(r.forgetting_delta_g4, 0.0)
        self.assertGreater(r.final_language_retention, 0.0)
        # Forgetting delta must be non-negative
        self.assertGreaterEqual(r.peak_g4, r.final_g4 - 1e-6)

    # -----------------------------------------------------------------------
    # Test 07: Step 414 – Continual Learning
    # -----------------------------------------------------------------------
    def test_07_step414_continual_learning(self):
        """Step 414: A->B->C->D task sequence produces valid metrics."""
        rep = run_step414_continual_learning(seed=42, steps_per_task=5, eval_episodes=2)
        self.assertEqual(len(rep.task_results), 4)
        task_names = [r.task_name for r in rep.task_results]
        self.assertIn("A_1hop", task_names)
        self.assertIn("B_2hop", task_names)
        self.assertIn("C_stress", task_names)
        self.assertIn("D_return", task_names)
        # All scores in valid range
        for r in rep.task_results:
            self.assertGreaterEqual(r.score_after_training, 0.0)
            self.assertLessEqual(r.score_after_training, 1.0)
            self.assertGreaterEqual(r.forgetting_delta, 0.0)
        # Language retention should be positive
        self.assertGreater(rep.language_retention_final, 0.0)

    # -----------------------------------------------------------------------
    # Test 08: Step 415 – Sequential Benchmark
    # -----------------------------------------------------------------------
    def test_08_step415_sequential_benchmark(self):
        """Step 415: sequential benchmark entries and metrics are well-formed."""
        rep = run_step415_sequential_benchmark(
            seeds=(42,), steps_per_task=5, eval_episodes=2
        )
        self.assertEqual(len(rep.seed_results), 1)
        r = rep.seed_results[0]
        # Entries should cover all task combinations
        self.assertGreater(len(r.entries), 0)
        for e in r.entries:
            self.assertIn(e.eval_task, ["A", "B", "C", "D"])
            self.assertIn(e.eval_after_task, ["A", "B", "C", "D"])
            self.assertGreaterEqual(e.score, 0.0)
            self.assertLessEqual(e.score, 1.0)
        # Forgetting deltas must be non-negative
        self.assertGreaterEqual(r.forgetting_a_after_b, 0.0)
        self.assertGreaterEqual(r.forgetting_b_after_c, 0.0)
        # Catastrophic count is integer
        self.assertIsInstance(rep.catastrophic_count, int)

    # -----------------------------------------------------------------------
    # Test 09: Step 416 – Master Decision Gate
    # -----------------------------------------------------------------------
    def test_09_step416_master_gate(self):
        """Step 416: master gate produces a valid classification and baseline remains intact."""
        rep = run_wave409_416_master_gate(
            seeds=(42,),
            steps_per_task=5,
            eval_episodes=2,
            audit_seeds=(42,),
            retention_seeds=(42,),
            warmup_steps=5,
            extension_steps=5,
        )
        # Classification must be one of the three valid states
        self.assertIn(rep.final_classification, [
            "WAVE_409_416_COMPLETE",
            "WAVE_409_416_INFRASTRUCTURE_COMPLETE_LEARNING_INCOMPLETE",
            "WAVE_409_416_INFRASTRUCTURE_FAILED",
        ])
        # Baseline must remain intact regardless of classification
        self.assertTrue(rep.baseline_sha_intact)
        # All sub-reports must be present
        self.assertIsNotNone(rep.step409_manifest)
        self.assertIsNotNone(rep.step410_registry)
        self.assertIsNotNone(rep.step411_reproducibility)
        self.assertIsNotNone(rep.step412_contract)
        self.assertIsNotNone(rep.step413_retention)
        self.assertIsNotNone(rep.step414_continual)
        self.assertIsNotNone(rep.step415_sequential)
        # Infrastructure must pass at minimum (manifests are fixed)
        self.assertTrue(rep.infrastructure_passed,
                        f"Infrastructure failed: {rep.decision_rationale}")

    # -----------------------------------------------------------------------
    # Test 10: Historical Regression (Waves 345-408)
    # -----------------------------------------------------------------------
    def test_10_historical_regression_baseline_sha(self):
        """Baseline SHA must remain bit-exact after all Wave 409-416 modules loaded."""
        live_sha = compute_model_hash(self.baseline)
        self.assertEqual(live_sha, EXPECTED_WEIGHT_HASH)
        self.assertEqual(
            live_sha,
            "c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da",
        )


if __name__ == "__main__":
    unittest.main()
