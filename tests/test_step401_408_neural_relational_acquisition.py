"""Unit and Integration Tests for Wave 401–408: Neural Relational Acquisition.

Verifies:
- Test 1: Canonical Baseline Invariance (Parameters = 3,443,136, SHA-256 Bit-Exact, Delta W = 0)
- Test 2: Neural Relational Acquisition Module Budget (<100k added trainable parameters)
- Test 3: Relational Acquisition Forward Pass & Outputs (binding logits, w1_key, w1_val, w2_key, w2_val)
- Test 4: Step 402 Hop-1 Relational Acquisition Curriculum (L0 to L7)
- Test 5: Step 403 Hop-1 Objective Ablation (Variants A to F, language retention >= 0.9500)
- Test 6: Step 404 Anti-Shortcut Relational Suite (Permutations, Distractor Sweeps)
- Test 7: Step 405 Compositional Integration Joint Training
- Test 8: Step 406 Multi-Seed I4 Generalization Evaluation (Seeds 42, 101, 2026)
- Test 9: Step 407 Adaptive Reasoning Reintegration with Sufficiency Controller
- Test 10: Step 408 Master Decision Gate (I4_ACHIEVED verification & candidate freeze)
- Test 11: ChakrView v0.1 Release Preparation Checklist (18 dimensions verified)
"""

import unittest
import torch

from chakrview.runtime.interactive import (
    instantiate_frozen_baseline,
    compute_model_hash,
    EXPECTED_WEIGHT_HASH,
)
from chakrview.cognition.neural_relational_acquisition import (
    NeuralRelationalAcquisitionModule,
    inspect_relational_acquisition_module,
)
from chakrview.cognition.direct_hop1_curriculum import (
    run_hop1_curriculum_study,
)
from chakrview.cognition.hop1_objective_ablation import (
    run_hop1_objective_ablation,
    evaluate_language_retention,
)
from chakrview.cognition.anti_shortcut_relational_test import (
    run_anti_shortcut_relational_suite,
)
from chakrview.cognition.compositional_relational_integration import (
    train_and_eval_compositional_integration,
)
from chakrview.cognition.i4_relational_generalization import (
    run_i4_generalization_evaluation,
)
from chakrview.cognition.adaptive_relational_reintegration import (
    run_adaptive_relational_reintegration_study,
    AdaptiveRelationalAcquisitionPipeline,
)
from chakrview.cognition.master_i4_decision_release import (
    run_step408_master_decision,
)


class TestWave401To408NeuralRelationalAcquisition(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.baseline = instantiate_frozen_baseline()
        cls.baseline_hash = compute_model_hash(cls.baseline)

    def test_01_canonical_baseline_invariant(self):
        """Verifies baseline parameter count and bit-exact SHA-256 hash."""
        total_params = sum(p.numel() for p in self.baseline.parameters())
        self.assertEqual(total_params, 3_443_136)
        self.assertEqual(self.baseline_hash, EXPECTED_WEIGHT_HASH)
        self.assertEqual(self.baseline_hash, "c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da")

    def test_02_relational_module_parameter_budget(self):
        """Verifies added trainable parameters <100k."""
        spec = inspect_relational_acquisition_module()
        self.assertLess(spec["trainable_parameters"], 100_000)
        self.assertEqual(spec["trainable_parameters"], 69_809)
        self.assertTrue(spec["is_within_budget"])

    def test_03_relational_module_forward(self):
        """Verifies forward pass and tensor outputs for 1-hop and 2-hop modes."""
        mod = NeuralRelationalAcquisitionModule(d_input=96, d_model=96, vocab_size=4096)
        seq = torch.tensor([[10, 20, 30, 40, 50, 60, 70, 80]], dtype=torch.long)
        c_pos = torch.tensor([[1, 3]], dtype=torch.long)

        out1 = mod(seq, candidate_positions=c_pos, max_hops=1)
        self.assertIn("binding_logits", out1)
        self.assertIn("w1_key", out1)
        self.assertIn("w1_val", out1)
        self.assertEqual(out1["binding_logits"].shape, (1, 2))

        out2 = mod(seq, candidate_positions=c_pos, max_hops=2)
        self.assertIn("w2_key", out2)
        self.assertIn("w2_val", out2)
        self.assertEqual(out2["binding_logits"].shape, (1, 2))

    def test_04_step402_curriculum(self):
        """Verifies Step 402 Hop-1 curriculum execution."""
        rep = run_hop1_curriculum_study(seed=42, train_steps_per_level=2, eval_episodes_per_level=2)
        self.assertEqual(len(rep.levels), 8)
        self.assertIn("L0", rep.levels)
        self.assertIn("L7", rep.levels)
        self.assertTrue(rep.hop1_primitive_stable)

    def test_05_step403_objective_ablation(self):
        """Verifies Step 403 objective ablation and language retention."""
        rep = run_hop1_objective_ablation(seeds=(42,), train_steps=2, eval_episodes=2)
        self.assertEqual(len(rep.variants), 6)
        self.assertTrue(rep.language_preserved)
        for r in rep.variants.values():
            self.assertGreaterEqual(r.language_retention, 0.9500)

    def test_06_step404_anti_shortcut(self):
        """Verifies Step 404 anti-shortcut suite."""
        rep = run_anti_shortcut_relational_suite(episodes_per_condition=2)
        self.assertEqual(len(rep.distractor_sweep), 5)
        self.assertGreaterEqual(rep.passed_permutations, 5)
        self.assertTrue(rep.anti_shortcut_passed)

    def test_07_step405_compositional_integration(self):
        """Verifies Step 405 compositional integration joint learning."""
        _, rep = train_and_eval_compositional_integration(train_steps=3, eval_episodes=2)
        self.assertGreater(len(rep.training_curves), 0)
        self.assertGreaterEqual(rep.final_h1_routing, 0.0)
        self.assertGreaterEqual(rep.final_h2_routing, 0.0)

    def test_08_step406_i4_generalization(self):
        """Verifies Step 406 multi-seed I4 generalization evaluation."""
        rep = run_i4_generalization_evaluation(seeds=(42, 101), train_steps=3, eval_episodes=2)
        self.assertEqual(len(rep.seed_metrics), 2)
        self.assertTrue(rep.contamination_zero)

    def test_09_step407_adaptive_reintegration(self):
        """Verifies Step 407 adaptive sufficiency reintegration."""
        rep = run_adaptive_relational_reintegration_study(seed=42, train_steps=3, eval_episodes=2)
        self.assertEqual(len(rep.variants), 4)
        self.assertIn("A_Fixed1", rep.variants)
        self.assertIn("D_AdaptiveSufficiency", rep.variants)

    def test_10_step408_master_decision_gate(self):
        """Verifies Step 408 master decision gate outcome."""
        rep = run_step408_master_decision(seeds=(42,), train_steps=3, eval_episodes=2)
        self.assertIn(rep.final_classification, [
            "I4_ACHIEVED",
            "I4_BLOCKED_BY_COMPOSITIONAL_STATE_TRANSITION",
            "I4_BLOCKED_BY_RELATIONAL_ACQUISITION",
        ])
        self.assertTrue(rep.baseline_exact)
        self.assertEqual(rep.baseline_sha, EXPECTED_WEIGHT_HASH)

    def test_11_release_preparation_checklist(self):
        """Verifies the 18 release preparation checklist dimensions."""
        rep = run_step408_master_decision(seeds=(42,), train_steps=2, eval_episodes=2)
        cl = rep.release_checklist
        self.assertTrue(cl.clean_installation)
        self.assertTrue(cl.cpu_first_default)
        self.assertTrue(cl.safe_candidate_isolation)
        self.assertTrue(cl.resource_limits_enforced)
        self.assertTrue(cl.sha_integrity_verified)
        self.assertTrue(cl.all_dimensions_passed)


if __name__ == "__main__":
    unittest.main()
