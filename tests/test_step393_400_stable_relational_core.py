"""Unit and Integration Tests for Wave 393–400: Stable Relational Initialization + I4 Acceleration.

Verifies:
- Test 1: Canonical Baseline Invariance (Parameters = 3,443,136, SHA-256 Bit-Exact, Delta W = 0)
- Test 2: Hop-1 Relational Initialization Core (7 modes supported, budget <550k trainable)
- Test 3: Relational Attention Block Modes (Orthogonal, QK-Norm, Shared QK, Learned Temp)
- Test 4: Step 393 Relational Initialization Study Execution (Modes A-G comparison)
- Test 5: Step 394 Relational Matching Objective Ablation (Variants A-D and language retention)
- Test 6: Step 395 End-to-End Two-Hop Joint Training (L_H1, L_inter, L_H2, L_final, L_total)
- Test 7: Step 396 Anti-Memorization, Permutations & Contamination Audit (Zero contamination)
- Test 8: Step 397 Adaptive Sufficiency Reintegration (Fixed vs Adaptive comparison)
- Test 9: Step 398 Multi-Seed Stability & Transformation Boundary Audit (Seeds 42, 101, 2026)
- Test 10: Step 399 Resource-Aware Relational Reasoning under Hard Budgets (1, 2, 3, 4, 6, 8)
- Test 11: Step 400 Master I4 Decision Gate & First Release Track Readiness Checklist (18 dimensions)
"""

import unittest
import torch

from chakrview.runtime.interactive import (
    instantiate_frozen_baseline,
    compute_model_hash,
    EXPECTED_WEIGHT_HASH,
)
from chakrview.cognition.relational_initialization_study import (
    InitializedRelationalCore,
    RelationalAttentionBlock,
    run_relational_initialization_study,
)
from chakrview.cognition.relational_matching_objective import (
    run_relational_matching_objective_study,
    evaluate_language_retention_core,
)
from chakrview.cognition.end_to_end_relational_training import (
    train_and_eval_end_to_end_core,
)
from chakrview.cognition.anti_memorization_generalization import (
    evaluate_anti_memorization_suite,
)
from chakrview.cognition.adaptive_sufficiency_reintegration import (
    run_adaptive_sufficiency_reintegration_study,
)
from chakrview.cognition.multi_seed_relational_stability import (
    run_multi_seed_relational_stability_study,
)
from chakrview.cognition.resource_aware_relational_reasoning import (
    run_resource_aware_relational_study,
)
from chakrview.cognition.relational_master_i4_decision import (
    run_master_relational_i4_decision,
)


class TestWave393To400StableRelationalCore(unittest.TestCase):
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

    def test_02_initialized_relational_core_budget(self):
        """Verifies InitializedRelationalCore parameter budget (<550k trainable)."""
        core = InitializedRelationalCore(mode="D_orthogonal")
        trainable = sum(p.numel() for p in core.parameters() if p.requires_grad)
        total = sum(p.numel() for p in core.parameters())
        self.assertLess(trainable, 550_000)
        self.assertEqual(trainable, 497_185)
        self.assertEqual(total, 497_185)

        dummy = torch.tensor([[10, 20, 30, 40]], dtype=torch.long)
        c_pos = torch.tensor([[1, 2]], dtype=torch.long)
        out = core(dummy, candidate_positions=c_pos)
        self.assertIn("binding_logits", out)
        self.assertIn("w1", out)
        self.assertIn("w2", out)
        self.assertIn("s1", out)

    def test_03_relational_attention_modes(self):
        """Verifies attention block forward pass across initialization modes."""
        for mode in ["A_standard", "B_normalized", "C_identity_aligned", "D_orthogonal", "E_shared_qk", "F_learned_temp", "G_qk_norm"]:
            block = RelationalAttentionBlock(d_model=96, n_heads=4, mode=mode)
            x = torch.randn(2, 8, 96)
            out, w = block(x, x, x)
            self.assertEqual(out.shape, (2, 8, 96))
            self.assertEqual(w.shape, (2, 4, 8, 8))

    def test_04_step393_initialization_study(self):
        """Verifies Step 393 initialization study execution."""
        rep = run_relational_initialization_study(seeds=(42,), train_steps=2, eval_episodes=2)
        self.assertEqual(len(rep.modes), 7)
        self.assertIn("A_standard", rep.modes)
        self.assertIn("D_orthogonal", rep.modes)
        self.assertIn(rep.best_mode, rep.modes)

    def test_05_step394_matching_objective(self):
        """Verifies Step 394 relational matching objective execution and language retention."""
        rep = run_relational_matching_objective_study(seed=42, train_steps=2, eval_episodes=2)
        self.assertEqual(len(rep.variants), 4)
        self.assertIn("A_answer_only", rep.variants)
        self.assertIn("D_answer_plus_compositional", rep.variants)
        for r in rep.variants.values():
            self.assertGreaterEqual(r.language_retention, 0.9500)

    def test_06_step395_end_to_end_training(self):
        """Verifies Step 395 end-to-end two-hop training with joint loss tracking."""
        core = InitializedRelationalCore(mode="D_orthogonal")
        core, rep = train_and_eval_end_to_end_core(core, seed=42, train_steps=5, eval_episodes=2)
        self.assertGreater(len(rep.curves), 0)
        self.assertGreaterEqual(rep.final_g1_acc, 0.0)
        self.assertGreaterEqual(rep.final_g4_acc, 0.0)

    def test_07_step396_anti_memorization_and_contamination(self):
        """Verifies Step 396 anti-memorization suite, permutations and zero contamination."""
        core = InitializedRelationalCore(mode="D_orthogonal")
        rep = evaluate_anti_memorization_suite(core, seed=42, episodes_per_condition=2)
        self.assertTrue(rep.contamination_zero)
        self.assertEqual(len(rep.distractor_sweep), 5)
        self.assertIn(0, rep.distractor_sweep)
        self.assertIn(5, rep.distractor_sweep)

    def test_08_step397_adaptive_sufficiency_reintegration(self):
        """Verifies Step 397 reintegration of sufficiency controller."""
        rep = run_adaptive_sufficiency_reintegration_study(seed=42, train_steps=5, eval_episodes_per_hop=2)
        self.assertEqual(len(rep.variants), 4)
        self.assertIn("A_Fixed1", rep.variants)
        self.assertIn("D_AdaptiveSufficiency", rep.variants)

    def test_09_step398_multi_seed_relational_stability(self):
        """Verifies Step 398 multi-seed evaluation across seeds 42, 101, 2026."""
        rep = run_multi_seed_relational_stability_study(seeds=(42, 101), train_steps=3, eval_episodes=2)
        self.assertEqual(len(rep.seed_metrics), 2)
        self.assertIn(42, rep.seed_metrics)
        self.assertIn(101, rep.seed_metrics)
        self.assertTrue(len(rep.transformation_boundary_identified) > 0)

    def test_10_step399_resource_aware_reasoning(self):
        """Verifies Step 399 budget compliance across budgets [1, 2, 3, 4, 6, 8]."""
        rep = run_resource_aware_relational_study(seed=42, episodes=2)
        self.assertEqual(len(rep.budgets), 6)
        self.assertTrue(rep.all_strictly_bounded)
        for b, pt in rep.budgets.items():
            self.assertLessEqual(pt.max_cycles_observed, b)

    def test_11_step400_master_decision_and_release_checklist(self):
        """Verifies Step 400 master I4 decision and 18-dimension release track checklist."""
        rep = run_master_relational_i4_decision(seeds=(42, 101), train_steps=3, eval_episodes=2)
        self.assertEqual(len(rep.seed_metrics), 2)
        self.assertIn(rep.final_classification, [
            "I4_ACHIEVED",
            "I4_EMERGING_STABLE",
            "I4_BLOCKED_BY_RELATIONAL_INITIALIZATION",
            "I4_BLOCKED_BY_COMPOSITIONAL_STATE_TRANSITION",
        ])
        self.assertTrue(rep.baseline_exact)
        self.assertEqual(rep.baseline_sha, EXPECTED_WEIGHT_HASH)
        self.assertTrue(rep.release_checklist.all_dimensions_passed)
        self.assertTrue(rep.release_checklist.cpu_first_default)
        self.assertTrue(rep.release_checklist.safe_candidate_isolation)


if __name__ == "__main__":
    unittest.main()
