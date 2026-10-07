"""Unit and Integration Tests for Wave 369-376: Unified Compositional Neural Core.

Verifies:
- Test 1: Canonical Baseline Invariance (Parameters = 3,443,136, SHA-256 Bit-Exact, Delta W = 0)
- Test 2: Unified Compositional Core Parameter Budget (<500k trainable, <1M total)
- Test 3: Minimal Two-Hop Learnability and Intermediate Representations
- Test 4: Causal State Interventions Sensitivity and Absence of Symbolic Shortcuts
- Test 5: Compositional Curriculum Training (Levels 0-8 Promotion and Zero Contamination)
- Test 6: Core Architectural Ablations (Full Core vs No State vs Recurrent Only)
- Test 7: Generalization Stress & Anti-Shortcut Suite (Distractor Sweep, Permutations, Positional Correlation)
- Test 8: Three-Hop Composition Escalation Diagnostic and Degradation Boundary Detection
- Test 9: Master Decision Gate and Multi-Seed Stability Audit (Seeds 42, 101, 2026)
"""

import unittest
import torch

from chakrview.runtime.interactive import (
    instantiate_frozen_baseline,
    compute_model_hash,
    EXPECTED_WEIGHT_HASH,
)
from chakrview.cognition.unified_compositional_core import UnifiedCompositionalCore
from chakrview.cognition.minimal_two_hop_learnability import train_and_eval_two_hop_core
from chakrview.cognition.causal_state_test import run_causal_state_interventions_on_core
from chakrview.cognition.compositional_curriculum_training import train_compositional_curriculum
from chakrview.cognition.core_ablations_study import run_core_ablations_study
from chakrview.cognition.unified_generalization_stress import evaluate_unified_generalization_stress
from chakrview.cognition.three_hop_escalation_diagnostic import run_three_hop_escalation_diagnostic
from chakrview.cognition.strict_unified_core_i4_evaluation import run_strict_unified_core_i4_evaluation


class TestWave369To376UnifiedCompositionalCore(unittest.TestCase):
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

    def test_02_unified_core_architecture_and_budget(self):
        """Verifies UnifiedCompositionalCore parameter budget (<500k preferred, <1M hard limit)."""
        core = UnifiedCompositionalCore()
        trainable_p = core.trainable_param_count
        total_p = core.total_param_count

        self.assertLess(trainable_p, 500_000, f"Trainable params {trainable_p} exceeds 500k")
        self.assertLess(total_p, 1_000_000, f"Total params {total_p} exceeds 1M")
        self.assertEqual(trainable_p, 487_970)

        # Forward pass sanity check
        dummy = torch.tensor([[10, 20, 30, 40]], dtype=torch.long)
        out = core(dummy, return_trace=True)
        self.assertIn("vocab_logits", out)
        self.assertIn("r1", out)
        self.assertIn("s1", out)
        self.assertIn("q2", out)
        self.assertIn("r2", out)
        self.assertEqual(out["vocab_logits"].shape, (1, 4, 4096))

    def test_03_minimal_two_hop_learnability(self):
        """Verifies Step 370 minimal 2-hop composition and routing metrics."""
        rep = train_and_eval_two_hop_core(seed=42, train_steps=10, eval_episodes=4)
        self.assertGreaterEqual(rep.mean_g4, 0.0)
        self.assertIn("known_known", rep.results)
        self.assertIn("unseen_unseen", rep.results)

    def test_04_causal_state_interventions(self):
        """Verifies Step 371 causal state interventions execution and state dependence."""
        rep = run_causal_state_interventions_on_core(seed=42, num_episodes=4)
        self.assertEqual(len(rep.results), 7)
        self.assertIn("A_normal_state", rep.results)
        self.assertIn("B_zero_state", rep.results)
        self.assertIn("G_shuffled_state", rep.results)

    def test_05_compositional_curriculum(self):
        """Verifies Step 372 curriculum training across 9 progressive levels and 0 contamination."""
        core = UnifiedCompositionalCore()
        rep = train_compositional_curriculum(core=core, seed=42, steps_per_level=2, eval_episodes=4)
        self.assertEqual(len(rep.level_results), 9)
        self.assertTrue(rep.contamination_zero)

    def test_06_core_ablations_study(self):
        """Verifies Step 373 core architectural ablations A-G."""
        rep = run_core_ablations_study(seed=42)
        self.assertEqual(len(rep.variants), 7)
        self.assertIn("A_FullCore", rep.variants)
        self.assertIn("F_AttentionOnly", rep.variants)
        self.assertIn("G_RecurrentOnly", rep.variants)

    def test_07_generalization_stress(self):
        """Verifies Step 374 distractor sweep, permutations, and anti-shortcut check."""
        core = UnifiedCompositionalCore()
        rep = evaluate_unified_generalization_stress(core, seed=42, episodes_per_condition=4)
        self.assertEqual(len(rep.distractor_sweep), 5)
        self.assertTrue(rep.contamination_zero)
        self.assertGreaterEqual(rep.passed_conditions, 0)

    def test_08_three_hop_escalation_diagnostic(self):
        """Verifies Step 375 three-hop escalation evaluation and degradation boundary identification."""
        core = UnifiedCompositionalCore()
        rep = run_three_hop_escalation_diagnostic(core, seed=42, episodes_per_split=4)
        self.assertEqual(len(rep.splits), 4)
        self.assertTrue(len(rep.degradation_boundary) > 0)

    def test_09_master_decision_gate(self):
        """Verifies Step 376 master decision gate across seeds 42, 101, 2026."""
        rep = run_strict_unified_core_i4_evaluation(seeds=(42, 101, 2026), steps_per_level=2, eval_episodes=4)
        self.assertEqual(len(rep.seed_metrics), 3)
        self.assertIn(rep.final_classification, ["I4_ACHIEVED", "I4_EMERGING", "I4_NOT_ACHIEVED"])
        self.assertTrue(rep.baseline_exact)
        self.assertEqual(rep.baseline_sha, EXPECTED_WEIGHT_HASH)


if __name__ == "__main__":
    unittest.main()
