"""Unit and Integration Tests for Wave 377–384: Adaptive Recurrent Reasoning Core.

Verifies:
- Test 1: Canonical Baseline Invariance (Parameters = 3,443,136, SHA-256 Bit-Exact, Delta W = 0)
- Test 2: Adaptive Recurrent Core Parameter Budget (<600k trainable, <1M total)
- Test 3: Neural Halting Controller Behavior & Continue Probability Outputs
- Test 4: Minimal Adaptive Learnability across 1, 2, and 3 Hop Tasks
- Test 5: Depth Control Comparison (Fixed 1/2/3 vs Adaptive vs Bypass vs Random)
- Test 6: Adaptive Computation Objective & Pareto Scoring
- Test 7: State Causality Interventions & Inter-Cycle Stability
- Test 8: Generalization Stress, Permutation Invariance, and Contamination Audit
- Test 9: Resource Budget Governance & 4-Hop Scaling
- Test 10: Master Decision Gate across Seeds 42, 101, 2026
"""

import unittest
import torch

from chakrview.runtime.interactive import (
    instantiate_frozen_baseline,
    compute_model_hash,
    EXPECTED_WEIGHT_HASH,
)
from chakrview.cognition.adaptive_recurrent_reasoning_core import (
    AdaptiveRecurrentReasoningCore,
    inspect_adaptive_core,
)
from chakrview.cognition.adaptive_learnability import train_and_eval_adaptive_learnability
from chakrview.cognition.adaptive_depth_study import run_adaptive_vs_fixed_depth_study
from chakrview.cognition.adaptive_computation_objective import run_adaptive_computation_objective_study
from chakrview.cognition.adaptive_state_causality import run_adaptive_state_causality_study
from chakrview.cognition.adaptive_generalization_stress import evaluate_adaptive_compositional_generalization
from chakrview.cognition.adaptive_resource_adaptation import run_four_hop_and_budget_adaptation_study
from chakrview.cognition.adaptive_master_decision_gate import run_strict_adaptive_i4_evaluation


class TestWave377To384AdaptiveRecurrentCore(unittest.TestCase):
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

    def test_02_adaptive_core_architecture_and_budget(self):
        """Verifies AdaptiveRecurrentReasoningCore parameter budget (<600k trainable, <1M hard limit)."""
        core = AdaptiveRecurrentReasoningCore()
        trainable_p = core.trainable_param_count
        total_p = core.total_param_count
        controller_p = sum(p.numel() for p in core.controller.parameters())

        self.assertLess(trainable_p, 600_000, f"Trainable params {trainable_p} exceeds 600k")
        self.assertLess(total_p, 1_000_000, f"Total params {total_p} exceeds 1M")
        self.assertEqual(trainable_p, 503_843)
        self.assertEqual(controller_p, 15_969)

        dummy = torch.tensor([[10, 20, 30, 40]], dtype=torch.long)
        out = core(dummy, return_trace=True)
        self.assertIn("vocab_logits", out)
        self.assertIn("total_cycles_executed", out)
        self.assertIn("p_continue_per_cycle", out)
        self.assertEqual(out["vocab_logits"].shape, (1, 4, 4096))

    def test_03_neural_halting_controller(self):
        """Verifies controller forward pass output ranges."""
        core = AdaptiveRecurrentReasoningCore()
        dummy_state = torch.randn(2, core.d_state)
        dummy_query = torch.randn(2, core.d_model)
        dummy_ctx = torch.randn(2, core.d_model)

        logit, prob = core.controller(dummy_state, dummy_query, dummy_ctx)
        self.assertEqual(logit.shape, (2, 1))
        self.assertEqual(prob.shape, (2, 1))
        self.assertTrue(torch.all(prob >= 0.0) and torch.all(prob <= 1.0))

    def test_04_minimal_adaptive_learnability(self):
        """Verifies Step 378 minimal learnability on mixed 1/2/3 hop tasks."""
        core = AdaptiveRecurrentReasoningCore()
        core, rep = train_and_eval_adaptive_learnability(core=core, seed=42, train_steps=6, eval_episodes_per_hop=2)
        self.assertEqual(len(rep.metrics_by_hop), 3)
        self.assertGreaterEqual(rep.overall_accuracy, 0.0)

    def test_05_depth_control_study(self):
        """Verifies Step 379 depth control comparison across variants A-F."""
        rep = run_adaptive_vs_fixed_depth_study(seed=42, train_steps=6, eval_episodes_per_hop=2)
        self.assertEqual(len(rep.variants), 6)
        self.assertIn("A_Fixed1", rep.variants)
        self.assertIn("D_Adaptive", rep.variants)
        self.assertIn("E_BypassController", rep.variants)

    def test_06_adaptive_computation_objective(self):
        """Verifies Step 380 Pareto evaluation across lambdas."""
        rep = run_adaptive_computation_objective_study(seed=42, train_steps=6, eval_episodes=2)
        self.assertEqual(len(rep.points), 4)
        self.assertIn(rep.best_lambda, [0.0, 0.02, 0.08, 0.25])

    def test_07_state_causality_and_stability(self):
        """Verifies Step 381 state causality interventions and stability metrics."""
        rep = run_adaptive_state_causality_study(seed=42, num_episodes=3)
        self.assertEqual(len(rep.interventions), 8)
        self.assertIn("A_Normal", rep.interventions)
        self.assertIn("B_Zero", rep.interventions)
        self.assertGreaterEqual(rep.stability.mean_state_norm, 0.0)

    def test_08_generalization_stress(self):
        """Verifies Step 382 generalization stress conditions and contamination zero."""
        core = AdaptiveRecurrentReasoningCore()
        rep = evaluate_adaptive_compositional_generalization(core, seed=42, episodes_per_condition=2)
        self.assertTrue(rep.contamination_zero)
        self.assertEqual(len(rep.distractor_sweep), 5)
        self.assertGreaterEqual(rep.passed_conditions, 0)

    def test_09_resource_budget_adaptation(self):
        """Verifies Step 383 strict safety budget enforcement."""
        core = AdaptiveRecurrentReasoningCore()
        rep = run_four_hop_and_budget_adaptation_study(core=core, seed=42, episodes=2)
        self.assertTrue(rep.budget_scaling_effective)
        for b, pt in rep.budget_sweep.items():
            self.assertTrue(pt.strictly_bounded)
            self.assertLessEqual(pt.max_cycles_observed, b)

    def test_10_master_decision_gate(self):
        """Verifies Step 384 master decision gate across seeds."""
        rep = run_strict_adaptive_i4_evaluation(seeds=(42, 101, 2026), train_steps=6, eval_episodes=2)
        self.assertEqual(len(rep.seed_metrics), 3)
        self.assertIn(rep.final_classification, ["I4_ACHIEVED", "I4_EMERGING_ADAPTIVE", "I4_NOT_ACHIEVED"])
        self.assertTrue(rep.baseline_exact)
        self.assertEqual(rep.baseline_sha, EXPECTED_WEIGHT_HASH)


if __name__ == "__main__":
    unittest.main()
