"""Unit and Integration Tests for Wave 385–392: Need-Based Adaptive Reasoning.

Verifies:
- Test 1: Canonical Baseline Invariance (Parameters = 3,443,136, SHA-256 Bit-Exact, Delta W = 0)
- Test 2: Sufficiency Adaptive Core Parameter Budget (<550k trainable, <50k added controller)
- Test 3: Answer Sufficiency Controller Outputs (sufficiency score & continue prob bounds)
- Test 4: Lazy-Halt Ablation Study execution
- Test 5: Counterfactual Continuation Evaluation (Path A vs Path B)
- Test 6: Difficulty Generalization (1h through 4h)
- Test 7: Strict Compositional Generalization Suite & Contamination Audit
- Test 8: Multi-Seed Stability & Seed 2026 Forensic Diagnosis
- Test 9: Resource-Aware Budget Governance & Bounded Computation
- Test 10: Master Decision Gate across Seeds 42, 101, 2026
"""

import unittest
import torch

from chakrview.runtime.interactive import (
    instantiate_frozen_baseline,
    compute_model_hash,
    EXPECTED_WEIGHT_HASH,
)
from chakrview.cognition.sufficiency_adaptive_reasoning import (
    SufficiencyAdaptiveReasoningCore,
    inspect_sufficiency_core,
)
from chakrview.cognition.lazy_halt_ablation import (
    run_lazy_halt_ablation_study,
    train_and_eval_sufficiency_core,
)
from chakrview.cognition.counterfactual_continuation import run_counterfactual_continuation_test
from chakrview.cognition.difficulty_generalization import run_difficulty_generalization_study
from chakrview.cognition.sufficiency_generalization_suite import evaluate_sufficiency_strict_generalization
from chakrview.cognition.multi_seed_stability_study import run_multi_seed_controller_stability_study
from chakrview.cognition.resource_aware_reasoning import run_resource_aware_reasoning_study
from chakrview.cognition.sufficiency_master_decision_gate import run_strict_sufficiency_i4_evaluation


class TestWave385To392NeedBasedAdaptiveReasoning(unittest.TestCase):
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

    def test_02_sufficiency_core_architecture_and_budget(self):
        """Verifies SufficiencyAdaptiveReasoningCore parameter budget (<550k trainable, <50k added)."""
        spec = inspect_sufficiency_core()
        self.assertLess(spec["trainable_parameters"], 550_000)
        self.assertLess(spec["controller_parameters"], 50_000)
        self.assertEqual(spec["trainable_parameters"], 504_041)
        self.assertEqual(spec["controller_parameters"], 16_167)

        core = SufficiencyAdaptiveReasoningCore()
        dummy = torch.tensor([[10, 20, 30, 40]], dtype=torch.long)
        out = core(dummy, return_trace=True)
        self.assertIn("vocab_logits", out)
        self.assertIn("sufficiency_scores_per_cycle", out)
        self.assertIn("p_continue_per_cycle", out)

    def test_03_answer_sufficiency_controller(self):
        """Verifies AnswerSufficiencyController output tensors and ranges."""
        core = SufficiencyAdaptiveReasoningCore()
        dummy_state = torch.randn(2, core.d_state)
        dummy_query = torch.randn(2, core.d_model)
        dummy_ctx = torch.randn(2, core.d_model)
        dummy_feat = torch.randn(2, 3)

        s_logit, s_score, c_prob = core.sufficiency_controller(dummy_state, dummy_query, dummy_ctx, dummy_feat)
        self.assertEqual(s_score.shape, (2, 1))
        self.assertEqual(c_prob.shape, (2, 1))
        self.assertTrue(torch.all(s_score >= 0.0) and torch.all(s_score <= 1.0))
        self.assertTrue(torch.all(c_prob >= 0.0) and torch.all(c_prob <= 1.0))

    def test_04_lazy_halt_ablation(self):
        """Verifies Step 386 lazy-halt ablation execution."""
        rep = run_lazy_halt_ablation_study(seed=42)
        self.assertEqual(len(rep.variants), 5)
        self.assertIn("A_Wave384Baseline", rep.variants)
        self.assertIn("B_SufficiencyDefault", rep.variants)

    def test_05_counterfactual_continuation(self):
        """Verifies Step 387 counterfactual continuation evaluation."""
        core = SufficiencyAdaptiveReasoningCore()
        rep = run_counterfactual_continuation_test(core=core, seed=42, episodes_per_hop=2)
        self.assertGreater(len(rep.pairs), 0)
        self.assertGreaterEqual(rep.alignment_accuracy, 0.0)

    def test_06_difficulty_generalization(self):
        """Verifies Step 388 evaluation across 1-hop to 4-hop tasks."""
        core = SufficiencyAdaptiveReasoningCore()
        rep = run_difficulty_generalization_study(core=core, seed=42, eval_episodes_per_level=2)
        self.assertEqual(len(rep.levels), 4)
        self.assertIn(1, rep.levels)
        self.assertIn(4, rep.levels)

    def test_07_strict_generalization_and_contamination(self):
        """Verifies Step 389 strict generalization and 0 data contamination."""
        core = SufficiencyAdaptiveReasoningCore()
        rep = evaluate_sufficiency_strict_generalization(core, seed=42, episodes_per_condition=2)
        self.assertTrue(rep.contamination_zero)
        self.assertEqual(len(rep.distractor_sweep), 5)

    def test_08_multi_seed_stability(self):
        """Verifies Step 390 multi-seed investigation and diagnosis."""
        rep = run_multi_seed_controller_stability_study(seeds=(42, 101, 2026), train_steps=5, eval_episodes=2)
        self.assertEqual(len(rep.seed_metrics), 3)
        self.assertIn(2026, rep.seed_metrics)
        self.assertTrue(len(rep.seed2026_failure_diagnosis) > 0)

    def test_09_resource_aware_reasoning(self):
        """Verifies Step 391 budget governance and bounded consumption."""
        rep = run_resource_aware_reasoning_study(core=None, seed=42, episodes=2)
        self.assertTrue(rep.all_strictly_bounded)
        self.assertTrue(rep.non_greedy_consumption_proven)

    def test_10_master_decision_gate(self):
        """Verifies Step 392 master decision gate execution and baseline exactness."""
        rep = run_strict_sufficiency_i4_evaluation(seeds=(42, 101, 2026), train_steps=5, eval_episodes=2)
        self.assertEqual(len(rep.seed_metrics), 3)
        self.assertIn(rep.final_classification, ["I4_ACHIEVED", "I4_EMERGING_NEED_BASED", "I4_NOT_ACHIEVED"])
        self.assertTrue(rep.baseline_exact)
        self.assertEqual(rep.baseline_sha, EXPECTED_WEIGHT_HASH)


if __name__ == "__main__":
    unittest.main()
