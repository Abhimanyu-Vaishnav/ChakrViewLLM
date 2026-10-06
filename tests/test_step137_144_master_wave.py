"""
ChakrView Step 137-144: Master Development Wave Unit & Integration Tests.

Validates:
- Step 137: Sovereign Domain Module Architecture (Contract & Registry Lifecycle)
- Step 138: Domain Curriculum Engine (Staging, Manifests, Held-Out Isolation)
- Step 139: Multi-Domain Learning & Anti-Forgetting (Retention, Degradation Detection & Rollback)
- Step 140: Governed Domain Adaptation (Bounded Residual Bottleneck Adapters without Base Mutation)
- Step 141: Continuous Intelligence Probing (Multi-Dimensional Capabilities, Held-out Generalization)
- Step 142: Critical Thinking & Epistemic Reasoning (Uncertainty, Abstention & Revision)
- Step 143: Governed Experience Learning (RIL Loops, Failure Signals & Policy Recording)
- Step 144: Universal Neural Core Immutability (3,443,136 parameters, SHA-256 c5571c... bit-exact ΔW ≡ 0)
"""

import time
from pathlib import Path
import pytest
import torch

from chakrview.brain.model import ChakrMicro
from chakrview.runtime.interactive import (
    instantiate_frozen_baseline,
    compute_model_hash,
    EXPECTED_WEIGHT_HASH,
)
from chakrview.cognition.domain_module import (
    DomainModuleContract,
    DomainModuleStatus,
    DomainCapabilityProfile,
    DomainResourceRequirements,
    GovernedDomainRegistry,
)
from chakrview.cognition.domain_curriculum import (
    CurriculumStage,
    DomainCurriculumSample,
    DomainCurriculumEngine,
)
from chakrview.cognition.anti_forgetting import (
    MultiDomainAntiForgettingCoordinator,
)
from chakrview.cognition.domain_adapter import (
    BoundedDomainAdapter,
    GovernedAdaptedModel,
)
from chakrview.cognition.intelligence_evaluator import (
    IntelligenceCapabilityDimension,
    IntelligenceProbeItem,
    ContinuousIntelligenceEvaluator,
)
from chakrview.cognition.critical_thinking import (
    EpistemicState,
    EvidenceStatement,
    CriticalThinkingEngine,
)
from chakrview.cognition.experience_learning import (
    ExperienceSignalOutcome,
    GovernedExperienceLearningEngine,
)


def test_step137_domain_module_contract_and_lifecycle(tmp_path):
    db_path = tmp_path / "domain_registry.db"
    registry = GovernedDomainRegistry(db_path)

    contract = DomainModuleContract(
        domain_id="domain_science_v1",
        version="1.0.0",
        description="Scientific reasoning domain",
        domain_vocabulary=["hypothesis", "empirical", "falsify", "p_value"],
        knowledge_sources=["docs/science.md"],
        curriculum_identity="curr_sci_01",
        evaluation_suite="eval_sci_probes",
        capability_profile=DomainCapabilityProfile(primary_tasks=["hypothesis_test"]),
        resource_requirements=DomainResourceRequirements(cpu_cores=1, memory_mb=256),
    )

    # Register
    ok = registry.register_domain(contract)
    assert ok is True
    dom = registry.get_domain("domain_science_v1")
    assert dom is not None
    assert dom.status == DomainModuleStatus.REGISTERED
    assert dom.validation_status is False

    # Validate
    assert registry.validate_domain("domain_science_v1", is_valid=True) is True
    dom_val = registry.get_domain("domain_science_v1")
    assert dom_val.status == DomainModuleStatus.VALIDATED
    assert dom_val.validation_status is True

    # Activate
    assert registry.activate_domain("domain_science_v1") is True
    dom_act = registry.get_domain("domain_science_v1")
    assert dom_act.status == DomainModuleStatus.ACTIVATED

    # Deprecate
    assert registry.deprecate_domain("domain_science_v1") is True
    dom_dep = registry.get_domain("domain_science_v1")
    assert dom_dep.status == DomainModuleStatus.DEPRECATED


def test_step138_domain_curriculum_engine():
    engine = DomainCurriculumEngine("curr_math_01", "domain_math_v1", seed=1337)
    engine.add_sample(DomainCurriculumSample("s1", "domain_math_v1", CurriculumStage.FOUNDATION, "1 + 1", "2"))
    engine.add_sample(DomainCurriculumSample("s2", "domain_math_v1", CurriculumStage.DOMAIN_INTRO, "2 * 3", "6"))
    engine.add_sample(DomainCurriculumSample("s3", "domain_math_v1", CurriculumStage.DOMAIN_REASONING, "x + 2 = 5", "x = 3"))
    engine.add_sample(DomainCurriculumSample("s_eval", "domain_math_v1", CurriculumStage.EVALUATION, "10 / 2", "5"), is_held_out=True)

    manifest = engine.create_execution_manifest(
        candidate_checkpoint_id="chkpt_math_cand_1",
        baseline_hash=EXPECTED_WEIGHT_HASH,
        tokenizer_checksum="tok_4096_hash",
        metrics={"accuracy": 0.95},
    )

    assert manifest.curriculum_id == "curr_math_01"
    assert manifest.total_samples == 4
    assert len(manifest.dataset_checksum) == 64
    assert manifest.baseline_hash == EXPECTED_WEIGHT_HASH


def test_step139_anti_forgetting_retention_and_rollback():
    coord = MultiDomainAntiForgettingCoordinator(max_allowed_general_regression=0.04)

    # Clean retention (negligible regression)
    rep_ok = coord.evaluate_retention(
        domain_accuracies={"coding": 0.90, "math": 0.88},
        initial_general_accuracy=0.92,
        current_general_accuracy=0.915,
    )
    assert not rep_ok.catastrophic_forgetting_detected
    assert rep_ok.recommended_action == "PROCEED_CANDIDATE"

    # Mild regression -> ADAPT_SAMPLING
    rep_warn = coord.evaluate_retention(
        domain_accuracies={"coding": 0.90, "math": 0.88},
        initial_general_accuracy=0.92,
        current_general_accuracy=0.895,  # -0.025 > 0.02 (0.5 * 0.04)
    )
    assert not rep_warn.catastrophic_forgetting_detected
    assert rep_warn.recommended_action == "ADAPT_SAMPLING"

    # Severe regression -> Rollback
    rep_rollback = coord.evaluate_retention(
        domain_accuracies={"coding": 0.98},
        initial_general_accuracy=0.92,
        current_general_accuracy=0.85,  # -0.07 > 0.04 threshold
    )
    assert rep_rollback.catastrophic_forgetting_detected
    assert rep_rollback.recommended_action == "ROLLBACK_CHECKPOINT"


def test_step140_governed_domain_adapter_isolation():
    base = instantiate_frozen_baseline()
    adapted = GovernedAdaptedModel(base)

    adapter_code = BoundedDomainAdapter(d_model=192, bottleneck_dim=16)
    adapter_math = BoundedDomainAdapter(d_model=192, bottleneck_dim=16)

    adapted.mount_domain_adapter("coding", adapter_code)
    adapted.mount_domain_adapter("math", adapter_math)

    # Forward pass with coding domain active
    adapted.activate_domain("coding")
    x = torch.tensor([[1, 2, 3]], dtype=torch.long)
    logits = adapted(x)
    assert logits.shape == (1, 3, 4096)

    # Unmount math adapter
    adapted.unmount_domain_adapter("math")
    assert "math" not in adapted.mounted_adapters

    # Baseline remains 100% frozen & bit-exact
    assert adapted.verify_baseline_intact() is True


def test_step141_continuous_intelligence_evaluator():
    evaluator = ContinuousIntelligenceEvaluator()
    evaluator.add_probe(IntelligenceProbeItem("p1", IntelligenceCapabilityDimension.REASONING, "Is 5 > 3?", "yes", is_held_out=False))
    evaluator.add_probe(IntelligenceProbeItem("p2", IntelligenceCapabilityDimension.CRITICAL_THINKING, "What is the password?", "not enough evidence", is_held_out=True, allowed_abstention=True))

    def predictor(prompt: str):
        if "password" in prompt:
            return "I do not have enough evidence.", 0.95
        return "yes", 0.99

    summary = evaluator.evaluate_system(predictor)
    assert summary.evaluation_passed is True
    assert summary.overall_generalization_score == 1.0


def test_step142_critical_thinking_epistemic_reasoning():
    # Complete absence of evidence -> Honest Abstention
    res_unknown = CriticalThinkingEngine.evaluate_claim("Quantum speedup in sorting", [], [])
    assert res_unknown.state == EpistemicState.UNKNOWN
    assert res_unknown.abstention_triggered is True

    # Conflicting evidence -> CONFLICTING
    supp = [EvidenceStatement("e1", "Speedup observed in benchmark", 0.8, time.time(), polarity=True)]
    ref = [EvidenceStatement("e2", "Replication failed across trials", 0.85, time.time(), polarity=False)]
    res_contra = CriticalThinkingEngine.evaluate_claim("Quantum speedup in sorting", supp, ref)
    assert res_contra.state == EpistemicState.CONFLICTING

    # Revision upon new decisive evidence
    decisive = [EvidenceStatement("e3", "Hardware flaw identified in trial 1", 0.99, time.time(), polarity=False)]
    revised = CriticalThinkingEngine.revise_conclusion(res_contra, decisive)
    assert revised.state == EpistemicState.KNOWN
    assert "Revised" in revised.justification


def test_step143_governed_experience_learning(tmp_path):
    db_path = tmp_path / "exp_learn.db"
    engine = GovernedExperienceLearningEngine(db_path)

    ep = engine.process_experience(
        task_id="t-99",
        domain_id="domain_sec_v1",
        outcome=ExperienceSignalOutcome.REVIEWER_REJECTED,
        raw_observations=["Weak hash algorithm selected"],
        error_context="MD5 prohibited in security domain",
    )
    assert ep.applied_to_policy is True
    lessons = engine.query_lessons_for_domain("domain_sec_v1")
    assert len(lessons) == 1
    assert "validate syntax and boundary conditions" in lessons[0]


def test_step144_canonical_baseline_immutability():
    model = instantiate_frozen_baseline()
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params == 3443136, f"Parameter count mismatch: {total_params} != 3443136"

    model_hash = compute_model_hash(model)
    assert model_hash == EXPECTED_WEIGHT_HASH
    assert model_hash == "c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da"
