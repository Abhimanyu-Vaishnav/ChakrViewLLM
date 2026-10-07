"""Step 416: Wave 409-416 Master Decision Gate.

Integrates all steps of Wave 409-416 to produce the final decision:
- Step 409: I4 Candidate Manifest verified
- Step 410: Model Registry lifecycle enforced
- Step 411: Reproducibility audit passed
- Step 412: Capability contract verified
- Step 413: Long-horizon retention evaluated
- Step 414: Continual learning evaluated
- Step 415: Sequential benchmark (A->B->C->D) completed
- Step 416: Final wave verdict

Decision Logic
--------------
WAVE_409_416_COMPLETE:
    All manifest, registry, contract, and reproducibility checks pass.
    Continual learning shows no catastrophic forgetting.
    Sequential benchmark acquisition >= 30%.

WAVE_409_416_INFRASTRUCTURE_COMPLETE_LEARNING_INCOMPLETE:
    Infrastructure checks pass but continual learning shows
    forgetting > 40% or sequential benchmark acquisition < 30%.

WAVE_409_416_INFRASTRUCTURE_FAILED:
    Manifest, registry, or reproducibility failures.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, List, Optional, Tuple

from chakrview.runtime.interactive import (
    instantiate_frozen_baseline,
    compute_model_hash,
    EXPECTED_WEIGHT_HASH,
)
from chakrview.cognition.i4_candidate_manifest import (
    run_step409_candidate_manifest,
    Step409ManifestReport,
)
from chakrview.cognition.model_registry import (
    run_step410_model_registry,
    Step410RegistryReport,
)
from chakrview.cognition.reproducibility_audit import (
    run_step411_reproducibility_audit,
    Step411ReproducibilityReport,
)
from chakrview.cognition.capability_contract import (
    run_step412_capability_contract,
    Step412CapabilityContractReport,
)
from chakrview.cognition.long_horizon_retention import (
    run_step413_long_horizon_retention,
    Step413RetentionReport,
)
from chakrview.cognition.continual_learning_evaluation import (
    run_step414_continual_learning,
    Step414ContinualLearningReport,
)
from chakrview.cognition.sequential_capability_benchmark import (
    run_step415_sequential_benchmark,
    Step415SequentialBenchmarkReport,
)


# ---------------------------------------------------------------------------
# Master report
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class Wave409To416MasterReport:
    # Per-step reports
    step409_manifest: Step409ManifestReport
    step410_registry: Step410RegistryReport
    step411_reproducibility: Step411ReproducibilityReport
    step412_contract: Step412CapabilityContractReport
    step413_retention: Step413RetentionReport
    step414_continual: Step414ContinualLearningReport
    step415_sequential: Step415SequentialBenchmarkReport

    # Gate decisions
    infrastructure_passed: bool
    continual_learning_passed: bool
    sequential_benchmark_passed: bool
    catastrophic_forgetting_observed: bool
    baseline_sha_intact: bool

    # Final verdict
    final_classification: str
    decision_rationale: str
    summary: str


# ---------------------------------------------------------------------------
# Gate logic
# ---------------------------------------------------------------------------

def run_wave409_416_master_gate(
    seeds: Tuple[int, ...] = (42, 101),
    steps_per_task: int = 15,
    eval_episodes: int = 4,
    audit_seeds: Tuple[int, ...] = (42,),
    retention_seeds: Tuple[int, ...] = (42,),
    warmup_steps: int = 20,
    extension_steps: int = 20,
) -> Wave409To416MasterReport:
    """
    Executes the full Wave 409-416 master decision gate.

    Parameters are intentionally small for fast CPU execution.
    Full-scale runs should use steps_per_task=40, eval_episodes=8.
    """

    # -----------------------------------------------------------------------
    # Step 409: I4 Candidate Manifest
    # -----------------------------------------------------------------------
    rep409 = run_step409_candidate_manifest()

    # -----------------------------------------------------------------------
    # Step 410: Model Registry
    # -----------------------------------------------------------------------
    rep410 = run_step410_model_registry(manifest=rep409.manifest)

    # -----------------------------------------------------------------------
    # Step 411: Reproducibility Audit
    # -----------------------------------------------------------------------
    rep411 = run_step411_reproducibility_audit(
        seeds=audit_seeds,
        train_steps=warmup_steps,
        eval_episodes=eval_episodes,
    )

    # -----------------------------------------------------------------------
    # Step 412: Capability Contract
    # -----------------------------------------------------------------------
    rep412 = run_step412_capability_contract()

    # -----------------------------------------------------------------------
    # Infrastructure gate
    # -----------------------------------------------------------------------
    infra_passed = (
        rep409.all_gates_passed
        and rep409.baseline_integrity_verified
        and rep409.manifest_hash_verified
        and rep410.promotion_to_frozen_ok
        and rep410.invalid_promotion_blocked
        and rep411.all_audits_passed
        and rep412.verification.all_passed
    )

    # -----------------------------------------------------------------------
    # Step 413: Long-Horizon Retention
    # -----------------------------------------------------------------------
    rep413 = run_step413_long_horizon_retention(
        seeds=retention_seeds,
        warmup_steps=warmup_steps,
        extension_steps=extension_steps,
        checkpoint_interval=max(5, warmup_steps // 4),
        eval_episodes=eval_episodes,
    )

    # -----------------------------------------------------------------------
    # Step 414: Continual Learning
    # -----------------------------------------------------------------------
    rep414 = run_step414_continual_learning(
        seed=seeds[0],
        steps_per_task=steps_per_task,
        eval_episodes=eval_episodes,
    )

    # -----------------------------------------------------------------------
    # Step 415: Sequential Benchmark
    # -----------------------------------------------------------------------
    rep415 = run_step415_sequential_benchmark(
        seeds=seeds,
        steps_per_task=steps_per_task,
        eval_episodes=eval_episodes,
    )

    # -----------------------------------------------------------------------
    # Continual learning gate
    # -----------------------------------------------------------------------
    continual_passed = (
        not rep414.catastrophic_forgetting_detected
        and rep414.language_retention_final >= 0.9400
        and rep413.retention_sustained_count >= len(retention_seeds) // 2
    )
    sequential_passed = (
        rep415.mean_acquisition_b >= 0.30
        and rep415.catastrophic_count == 0
    )
    catastrophic_observed = (
        rep414.catastrophic_forgetting_detected
        or rep415.catastrophic_count > 0
    )

    # -----------------------------------------------------------------------
    # Baseline SHA final check
    # -----------------------------------------------------------------------
    baseline = instantiate_frozen_baseline()
    live_sha = compute_model_hash(baseline)
    sha_ok = (live_sha == EXPECTED_WEIGHT_HASH)

    # -----------------------------------------------------------------------
    # Final verdict
    # -----------------------------------------------------------------------
    if not infra_passed or not sha_ok:
        classification = "WAVE_409_416_INFRASTRUCTURE_FAILED"
        rationale = (
            f"Infrastructure checks failed. "
            f"Manifest={rep409.all_gates_passed}, "
            f"Registry={rep410.promotion_to_frozen_ok}, "
            f"Reproducibility={rep411.all_audits_passed}, "
            f"Contract={rep412.verification.all_passed}, "
            f"BaselineSHA={sha_ok}."
        )
    elif continual_passed and sequential_passed:
        classification = "WAVE_409_416_COMPLETE"
        rationale = (
            f"All infrastructure and continual learning checks passed. "
            f"Sequential benchmark acquisition B={rep415.mean_acquisition_b:.2%}, "
            f"Forgetting delta A={rep415.mean_forgetting_a:.2%}, "
            f"CatastrophicForgetting={catastrophic_observed}."
        )
    else:
        classification = "WAVE_409_416_INFRASTRUCTURE_COMPLETE_LEARNING_INCOMPLETE"
        rationale = (
            f"Infrastructure passed but continual learning incomplete. "
            f"ContinualPassed={continual_passed}, SequentialPassed={sequential_passed}. "
            f"MeanAcqB={rep415.mean_acquisition_b:.2%}, "
            f"MeanForgetA={rep415.mean_forgetting_a:.2%}, "
            f"CatastrophicForgetting={catastrophic_observed}."
        )

    summary = (
        f"Step 416 | Wave 409-416 Master Gate | "
        f"Classification: {classification} | "
        f"Infrastructure={infra_passed} | "
        f"ContinualLearning={continual_passed} | "
        f"Sequential={sequential_passed} | "
        f"BaselineSHA={sha_ok}"
    )

    return Wave409To416MasterReport(
        step409_manifest=rep409,
        step410_registry=rep410,
        step411_reproducibility=rep411,
        step412_contract=rep412,
        step413_retention=rep413,
        step414_continual=rep414,
        step415_sequential=rep415,
        infrastructure_passed=infra_passed,
        continual_learning_passed=continual_passed,
        sequential_benchmark_passed=sequential_passed,
        catastrophic_forgetting_observed=catastrophic_observed,
        baseline_sha_intact=sha_ok,
        final_classification=classification,
        decision_rationale=rationale,
        summary=summary,
    )


if __name__ == "__main__":
    print("Step 416: Wave 409-416 Master Decision Gate...")
    rep = run_wave409_416_master_gate(
        seeds=(42,),
        steps_per_task=10,
        eval_episodes=3,
        audit_seeds=(42,),
        retention_seeds=(42,),
        warmup_steps=15,
        extension_steps=15,
    )
    print(rep.summary)
    print(f"Classification: {rep.final_classification}")
    print(f"Rationale: {rep.decision_rationale}")
    print(f"\nPer-step:")
    print(f"  409 Manifest:        {rep.step409_manifest.summary}")
    print(f"  410 Registry:        {rep.step410_registry.summary}")
    print(f"  411 Reproducibility: {rep.step411_reproducibility.summary}")
    print(f"  412 Contract:        {rep.step412_contract.summary}")
    print(f"  413 Retention:       {rep.step413_retention.summary}")
    print(f"  414 Continual:       {rep.step414_continual.summary}")
    print(f"  415 Sequential:      {rep.step415_sequential.summary}")
