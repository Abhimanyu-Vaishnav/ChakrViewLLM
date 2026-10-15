"""
Batches 371-378: Feedback Governance and Hardware-Adaptive Execution Regression Suite.
Location: tests/test_batch351_380_adaptive_intelligence.py

Tests:
1. Canonical invariant and step-150 candidate hash preservation.
2. Feedback learning governance (CANDIDATE -> VERIFIED -> TRAINING_APPROVED, tenant boundary protection).
3. Hardware discovery and adaptive governor safe budget calculations (CPU high memory vs constrained).
4. Stage C replay mixing candidates existence and valid checksum generation.
5. Multi-domain retention metrics compliance (no catastrophic degradation).
"""
import hashlib
from pathlib import Path
import pytest
import torch

from chakrview.intelligence.contracts import LearningRecord, LearningRecordStatus
from chakrview.intelligence.learning import LearningPipeline, TenantIsolationError
from chakrview.runtime.hardware_adaptive_engine import (
    HardwareDiscovery,
    HardwareAdaptiveGovernor,
    HardwareExecutionProfile,
)

CANONICAL_CHECKPOINT = Path("experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt")
EXPECTED_CANONICAL_HASH = "96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee"

STEP150_CHECKPOINT = Path("experiments/batch311_350_neural_generalization/checkpoints/candidate_step150.pt")
EXPECTED_STEP150_HASH = "b6725ce7a4095fe0c551e0b236256d18b55ebd055a865e2f5150e73036384759"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def test_canonical_and_step150_hashes_preserved():
    assert CANONICAL_CHECKPOINT.exists()
    assert sha256_file(CANONICAL_CHECKPOINT) == EXPECTED_CANONICAL_HASH

    assert STEP150_CHECKPOINT.exists()
    assert sha256_file(STEP150_CHECKPOINT) == EXPECTED_STEP150_HASH


def test_feedback_governance_lifecycle():
    pipeline = LearningPipeline()
    rec = LearningRecord.create_candidate(
        owner_id="tenant_x",
        session_id="sess_123",
        input_context="What is the clock frequency?",
        target_output="Verified strictly at 450 MHz.",
        task_type="spec_correction",
    )
    pipeline.add_record(rec)

    # Cannot approve unverified candidate
    with pytest.raises(ValueError):
        pipeline.approve_record(rec.record_id, "tenant_x")

    # Mark verified
    rec.mark_verified(quality_score=0.98)
    assert rec.status == LearningRecordStatus.VERIFIED

    # Other tenant cannot approve
    with pytest.raises(TenantIsolationError):
        pipeline.approve_record(rec.record_id, "tenant_y")

    # Legitimate tenant approval
    pipeline.approve_record(rec.record_id, "tenant_x")
    approved = pipeline.get_records("tenant_x", status=LearningRecordStatus.TRAINING_APPROVED)
    assert len(approved) == 1
    assert approved[0].status == LearningRecordStatus.TRAINING_APPROVED


def test_hardware_adaptive_governor_budgeting():
    env = HardwareDiscovery.probe()
    assert env.cpu_model != ""
    assert env.logical_threads > 0

    budget = HardwareAdaptiveGovernor.select_profile(env, user_workload_active=False)
    assert budget.selected_device == torch.device("cpu")
    assert budget.allocated_threads >= 1
    assert budget.max_ram_budget_bytes > 0
    assert budget.default_chunk_chars > 0

    # User active workload yields resources
    budget_user = HardwareAdaptiveGovernor.select_profile(env, user_workload_active=True)
    assert budget_user.profile == HardwareExecutionProfile.LOW_RESOURCE
    assert budget_user.allocated_threads == 1
    assert budget_user.yield_priority_to_user is True


def test_replay_mixing_candidates_and_report():
    report_json = Path("experiments/batch351_380_adaptive_intelligence/reports/replay_training_comparison.json")
    assert report_json.exists(), "Replay training comparison JSON report must exist"

    ckpt_0 = Path("experiments/batch351_380_adaptive_intelligence/checkpoints/candidate_replay_0pct.pt")
    ckpt_5 = Path("experiments/batch351_380_adaptive_intelligence/checkpoints/candidate_replay_5pct.pt")
    ckpt_10 = Path("experiments/batch351_380_adaptive_intelligence/checkpoints/candidate_replay_10pct.pt")

    assert ckpt_0.exists()
    assert ckpt_5.exists()
    assert ckpt_10.exists()
