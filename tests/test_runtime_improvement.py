"""
Unit tests for Step 9 Controlled Self-Improvement Layer (chakrview.runtime.improvement).
"""

import pytest
from pathlib import Path
from chakrview.runtime.improvement import (
    ChangeType,
    RiskLevel,
    ProposalStatus,
    ImprovementProposal,
    ImprovementProposalManager,
)


def test_proposal_lifecycle_and_promotion(tmp_path: Path):
    manager = ImprovementProposalManager(storage_dir=tmp_path)

    prop = manager.create_proposal(
        proposal_id="prop_adapter_legal_v1",
        target="chakrview.brain.adapters.legal",
        change_type=ChangeType.ADAPTER,
        parent_version="chakrview-v0.1.0-universal",
        evidence={"val_loss": 3.85, "perplexity": 47.2},
        evaluation_suite="legal_eval_battery_v1",
        expected_change="Reduce legal domain loss by 0.5 without prose regression.",
        risk_level=RiskLevel.MEDIUM,
        artifact_hash="a1b2c3d4e5f67890" * 4,
    )
    assert prop.status == ProposalStatus.DRAFT
    assert not prop.can_promote()

    # Move to evaluating
    prop.submit_for_evaluation()
    assert prop.status == ProposalStatus.EVALUATING
    assert not prop.can_promote()

    # Approve
    prop.approve(approver="lead_engineer")
    assert prop.status == ProposalStatus.APPROVED
    assert prop.approved_by == "lead_engineer"
    assert prop.can_promote()

    # Persistence verification
    loaded = manager.get_proposal("prop_adapter_legal_v1")
    assert loaded is not None
    assert loaded.status == ProposalStatus.APPROVED
    assert (tmp_path / "prop_adapter_legal_v1.json").is_file()


def test_code_change_requires_critical_risk():
    manager = ImprovementProposalManager()
    # Code change type must automatically elevate risk_level to CRITICAL
    prop = manager.create_proposal(
        proposal_id="prop_kernel_opt",
        target="chakrview.brain.attention",
        change_type=ChangeType.CODE,
        parent_version="v0.1",
        evidence={"speedup": "1.4x"},
        evaluation_suite="full_test_suite",
        expected_change="Optimize RoPE vectorized kernel",
        risk_level=RiskLevel.LOW,  # Caller asks for LOW, but manager must enforce CRITICAL
    )
    assert prop.risk_level == RiskLevel.CRITICAL


def test_proposal_rejection_and_quarantine():
    prop = ImprovementProposal(
        proposal_id="prop_bad_weights",
        target="chakrview.brain.weights",
        change_type=ChangeType.WEIGHTS,
        parent_version="v0.1",
        evidence={"loss": 9.99},
        evaluation_suite="regression_suite",
        expected_change="Experimental fine-tuning",
        risk_level=RiskLevel.HIGH,
    )
    prop.submit_for_evaluation()
    prop.quarantine(reason="NaN loss observed during evaluation")
    assert prop.status == ProposalStatus.QUARANTINED
    assert "NaN loss" in (prop.rejection_reason or "")
    assert not prop.can_promote()
