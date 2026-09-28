"""
Unit tests for Step 9 Integrity, Verification & Rollback Layer (chakrview.runtime.integrity).
"""

import pytest
from pathlib import Path
import torch
import torch.nn as nn
from chakrview.runtime.integrity import (
    HealthStatus,
    HealthCheckResult,
    HealthReport,
    ArtifactVerifier,
    QuarantineManager,
    RollbackManager,
)


def test_artifact_verifier_file_hashing(tmp_path: Path):
    test_file = tmp_path / "model_weights.bin"
    content = b"CHAKRVIEW_DETERMINISTIC_MODEL_WEIGHTS"
    test_file.write_bytes(content)

    computed_hash = ArtifactVerifier.compute_file_sha256(test_file)
    assert len(computed_hash) == 64
    assert ArtifactVerifier.verify_file_sha256(test_file, computed_hash)
    assert not ArtifactVerifier.verify_file_sha256(test_file, "0" * 64)


def test_model_weights_health_check():
    # Healthy linear model
    model = nn.Linear(10, 10)
    result = ArtifactVerifier.verify_model_weights_health(model)
    assert result.passed is True
    assert result.status == HealthStatus.HEALTHY

    # Inject NaN into weights
    with torch.no_grad():
        model.weight[0, 0] = float("nan")

    result_nan = ArtifactVerifier.verify_model_weights_health(model)
    assert result_nan.passed is False
    assert result_nan.status == HealthStatus.CRITICAL
    assert "NaN" in result_nan.message


def test_quarantine_manager(tmp_path: Path):
    q_dir = tmp_path / "quarantine"
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    
    corrupt_file = data_dir / "corrupted_checkpoint.pt"
    corrupt_file.write_text("corrupted content", encoding="utf-8")

    qm = QuarantineManager(quarantine_dir=q_dir)
    record = qm.quarantine_artifact(
        corrupt_file,
        reason="Checksum failure: expected abc got def",
        artifact_id="corrupt_ckpt_01",
    )

    # Original file is moved out of active directory
    assert not corrupt_file.exists()
    assert Path(record.quarantine_path).exists()
    assert record.reason == "Checksum failure: expected abc got def"
    assert len(qm.list_records()) == 1


def test_rollback_manager(tmp_path: Path):
    state_dir = tmp_path / "state"
    rm = RollbackManager(state_dir=state_dir)

    # 1. Establish initial known-good version
    rm.set_active_version("v1.0.0", "hash_v1_0_0", {"notes": "initial release"})
    rm.mark_as_known_good()

    kg = rm.get_known_good_version()
    assert kg is not None
    assert kg["version_id"] == "v1.0.0"

    # 2. Update to candidate v2.0.0
    rm.set_active_version("v2.0.0", "hash_v2_0_0", {"notes": "candidate update"})
    active = rm.get_active_version()
    assert active is not None
    assert active["version_id"] == "v2.0.0"

    # 3. Simulate failure and execute rollback
    restored = rm.rollback()
    assert restored["version_id"] == "v1.0.0"
    
    current_active = rm.get_active_version()
    assert current_active is not None
    assert current_active["version_id"] == "v1.0.0"
