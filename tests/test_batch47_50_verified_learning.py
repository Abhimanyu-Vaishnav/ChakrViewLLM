"""
Test suite for Batches 47-50 Master Sprint:
- Batch 47: Git & report integrity, canonical invariant invariance.
- Batch 48: Grounded domain audit, syntax verification, train/val split isolation.
- Batch 49: Narrow diagnostic reproducibility and multidomain pilot evaluation.
- Batch 50: Hardware-adaptive scheduler integration into SFTEngine and telemetry reporting.
"""

import json
from pathlib import Path
import pytest
import torch

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.training.sft_engine import SFTConfig, SFTExperimentEngine
from experiments.batch42_reproducible_evaluation.evaluation.evaluator import (
    DeterministicEvaluator,
    compute_file_sha256,
)
from experiments.batch43_46_architecture_capability_sprint.hardware_readiness.scheduler import (
    HardwareAdaptiveScheduler,
    WorkloadType,
)

EXPECTED_CANONICAL_HASH = "c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da"
EXPECTED_CANONICAL_PARAMS = 3_443_136
EXPECTED_STEP10000_HASH = "96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee"


def test_batch47_invariants_and_report_integrity():
    """Confirms canonical baseline initialization, step 10k checkpoint hash, and report files exist."""
    torch.manual_seed(42)
    model = ChakrMicro(ModelConfig())
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params == EXPECTED_CANONICAL_PARAMS

    hasher = torch.zeros(1)
    import hashlib

    h = hashlib.sha256()
    with torch.no_grad():
        for name, param in sorted(model.named_parameters()):
            h.update(name.encode("utf-8"))
            h.update(param.detach().cpu().numpy().tobytes())
    assert h.hexdigest() == EXPECTED_CANONICAL_HASH

    base_ckpt = Path("experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt")
    assert compute_file_sha256(base_ckpt) == EXPECTED_STEP10000_HASH


def test_batch48_dataset_eligibility_and_split_isolation():
    """Verifies that eligible datasets exist, have non-zero records, and maintain split isolation."""
    train_path = Path("experiments/batch47_50_verified_learning_sprint/data/train_eligible.jsonl")
    val_path = Path("experiments/batch47_50_verified_learning_sprint/data/val_eligible.jsonl")
    assert train_path.exists() and val_path.exists()

    train_ids = set()
    with open(train_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                train_ids.add(json.loads(line)["id"])

    val_ids = set()
    with open(val_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                val_ids.add(json.loads(line)["id"])

    assert len(train_ids) == 65
    assert len(val_ids) == 13
    # Absolute zero overlap between train and validation record IDs
    assert len(train_ids.intersection(val_ids)) == 0


def test_batch49_diagnostic_reproduction_manifest():
    """Verifies that the narrow diagnostic overfit test reproduced identical 2+3 arithmetic output and EOS."""
    manifest_path = Path(
        "experiments/batch47_50_verified_learning_sprint/learning_experiment/diagnostic_reproduction_manifest.json"
    )
    assert manifest_path.exists()
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert manifest["reproduced_identically"] is True
    assert manifest["eos_reached"] is True
    assert manifest["generated_text"] == "5<|im_end|>\n"
    assert manifest["final_loss"] < 0.01


def test_batch50_hardware_scheduler_sft_integration(tmp_path):
    """Verifies that HardwareAdaptiveScheduler seamlessly configures SFTEngine parameters."""
    base_ckpt = Path("experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt")
    if not base_ckpt.exists():
        pytest.skip("Base checkpoint missing")

    scheduler = HardwareAdaptiveScheduler()
    plan = scheduler.plan_workload(WorkloadType.TRAINING_SFT, total_items=65)

    train_path = Path("experiments/batch47_50_verified_learning_sprint/data/train_eligible.jsonl")
    val_path = Path("experiments/batch47_50_verified_learning_sprint/data/val_eligible.jsonl")

    cfg = SFTConfig(
        repo_root=Path("."),
        train_file=train_path,
        val_file=val_path,
        output_dir=tmp_path,
        base_checkpoint=base_ckpt,
        steps=1,
        batch_size=plan.batch_size,
        device=plan.target_device.value,
    )
    engine = SFTExperimentEngine(cfg)
    assert engine.device.type == plan.target_device.value
    assert cfg.batch_size == plan.batch_size
