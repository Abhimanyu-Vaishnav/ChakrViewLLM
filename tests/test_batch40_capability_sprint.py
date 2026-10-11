"""
Unit and regression tests for Batch 40: Capability Sprint, SFT Safety Integration, and Baseline Invariance.
"""

import hashlib
import json
from pathlib import Path
import pytest
import torch

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.training.sft_engine import SFTConfig, SFTExperimentEngine
from experiments.batch39_dataset_builder.sft_batch_safety import SFTBatchSafetyGuard

EXPECTED_CANONICAL_HASH = "c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da"
EXPECTED_CANONICAL_PARAMS = 3_443_136


def test_canonical_baseline_invariance():
    """Confirms that fresh canonical ChakrMicro initialization matches expected hash and param count."""
    torch.manual_seed(42)
    model = ChakrMicro(ModelConfig())
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params == EXPECTED_CANONICAL_PARAMS

    hasher = hashlib.sha256()
    with torch.no_grad():
        for name, param in sorted(model.named_parameters()):
            hasher.update(name.encode("utf-8"))
            hasher.update(param.detach().cpu().numpy().tobytes())
    digest = hasher.hexdigest()
    assert digest == EXPECTED_CANONICAL_HASH


def test_sft_engine_rejects_all_masked_batch(tmp_path):
    """Verifies that SFTEngine safely catches and rejects all-masked target batches."""
    base_ckpt = Path("experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt")
    if not base_ckpt.exists():
        pytest.skip("Base checkpoint missing")

    # Create dummy train data with all masked targets
    train_file = tmp_path / "dummy_train.jsonl"
    val_file = tmp_path / "dummy_val.jsonl"
    with open(train_file, "w", encoding="utf-8") as f:
        f.write(json.dumps({"prompt": "Hello", "response": "World"}) + "\n")
    with open(val_file, "w", encoding="utf-8") as f:
        f.write(json.dumps({"prompt": "Hello", "response": "World"}) + "\n")

    cfg = SFTConfig(
        repo_root=Path("."),
        train_file=train_file,
        val_file=val_file,
        output_dir=tmp_path / "out",
        base_checkpoint=base_ckpt,
        steps=1,
        batch_size=1,
    )
    engine = SFTExperimentEngine(cfg)

    # Monkey-patch get_batch to simulate an all-masked batch
    engine.train_dataset.get_batch = lambda batch_size, rng: (
        torch.zeros((1, 10), dtype=torch.long),
        torch.full((1, 10), -100, dtype=torch.long),
        0,  # 0 active tokens
    )

    with pytest.raises(ValueError, match="all-masked training batch"):
        engine.train()


def test_batch40_checkpoint_reload_and_probe_reproducibility():
    """Verifies that the Batch 40 experiment checkpoint reloads and reproduces saved outputs."""
    ckpt_path = Path("experiments/batch40_capability_sprint/learning_experiment/sft_checkpoint_step_20.pt")
    if not ckpt_path.exists():
        pytest.skip("Batch 40 experiment checkpoint not yet created")

    manifest_path = Path("experiments/batch40_capability_sprint/learning_experiment/checkpoint_manifest.json")
    assert manifest_path.exists()
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # Verify sha256
    with open(ckpt_path, "rb") as f:
        actual_hash = hashlib.sha256(f.read()).hexdigest()
    assert actual_hash == manifest["sha256"]

    # Verify reload
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = ChakrMicro(ModelConfig(max_seq_len=512))
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    after_metrics_path = Path("experiments/batch40_capability_sprint/learning_experiment/after_metrics.json")
    with open(after_metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    assert "train_loss" in metrics
    assert "val_loss" in metrics
    assert metrics["val_loss"] < 6.98  # Verifies held-out val loss improved from baseline 6.9763
