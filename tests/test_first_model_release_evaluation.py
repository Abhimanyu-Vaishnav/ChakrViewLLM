"""
Unit and integration tests for Phase 6 & 7: Release Evaluator and Release Packager.
"""

from pathlib import Path
import pytest
import torch
from torch.utils.data import DataLoader

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.runtime.interactive import (
    instantiate_frozen_baseline,
    compute_model_hash,
    EXPECTED_WEIGHT_HASH,
)
from chakrview.tokenizer.serialization import load_tokenizer_artifacts
from chakrview.training.config import (
    PretrainingConfig,
    TrainingHyperparameters,
    CheckpointConfig,
    EvaluationConfig,
    DataConfig,
)
from chakrview.training.trainer import Trainer
from chakrview.training.dataset import StreamingTokenDataset
from chakrview.training.release_evaluator import ReleaseEvaluator
from chakrview.training.release_packager import ReleasePackager

ROOT_DIR = Path(__file__).resolve().parents[1]
TOKENIZER_DIR = ROOT_DIR / "data" / "experiments" / "vocab_4096"
TRAIN_SHARD_DIR = ROOT_DIR / "data" / "tokenized" / "train"
VAL_SHARD_DIR = ROOT_DIR / "data" / "tokenized" / "val"


@pytest.fixture
def tokenizer():
    tok, _ = load_tokenizer_artifacts(TOKENIZER_DIR)
    return tok


def test_release_evaluator_and_packager_flow(tmp_path: Path, tokenizer):
    """
    Train a smoke checkpoint, evaluate via ReleaseEvaluator,
    and package cleanly into a release directory bundle with checksums.
    """
    ckpt_dir = tmp_path / "train_ckpts"
    release_dir = tmp_path / "releases"
    seq_len = 64

    # 1. Train 4 steps
    config = PretrainingConfig(
        model=ModelConfig(),
        training=TrainingHyperparameters(
            max_steps=4,
            learning_rate=1e-3,
            warmup_steps=1,
            seed=42,
        ),
        checkpoint=CheckpointConfig(
            directory=str(ckpt_dir),
            save_interval=4,
        ),
        data=DataConfig(
            train_path=str(TRAIN_SHARD_DIR),
            validation_path=str(VAL_SHARD_DIR),
            sequence_length=seq_len,
        ),
    )
    train_ds = StreamingTokenDataset(shard_dir=TRAIN_SHARD_DIR, sequence_length=seq_len, loop=True, seed=42)
    val_ds = StreamingTokenDataset(shard_dir=VAL_SHARD_DIR, sequence_length=seq_len, loop=True, seed=42)
    train_loader = DataLoader(train_ds, batch_size=config.training.batch_size)
    val_loader = DataLoader(val_ds, batch_size=config.training.batch_size)

    trainer = Trainer(config=config, train_loader=train_loader, val_loader=val_loader)
    res = trainer.train()
    final_ckpt = Path(res["final_checkpoint"])
    assert final_ckpt.is_file()

    # 2. Evaluate using ReleaseEvaluator
    evaluator = ReleaseEvaluator(
        tokenizer=tokenizer,
        val_shard_dir=VAL_SHARD_DIR,
        device="cpu",
    )
    metrics = evaluator.evaluate_checkpoint(
        checkpoint_path=final_ckpt,
        model_name="ChakrMicro-v0.1-Smoke",
        max_val_batches=2,
        sequence_length=seq_len,
    )

    assert metrics.parameter_count == 3_443_136
    assert metrics.weights_finite is True
    assert metrics.inference_delta_w_zero is True
    assert metrics.val_loss > 0.0
    assert metrics.val_perplexity > 0.0
    assert metrics.is_canonical_baseline is False
    assert metrics.proven_status["open_domain_fluency"] == "UNPROVEN"

    # 3. Package release bundle
    packager = ReleasePackager(output_dir=release_dir)
    bundle_dir = packager.package_release(
        checkpoint_path=final_ckpt,
        tokenizer_dir=TOKENIZER_DIR,
        eval_metrics=metrics,
        model_config_dict=ModelConfig().__dict__,
        training_config_dict=config.training.__dict__,
        release_tag="chakrmicro_v0.1_smoke",
    )

    assert bundle_dir.is_dir()
    assert (bundle_dir / "checkpoint" / "model.pt").is_file()
    assert (bundle_dir / "checkpoint" / "latest_checkpoint.json").is_file()
    assert (bundle_dir / "tokenizer" / "vocab.json").is_file()
    assert (bundle_dir / "config" / "model_config.json").is_file()
    assert (bundle_dir / "evaluation" / "eval_summary.json").is_file()
    assert (bundle_dir / "MODEL_CARD.md").is_file()
    assert (bundle_dir / "checksums.sha256").is_file()

    # Verify baseline untouched
    baseline = instantiate_frozen_baseline()
    assert compute_model_hash(baseline) == EXPECTED_WEIGHT_HASH
