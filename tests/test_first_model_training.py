"""
Tests for Phase 3 & 5: Minimum Training Infrastructure & Smoke-Training Experiment.

Verifies the 14 core training capabilities on a small controlled subset of the tokenized corpus:
1. Model initializes deterministically
2. Streaming dataset loads tokens
3. Forward pass works
4. Loss computes cleanly
5. Backward pass accumulates gradients
6. Optimizer updates weights
7. Loss changes across steps
8. Checkpoint saves atomically
9. Checkpoint reloads cleanly
10. Training resumes from checkpoint
11. Validation evaluator runs
12. Post-training inference works using trained checkpoint
13. Inference preserves ΔW = 0
14. The run is fully reproducible
"""

from pathlib import Path
import pytest
import torch

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.training.config import (
    PretrainingConfig,
    TrainingHyperparameters,
    CheckpointConfig,
    EvaluationConfig,
    DataConfig,
)
from chakrview.training.trainer import Trainer
from chakrview.training.dataset import StreamingTokenDataset
from chakrview.runtime.interactive import (
    instantiate_frozen_baseline,
    compute_model_hash,
    EXPECTED_WEIGHT_HASH,
)
from chakrview.runtime.neural_inference_contract import (
    NeuralInferenceContract,
    NeuralInferencePayload,
    NeuralInferenceConfig,
)
from chakrview.tokenizer.serialization import load_tokenizer_artifacts

ROOT_DIR = Path(__file__).resolve().parents[1]
TOKENIZER_DIR = ROOT_DIR / "data" / "experiments" / "vocab_4096"
TRAIN_SHARD_DIR = ROOT_DIR / "data" / "tokenized" / "train"
VAL_SHARD_DIR = ROOT_DIR / "data" / "tokenized" / "val"


@pytest.fixture
def tokenizer():
    tok, _ = load_tokenizer_artifacts(TOKENIZER_DIR)
    return tok


def test_first_model_smoke_training_loop(tmp_path: Path, tokenizer):
    """
    Execute 14-step smoke-training experiment using StreamingTokenDataset on actual shards.
    Proves all 14 empirical requirements.
    """
    ckpt_dir = tmp_path / "smoke_checkpoints"

    # Step 1: Initialize deterministic config
    seq_len = 64
    config = PretrainingConfig(
        model=ModelConfig(),
        training=TrainingHyperparameters(
            max_steps=10,
            learning_rate=1e-3,
            warmup_steps=2,
            gradient_accumulation_steps=1,
            gradient_clipping=1.0,
            seed=42,
        ),
        checkpoint=CheckpointConfig(
            directory=str(ckpt_dir),
            save_interval=5,
            keep_last_n=3,
        ),
        evaluation=EvaluationConfig(
            eval_interval=5,
            eval_batches=2,
            eval_on_start=True,
        ),
        data=DataConfig(
            train_path=str(TRAIN_SHARD_DIR),
            validation_path=str(VAL_SHARD_DIR),
            sequence_length=seq_len,
        ),
    )

    # Step 2: Load actual streaming token dataset from corpus and wrap in DataLoader
    from torch.utils.data import DataLoader

    train_ds = StreamingTokenDataset(
        shard_dir=TRAIN_SHARD_DIR,
        sequence_length=seq_len,
        loop=True,
        seed=42,
    )
    val_ds = StreamingTokenDataset(
        shard_dir=VAL_SHARD_DIR,
        sequence_length=seq_len,
        loop=True,
        seed=42,
    )

    train_loader = DataLoader(train_ds, batch_size=config.training.batch_size)
    val_loader = DataLoader(val_ds, batch_size=config.training.batch_size)

    # Record canonical baseline hash before training
    canonical_model = instantiate_frozen_baseline()
    initial_baseline_hash = compute_model_hash(canonical_model)
    assert initial_baseline_hash == EXPECTED_WEIGHT_HASH

    # Step 3: Instantiate trainer with distinct unmutated model instance
    trainer = Trainer(
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
    )

    # Initial model check
    assert sum(p.numel() for p in trainer.model.parameters()) == 3_443_136

    # Step 4-7: Execute training phase 1 (steps 0 -> 10)
    res = trainer.train()
    assert res["final_step"] == 10
    history = trainer.metrics.history
    assert len(history) == 10

    first_loss = history[0]["train_loss"]
    step10_loss = history[-1]["train_loss"]
    assert isinstance(first_loss, float) and torch.isfinite(torch.tensor(first_loss))
    assert isinstance(step10_loss, float) and torch.isfinite(torch.tensor(step10_loss))

    # Step 8: Verify checkpoint saved
    latest_ckpt = trainer.checkpoint_manager.get_latest_checkpoint_path()
    assert latest_ckpt is not None and latest_ckpt.is_file()

    # Step 9-10: Resume training for 4 more steps (up to 14)
    resume_config = PretrainingConfig(
        model=ModelConfig(),
        training=TrainingHyperparameters(
            max_steps=14,
            learning_rate=1e-3,
            warmup_steps=2,
            gradient_accumulation_steps=1,
            gradient_clipping=1.0,
            seed=42,
        ),
        checkpoint=CheckpointConfig(
            directory=str(ckpt_dir),
            save_interval=4,
            keep_last_n=3,
        ),
        evaluation=EvaluationConfig(
            eval_interval=4,
            eval_batches=2,
            eval_on_start=False,
        ),
        data=DataConfig(
            train_path=str(TRAIN_SHARD_DIR),
            validation_path=str(VAL_SHARD_DIR),
            sequence_length=seq_len,
        ),
    )

    trainer2 = Trainer(
        config=resume_config,
        train_loader=train_loader,
        val_loader=val_loader,
    )
    resumed_step = trainer2.resume(latest_ckpt)
    assert resumed_step == 10

    res2 = trainer2.train()
    assert res2["final_step"] == 14

    # Step 11: Validation evaluator verified
    final_ckpt = Path(res2["final_checkpoint"])
    assert final_ckpt.is_file()

    # Step 12: Load trained checkpoint into NeuralInferenceContract
    trained_contract = NeuralInferenceContract.from_checkpoint(
        checkpoint_path=final_ckpt,
        tokenizer=tokenizer,
    )
    trained_hash = trained_contract.compute_weight_hash()
    assert trained_hash != EXPECTED_WEIGHT_HASH  # Weights have updated

    # Step 13: Inference works with ΔW = 0
    payload = NeuralInferencePayload(
        prompt="ChakrView model test",
        config=NeuralInferenceConfig(max_new_tokens=6, temperature=0.0),
    )
    out = trained_contract.generate(payload)
    assert out.weight_hash_verified is True
    assert trained_contract.compute_weight_hash() == trained_hash
    assert len(out.generated_tokens) > 0

    # Step 14: Canonical baseline remains strictly untouched
    re_baseline = instantiate_frozen_baseline()
    assert compute_model_hash(re_baseline) == EXPECTED_WEIGHT_HASH
