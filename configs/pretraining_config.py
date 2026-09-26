"""
Authoritative entrypoint configuration script for ChakrView pre-training.

Exports default and profile-specific pre-training configurations.
"""

from pathlib import Path
from chakrview.training.config import (
    PretrainingConfig,
    TrainingHyperparameters,
    DataConfig,
    CheckpointConfig,
    EvaluationConfig,
)
from chakrview.brain.config import ModelConfig


def get_default_pretraining_config() -> PretrainingConfig:
    """Return default validated PretrainingConfig for ChakrMicro v0.1 on CPU."""
    return PretrainingConfig(
        model=ModelConfig(),  # Frozen 3.44M architecture
        training=TrainingHyperparameters(
            seed=42,
            batch_size=2,
            gradient_accumulation_steps=1,
            learning_rate=5e-4,
            min_learning_rate=5e-5,
            weight_decay=0.01,
            max_steps=100,
            warmup_steps=10,
            gradient_clipping=1.0,
        ),
        data=DataConfig(
            train_path="data/tokenized/train",
            validation_path="data/tokenized/validation",
            sequence_length=512,
        ),
        checkpoint=CheckpointConfig(
            directory="checkpoints",
            save_interval=25,
            keep_last_n=3,
        ),
        evaluation=EvaluationConfig(
            eval_interval=25,
            eval_batches=5,
        ),
    )


def get_micro_test_config() -> PretrainingConfig:
    """Return lightweight micro-test configuration for smoke testing."""
    cfg = get_default_pretraining_config()
    cfg.training.max_steps = 20
    cfg.training.warmup_steps = 4
    cfg.training.batch_size = 2
    cfg.data.sequence_length = 64
    cfg.checkpoint.save_interval = 10
    cfg.evaluation.eval_interval = 10
    cfg.evaluation.eval_batches = 2
    return cfg


if __name__ == "__main__":
    default_cfg = get_default_pretraining_config()
    out_dir = Path("configs")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "default_pretraining_config.json"
    default_cfg.save(out_path)
    print(f"Authoritative default pretraining configuration written to: {out_path}")
