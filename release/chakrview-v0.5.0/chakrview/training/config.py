"""
Authoritative Pre-Training Configuration System for ChakrView.

Defines all configuration dataclasses for pre-training:
- ModelConfig (frozen neural architecture)
- TrainingConfig (optimization & schedule)
- DataConfig (paths, sequence length, sharding)
- CheckpointConfig (directory, intervals, retention)
- EvaluationConfig (validation frequency and size)
- PretrainingConfig (unified top-level configuration)
"""

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Dict, Any

from chakrview.brain.config import ModelConfig


@dataclass
class TrainingHyperparameters:
    """
    Hyperparameters governing the optimization loop.
    
    Attributes:
        seed: Random seed for deterministic execution.
        batch_size: Micro-batch size per optimization step.
        gradient_accumulation_steps: Number of forward/backward steps before optimizer step.
        learning_rate: Peak learning rate for cosine schedule.
        min_learning_rate: Minimum learning rate floor after decay.
        weight_decay: Decoupled weight decay for 2D matrix parameters.
        optimizer: Optimizer name ("adamw").
        adam_beta1: First moment decay factor.
        adam_beta2: Second moment decay factor.
        adam_eps: Epsilon for numerical stability.
        max_steps: Total number of training steps.
        warmup_steps: Number of linear warmup steps.
        gradient_clipping: Maximum gradient L2 norm for clipping (0.0 to disable).
        lr_decay_style: Learning rate decay schedule ("cosine" or "constant").
    """
    seed: int = 42
    batch_size: int = 2
    gradient_accumulation_steps: int = 1
    learning_rate: float = 5e-4
    min_learning_rate: float = 5e-5
    weight_decay: float = 0.01
    optimizer: str = "adamw"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-8
    max_steps: int = 100
    warmup_steps: int = 10
    gradient_clipping: float = 1.0
    lr_decay_style: str = "cosine"

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.gradient_accumulation_steps <= 0:
            raise ValueError(f"gradient_accumulation_steps must be positive, got {self.gradient_accumulation_steps}")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")
        if self.max_steps <= 0:
            raise ValueError(f"max_steps must be positive, got {self.max_steps}")
        if self.warmup_steps < 0:
            raise ValueError(f"warmup_steps cannot be negative, got {self.warmup_steps}")
        if self.gradient_clipping < 0.0:
            raise ValueError(f"gradient_clipping cannot be negative, got {self.gradient_clipping}")


@dataclass
class DataConfig:
    """
    Data paths, sequence lengths, and streaming options.
    
    Attributes:
        train_path: Directory or path to tokenized training shards.
        validation_path: Directory or path to tokenized validation shards.
        tokenizer_path: Directory containing vocabulary and merges.
        sequence_length: Target context window for pre-training (<= 512).
        pad_token_id: Special token ID for padding sequences.
        eos_token_id: Special token ID for end-of-sequence.
        bos_token_id: Special token ID for beginning-of-sequence.
        workers: Number of background data loader worker processes.
        shuffle_shards: Whether to shuffle shards per epoch.
    """
    train_path: str = "data/tokenized/train"
    validation_path: str = "data/tokenized/validation"
    tokenizer_path: str = "data/experiments/vocab_4096"
    sequence_length: int = 512
    pad_token_id: int = 2
    eos_token_id: int = 1
    bos_token_id: int = 0
    workers: int = 0
    shuffle_shards: bool = True

    def __post_init__(self) -> None:
        if self.sequence_length <= 0:
            raise ValueError(f"sequence_length must be positive, got {self.sequence_length}")
        if self.sequence_length > 512:
            raise ValueError(f"sequence_length ({self.sequence_length}) exceeds max context 512")


@dataclass
class CheckpointConfig:
    """
    Checkpoint directory, frequency, and retention policies.
    
    Attributes:
        directory: Directory where checkpoints are saved.
        save_interval: Steps between checkpoint saves.
        keep_last_n: Maximum number of recent checkpoints to retain.
        save_optimizer: Whether to serialize optimizer states.
    """
    directory: str = "checkpoints"
    save_interval: int = 25
    keep_last_n: int = 3
    save_optimizer: bool = True

    def __post_init__(self) -> None:
        if self.save_interval <= 0:
            raise ValueError(f"save_interval must be positive, got {self.save_interval}")
        if self.keep_last_n <= 0:
            raise ValueError(f"keep_last_n must be positive, got {self.keep_last_n}")


@dataclass
class EvaluationConfig:
    """
    Validation evaluation frequency and batch parameters.
    
    Attributes:
        eval_interval: Steps between validation passes.
        eval_batches: Number of batches to evaluate per validation pass.
        eval_on_start: Whether to evaluate before training step 1.
    """
    eval_interval: int = 25
    eval_batches: int = 5
    eval_on_start: bool = True

    def __post_init__(self) -> None:
        if self.eval_interval <= 0:
            raise ValueError(f"eval_interval must be positive, got {self.eval_interval}")
        if self.eval_batches <= 0:
            raise ValueError(f"eval_batches must be positive, got {self.eval_batches}")


@dataclass
class PretrainingConfig:
    """
    Unified authoritative pre-training configuration.
    """
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingHyperparameters = field(default_factory=TrainingHyperparameters)
    data: DataConfig = field(default_factory=DataConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to JSON-serializable dictionary."""
        return {
            "model": asdict(self.model),
            "training": asdict(self.training),
            "data": asdict(self.data),
            "checkpoint": asdict(self.checkpoint),
            "evaluation": asdict(self.evaluation),
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize configuration to JSON formatted string."""
        return json.dumps(self.to_dict(), indent=indent)

    def save(self, path: Path | str) -> None:
        """Write configuration to disk as JSON."""
        target_path = Path(path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with open(target_path, "w", encoding="utf-8") as f:
            f.write(self.to_json())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PretrainingConfig":
        """Reconstruct configuration from dictionary."""
        model_cfg = ModelConfig(**data.get("model", {}))
        training_cfg = TrainingHyperparameters(**data.get("training", {}))
        data_cfg = DataConfig(**data.get("data", {}))
        checkpoint_cfg = CheckpointConfig(**data.get("checkpoint", {}))
        eval_cfg = EvaluationConfig(**data.get("evaluation", {}))
        return cls(
            model=model_cfg,
            training=training_cfg,
            data=data_cfg,
            checkpoint=checkpoint_cfg,
            evaluation=eval_cfg,
        )

    @classmethod
    def load(cls, path: Path | str) -> "PretrainingConfig":
        """Load configuration from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)
