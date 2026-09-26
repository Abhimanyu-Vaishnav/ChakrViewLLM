"""
ChakrView Pre-Training Infrastructure Package.
"""

from chakrview.training.config import (
    TrainingHyperparameters,
    DataConfig,
    CheckpointConfig,
    EvaluationConfig,
    PretrainingConfig,
)
from chakrview.training.seed import set_seed, get_rng_state, set_rng_state
from chakrview.training.sharding import ShardWriter, read_shard_tokens, verify_shard_integrity
from chakrview.training.dataset import StreamingTokenDataset
from chakrview.training.collator import CausalLanguageModelingCollator
from chakrview.training.loss import CausalLoss
from chakrview.training.optimizer import build_optimizer, build_lr_scheduler
from chakrview.training.checkpoint import CheckpointManager
from chakrview.training.metrics import MetricsTracker
from chakrview.training.monitoring import ResourceMonitor
from chakrview.training.evaluator import evaluate
from chakrview.training.trainer import Trainer

__all__ = [
    "TrainingHyperparameters",
    "DataConfig",
    "CheckpointConfig",
    "EvaluationConfig",
    "PretrainingConfig",
    "set_seed",
    "get_rng_state",
    "set_rng_state",
    "ShardWriter",
    "read_shard_tokens",
    "verify_shard_integrity",
    "StreamingTokenDataset",
    "CausalLanguageModelingCollator",
    "CausalLoss",
    "build_optimizer",
    "build_lr_scheduler",
    "CheckpointManager",
    "MetricsTracker",
    "ResourceMonitor",
    "evaluate",
    "Trainer",
]
