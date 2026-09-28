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

from chakrview.training.contract import (
    TrainingExample,
    TrainingDatasetManifest,
    TokenizerFingerprint,
    TrainingRecordEligibilityError,
    validate_learning_record_for_training,
)
from chakrview.training.builder import ChakrOfflineDataset, DatasetBuilder
from chakrview.training.safety import (
    TrainingSafetyChecker,
    TrainingSafetyError,
    NumericalInstabilityError,
    InvariantViolationError,
    CheckpointCorruptionError,
)
from chakrview.training.validation import ValidationEngine, ValidationResult
from chakrview.training.manifest import TrainingRunManifest
from chakrview.training.engine import CPUTrainingEngine, TrainingResult
from chakrview.training.regression import (
    RegressionGate,
    RegressionGateResult,
    PromotionStatus,
)

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
    # Step 22 Additions
    "TrainingExample",
    "TrainingDatasetManifest",
    "TokenizerFingerprint",
    "TrainingRecordEligibilityError",
    "validate_learning_record_for_training",
    "ChakrOfflineDataset",
    "DatasetBuilder",
    "TrainingSafetyChecker",
    "TrainingSafetyError",
    "NumericalInstabilityError",
    "InvariantViolationError",
    "CheckpointCorruptionError",
    "ValidationEngine",
    "ValidationResult",
    "TrainingRunManifest",
    "CPUTrainingEngine",
    "TrainingResult",
    "RegressionGate",
    "RegressionGateResult",
    "PromotionStatus",
]
