"""
ChakrView Sovereign Semantic Encoder subsystem (Step 14).

Provides trainable neural semantic encoding, InfoNCE contrastive learning,
and seamless integration into the Step 13 hybrid retrieval foundation.
"""

from chakrview.semantic.config import SemanticEncoderConfig
from chakrview.semantic.encoder import SemanticEncoder, TransformerEncoderBlock
from chakrview.semantic.pooling import MeanPooling, MaskedMeanPooling, CLSPooling, PoolingLayer
from chakrview.semantic.projection import SemanticProjection
from chakrview.semantic.loss import InfoNCELoss
from chakrview.semantic.dataset import (
    SemanticPair,
    SemanticDataset,
    SemanticCollator,
    create_reference_fixture_dataset,
)
from chakrview.semantic.training import (
    SemanticTrainingConfig,
    TrainingHistory,
    SemanticTrainer,
)
from chakrview.semantic.evaluation import (
    SemanticRetrievalMetrics,
    evaluate_semantic_retrieval,
)
from chakrview.semantic.serialization import (
    save_semantic_encoder,
    load_semantic_encoder,
)
from chakrview.semantic.provider import (
    NeuralSemanticEmbeddingProvider,
)

__all__ = [
    # Config
    "SemanticEncoderConfig",
    # Model Core
    "SemanticEncoder",
    "TransformerEncoderBlock",
    # Pooling & Projection
    "MeanPooling",
    "MaskedMeanPooling",
    "CLSPooling",
    "PoolingLayer",
    "SemanticProjection",
    # Loss
    "InfoNCELoss",
    # Dataset
    "SemanticPair",
    "SemanticDataset",
    "SemanticCollator",
    "create_reference_fixture_dataset",
    # Training
    "SemanticTrainingConfig",
    "TrainingHistory",
    "SemanticTrainer",
    # Evaluation
    "SemanticRetrievalMetrics",
    "evaluate_semantic_retrieval",
    # Serialization
    "save_semantic_encoder",
    "load_semantic_encoder",
    # Retrieval Adapter
    "NeuralSemanticEmbeddingProvider",
]
