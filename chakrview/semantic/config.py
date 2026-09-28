"""
Configuration for the Sovereign Semantic Encoder (Step 14).

Defines architectural hyperparameters, dimensional bounds, pooling modes,
and contrastive training parameters.
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional


@dataclass
class SemanticEncoderConfig:
    """
    Configuration for the trainable neural semantic encoder.
    
    Attributes:
        vocab_size: Tokenizer vocabulary size (frozen at 4096 to match ChakrView).
        d_model: Internal transformer representation dimension (default: 128).
        n_layers: Number of stacked bidirectional transformer layers (default: 2).
        n_heads: Number of attention query/key/value heads (default: 4, d_head=32).
        d_ff: Feed-forward intermediate expansion dimension (default: 256).
        embedding_dim: Output dense semantic vector dimension after projection (default: 128).
        max_seq_len: Maximum input sequence length supported (default: 256).
        dropout: Dropout probability for attention and FFN layers (default: 0.1).
        pooling_mode: Token aggregation strategy ('mean', 'masked_mean', 'cls').
        normalize_embeddings: Whether to apply L2 normalization to output vectors (default: True).
        pad_token_id: Special token ID for padding (strictly 2 in ChakrView).
        temperature: Default contrastive InfoNCE temperature scaling factor (default: 0.05).
        layer_norm_eps: Epsilon for LayerNorm stability (default: 1e-5).
    """
    vocab_size: int = 4096
    d_model: int = 128
    n_layers: int = 2
    n_heads: int = 4
    d_ff: int = 256
    embedding_dim: int = 128
    max_seq_len: int = 256
    dropout: float = 0.1
    pooling_mode: str = "masked_mean"
    normalize_embeddings: bool = True
    pad_token_id: int = 2
    temperature: float = 0.05
    layer_norm_eps: float = 1e-5

    def __post_init__(self) -> None:
        if self.vocab_size != 4096:
            raise ValueError(f"vocab_size must strictly equal 4096, got {self.vocab_size}")
        if self.d_model <= 0 or self.d_model % self.n_heads != 0:
            raise ValueError(f"d_model ({self.d_model}) must be positive and divisible by n_heads ({self.n_heads})")
        if self.embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be positive, got {self.embedding_dim}")
        if self.max_seq_len <= 0 or self.max_seq_len > 512:
            raise ValueError(f"max_seq_len must be in [1, 512], got {self.max_seq_len}")
        if self.pooling_mode not in ("mean", "masked_mean", "cls"):
            raise ValueError(f"Unsupported pooling_mode: {self.pooling_mode}")
        if self.temperature <= 0.0:
            raise ValueError(f"temperature must be positive, got {self.temperature}")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SemanticEncoderConfig":
        return cls(**data)
