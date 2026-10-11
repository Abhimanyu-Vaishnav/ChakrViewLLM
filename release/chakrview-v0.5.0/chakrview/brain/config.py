"""
Model configuration for ChakrView Neural Core (Chakr-Micro v0.1).

Defines the exact structural hyperparameters, bounds checking, and validation rules.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    """
    Formal configuration for the ChakrMicro v0.1 neural core.
    
    Attributes:
        vocab_size: Total discrete vocabulary size (frozen at 4096).
        d_model: Hidden representation dimension (frozen at 192).
        n_layers: Number of stacked transformer layers (frozen at 6).
        n_heads: Number of attention query heads (frozen at 6).
        max_seq_len: Maximum sequence context length (frozen at 512).
        rms_norm_eps: Numerical stability constant for RMSNorm (1e-5).
        rope_theta: Base frequency for Rotary Position Embeddings (10000.0).
        hidden_dim: Intermediate SwiGLU dimension (frozen at 512, exactly (8/3)*192).
        dropout: Dropout probability (0.0 for deterministic evaluation / default).
        initializer_range: Base standard deviation for truncated normal initialization (0.02).
        dtype: Default floating point representation ("float32").
        use_bias: Whether linear layers contain additive bias (frozen at False).
        tie_embeddings: Whether input embedding and output head share weights (frozen at True).
        bos_token_id: Token ID for Beginning-Of-Sequence (0).
        eos_token_id: Token ID for End-Of-Sequence (1).
        pad_token_id: Token ID for Padding (2).
    """
    vocab_size: int = 4096
    d_model: int = 192
    n_layers: int = 6
    n_heads: int = 6
    max_seq_len: int = 512
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    hidden_dim: int = 512
    dropout: float = 0.0
    initializer_range: float = 0.02
    dtype: str = "float32"
    use_bias: bool = False
    tie_embeddings: bool = True
    bos_token_id: int = 0
    eos_token_id: int = 1
    pad_token_id: int = 2

    def __post_init__(self) -> None:
        """Validate dimensional constraints and invariants."""
        if self.vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {self.vocab_size}")
        if self.d_model <= 0:
            raise ValueError(f"d_model must be positive, got {self.d_model}")
        if self.n_layers <= 0:
            raise ValueError(f"n_layers must be positive, got {self.n_layers}")
        if self.n_heads <= 0:
            raise ValueError(f"n_heads must be positive, got {self.n_heads}")
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
            )
        head_dim = self.d_model // self.n_heads
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim ({head_dim}) must be even for pairwise RoPE rotation")
        if self.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {self.hidden_dim}")
        if self.max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be positive, got {self.max_seq_len}")
        if self.rms_norm_eps <= 0:
            raise ValueError(f"rms_norm_eps must be positive, got {self.rms_norm_eps}")
        if self.rope_theta <= 0:
            raise ValueError(f"rope_theta must be positive, got {self.rope_theta}")

    @property
    def head_dim(self) -> int:
        """Individual attention head dimension: d_model // n_heads."""
        return self.d_model // self.n_heads
