"""
chakrview/brain/config_15m.py
Formal experimental configuration for ChakrMicro-15M neural core.

Scales ChakrMicro from 3.44M to ~15.7M parameters (+4.9% of 15M target)
while preserving exact vocabulary (4,096), rotary embeddings, RMSNorm,
SwiGLU, tied weights, and CPU execution.
"""

from __future__ import annotations
from dataclasses import dataclass
from chakrview.brain.config import ModelConfig


@dataclass(frozen=True)
class ChakrMicro15MConfig(ModelConfig):
    """
    Experimental 15.7M parameter configuration for ChakrMicro.
    
    Dimensions:
    - vocab_size: 4096 (Identical to v0.1)
    - d_model: 384 (2x baseline width of 192)
    - n_layers: 8 (vs 6 in baseline)
    - n_heads: 6 (head_dim = 64)
    - hidden_dim: 1024 (SwiGLU intermediate dimension = 8/3 * 384)
    - max_seq_len: 512
    - tie_embeddings: True
    
    Analytical Parameter Breakdown:
    - Embedding: 4,096 * 384 = 1,572,864
    - Layer (x8):
        - Q, K, V, Out: 4 * 384 * 384 = 589,824
        - Norms (input + post-attn): 2 * 384 = 768
        - SwiGLU (Gate, Up, Down): 3 * 384 * 1024 = 1,179,648
        - Total per layer: 1,770,240
    - 8 Layers total: 14,161,920
    - Final Norm: 384
    - LM Head: Tied to embedding (0 unique extra parameters)
    - Total Unique Trainable Parameters: 1,572,864 + 14,161,920 + 384 = 15,735,168
    """
    vocab_size: int = 4096
    d_model: int = 384
    n_layers: int = 8
    n_heads: int = 6
    max_seq_len: int = 512
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    hidden_dim: int = 1024
    dropout: float = 0.0
    initializer_range: float = 0.02
    dtype: str = "float32"
    use_bias: bool = False
    tie_embeddings: bool = True
    bos_token_id: int = 0
    eos_token_id: int = 1
    pad_token_id: int = 2
