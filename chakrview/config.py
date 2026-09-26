"""
ChakrView Neural Core Configuration & Tensor Contract Specification.

This module provides the formal structural configuration, dimensional validation,
parameter accounting, memory estimation, and FLOP calculation utilities for the
ChakrView Neural Core architecture (Chakr-Micro v0.1).

ARCHITECTURAL PRINCIPLE:
Strictly zero neural model implementation, zero weights, zero training logic,
and zero external ML framework dependencies. Pure standard library.
"""

from dataclasses import dataclass
from typing import Dict, Any, Tuple


@dataclass(frozen=True)
class ChakrConfig:
    """
    Formal architectural configuration for Chakr-Micro v0.1 neural core.
    
    Attributes:
        vocab_size (V): Total vocabulary size (frozen at 4096).
        d_model (int): Hidden representation dimension (frozen at 192).
        num_layers (N): Number of stacked transformer blocks (frozen at 6 for v0.1).
        num_heads (H): Number of attention query heads (frozen at 6 for v0.1).
        num_kv_heads (H_kv): Number of key/value heads for MHA (frozen at 6 for v0.1).
        d_ff (int): SwiGLU intermediate dimension (frozen at 512, exactly (8/3)*192).
        max_seq_len (T_max): Maximum context window (frozen at 512).
        rope_theta (float): Base frequency for Rotary Position Embeddings (10000.0).
        rope_dim (int): Dimensionality receiving RoPE (frozen at 32 = d_head).
        rms_norm_eps (float): Epsilon for numerical stability in RMSNorm (1e-5).
        bias (bool): Whether linear projections include additive bias (frozen at False).
        tie_weights (bool): Whether input embedding and LM head are tied (frozen at True).
        bos_token_id (int): Special token ID for Beginning-Of-Sequence (0).
        eos_token_id (int): Special token ID for End-Of-Sequence (1).
        pad_token_id (int): Special token ID for Padding (2).
    """
    vocab_size: int = 4096
    d_model: int = 192
    num_layers: int = 6
    num_heads: int = 6
    num_kv_heads: int = 6
    d_ff: int = 512
    max_seq_len: int = 512
    rope_theta: float = 10000.0
    rope_dim: int = 32
    rms_norm_eps: float = 1e-5
    bias: bool = False
    tie_weights: bool = True
    bos_token_id: int = 0
    eos_token_id: int = 1
    pad_token_id: int = 2

    def __post_init__(self) -> None:
        """Validate architectural integrity and divisibility invariants."""
        if self.vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {self.vocab_size}")
        if self.d_model <= 0:
            raise ValueError(f"d_model must be positive, got {self.d_model}")
        if self.num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {self.num_layers}")
        if self.num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {self.num_heads}")
        if self.d_model % self.num_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by num_heads ({self.num_heads})"
            )
        if self.num_kv_heads != self.num_heads:
            # v0.1 Simple MHA requirement
            raise ValueError(
                f"v0.1 requires Simple MHA (num_kv_heads == num_heads), "
                f"got num_kv_heads={self.num_kv_heads}, num_heads={self.num_heads}"
            )
        head_dim = self.d_model // self.num_heads
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim ({head_dim}) must be even for pairwise RoPE rotation")
        if self.rope_dim > head_dim:
            raise ValueError(f"rope_dim ({self.rope_dim}) cannot exceed head_dim ({head_dim})")
        if self.rope_dim % 2 != 0:
            raise ValueError(f"rope_dim ({self.rope_dim}) must be even for pairwise RoPE rotation")
        if self.d_ff <= 0:
            raise ValueError(f"d_ff must be positive, got {self.d_ff}")
        if self.max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be positive, got {self.max_seq_len}")

    @property
    def head_dim(self) -> int:
        """Compute individual attention head dimension: d_model / num_heads."""
        return self.d_model // self.num_heads


def calculate_parameter_breakdown(config: ChakrConfig) -> Dict[str, Any]:
    """
    Calculate the exact parameter counts across all model subcomponents.
    
    Args:
        config: The ChakrConfig instance.
        
    Returns:
        Dictionary containing granular parameter breakdown and memory footprints.
    """
    # 1. Embedding
    embedding_params = config.vocab_size * config.d_model
    
    # 2. Per-Layer Parameters
    # Attention: W_q, W_k, W_v, W_o (all bias-free)
    attn_q = config.d_model * (config.num_heads * config.head_dim)
    attn_k = config.d_model * (config.num_kv_heads * config.head_dim)
    attn_v = config.d_model * (config.num_kv_heads * config.head_dim)
    attn_o = (config.num_heads * config.head_dim) * config.d_model
    attn_params = attn_q + attn_k + attn_v + attn_o
    
    # RMSNorm 1 (Attention pre-norm scale gamma, no bias)
    rmsnorm_1_params = config.d_model
    
    # SwiGLU: W_gate, W_up, W_down (all bias-free)
    swiglu_gate = config.d_model * config.d_ff
    swiglu_up = config.d_model * config.d_ff
    swiglu_down = config.d_ff * config.d_model
    ffn_params = swiglu_gate + swiglu_up + swiglu_down
    
    # RMSNorm 2 (FFN pre-norm scale gamma, no bias)
    rmsnorm_2_params = config.d_model
    
    layer_params = attn_params + rmsnorm_1_params + ffn_params + rmsnorm_2_params
    total_layers_params = config.num_layers * layer_params
    
    # 3. Final RMSNorm
    final_norm_params = config.d_model
    
    # 4. Output LM Head
    # If tied, shares embedding matrix (0 additional unique parameters)
    # If untied, requires config.d_model * config.vocab_size
    lm_head_params = 0 if config.tie_weights else (config.d_model * config.vocab_size)
    
    total_params = embedding_params + total_layers_params + final_norm_params + lm_head_params
    
    # Static weight memory footprints in bytes
    fp32_bytes = total_params * 4
    fp16_bytes = total_params * 2
    int8_bytes = total_params * 1
    int4_bytes = int(total_params * 0.5)
    
    return {
        "vocab_size": config.vocab_size,
        "d_model": config.d_model,
        "num_layers": config.num_layers,
        "num_heads": config.num_heads,
        "head_dim": config.head_dim,
        "d_ff": config.d_ff,
        "embedding_params": embedding_params,
        "per_layer": {
            "attn_q": attn_q,
            "attn_k": attn_k,
            "attn_v": attn_v,
            "attn_o": attn_o,
            "attn_total": attn_params,
            "rmsnorm_1": rmsnorm_1_params,
            "ffn_gate": swiglu_gate,
            "ffn_up": swiglu_up,
            "ffn_down": swiglu_down,
            "ffn_total": ffn_params,
            "rmsnorm_2": rmsnorm_2_params,
            "total_layer_params": layer_params,
        },
        "all_layers_params": total_layers_params,
        "final_norm_params": final_norm_params,
        "lm_head_params": lm_head_params,
        "total_parameters": total_params,
        "embedding_fraction": embedding_params / total_params,
        "layers_fraction": total_layers_params / total_params,
        "static_memory": {
            "fp32_bytes": fp32_bytes,
            "fp32_mib": fp32_bytes / (1024 ** 2),
            "fp16_bytes": fp16_bytes,
            "fp16_mib": fp16_bytes / (1024 ** 2),
            "int8_bytes": int8_bytes,
            "int8_mib": int8_bytes / (1024 ** 2),
            "int4_bytes": int4_bytes,
            "int4_mib": int4_bytes / (1024 ** 2),
        }
    }


def calculate_memory_budget(
    config: ChakrConfig,
    batch_size: int = 1,
    seq_len: int = 512
) -> Dict[str, Any]:
    """
    Calculate static weight memory, KV cache memory, and dynamic activation memory.
    
    Args:
        config: The ChakrConfig instance.
        batch_size: Batch dimension (B).
        seq_len: Context sequence length (T).
        
    Returns:
        Granular breakdown of memory requirements.
    """
    if seq_len > config.max_seq_len:
        raise ValueError(
            f"seq_len ({seq_len}) exceeds configured max_seq_len ({config.max_seq_len})"
        )
    
    params_breakdown = calculate_parameter_breakdown(config)
    
    # KV Cache:
    # Keys and Values stored for each layer: [B, H_kv, T, d_head]
    # Total elements per layer = 2 * B * H_kv * T * d_head = 2 * B * T * d_model
    kv_elements_per_layer = 2 * batch_size * seq_len * config.d_model
    total_kv_elements = config.num_layers * kv_elements_per_layer
    
    kv_fp32_bytes = total_kv_elements * 4
    kv_fp16_bytes = total_kv_elements * 2
    kv_int8_bytes = total_kv_elements * 1
    
    # Training Activations (stored for backward pass per layer):
    # - norm1: [B, T, d_model]
    # - Q, K, V: 3 * [B, T, d_model]
    # - Attn scores + softmax probs: 2 * [B, H, T, T]
    # - Attn context: [B, T, d_model]
    # - Attn proj out: [B, T, d_model]
    # - norm2: [B, T, d_model]
    # - gate, up, swish(gate), gate*up: 4 * [B, T, d_ff]
    # - down out: [B, T, d_model]
    layer_act_elements = (
        batch_size * seq_len * config.d_model * 7 +
        2 * batch_size * config.num_heads * seq_len * seq_len +
        4 * batch_size * seq_len * config.d_ff
    )
    total_train_act_elements = (
        config.num_layers * layer_act_elements +
        2 * batch_size * seq_len * config.d_model  # emb + final norm
    )
    
    # Inference Working Buffer (scratchpad for single forward pass):
    # Largest single activation tensor during sequence processing:
    # Attn scores: [B, H, T, T]
    # Intermediate SwiGLU: [B, T, d_ff]
    # Logits: [B, T, V]
    inference_scratch_elements = max(
        batch_size * config.num_heads * seq_len * seq_len,
        batch_size * seq_len * config.d_ff,
        batch_size * seq_len * config.vocab_size
    ) + (batch_size * seq_len * config.d_model * 4)
    
    return {
        "batch_size": batch_size,
        "seq_len": seq_len,
        "static_weights": params_breakdown["static_memory"],
        "kv_cache": {
            "total_elements": total_kv_elements,
            "fp32_bytes": kv_fp32_bytes,
            "fp32_mib": kv_fp32_bytes / (1024 ** 2),
            "fp16_bytes": kv_fp16_bytes,
            "fp16_mib": kv_fp16_bytes / (1024 ** 2),
            "int8_bytes": kv_int8_bytes,
            "int8_mib": kv_int8_bytes / (1024 ** 2),
        },
        "training_activations": {
            "total_elements": total_train_act_elements,
            "fp32_bytes": total_train_act_elements * 4,
            "fp32_mib": (total_train_act_elements * 4) / (1024 ** 2),
            "fp16_bytes": total_train_act_elements * 2,
            "fp16_mib": (total_train_act_elements * 2) / (1024 ** 2),
        },
        "inference_scratchpad": {
            "estimated_elements": inference_scratch_elements,
            "fp32_bytes": inference_scratch_elements * 4,
            "fp32_mib": (inference_scratch_elements * 4) / (1024 ** 2),
            "fp16_bytes": inference_scratch_elements * 2,
            "fp16_mib": (inference_scratch_elements * 2) / (1024 ** 2),
        }
    }


def calculate_flops_breakdown(config: ChakrConfig, seq_len: int) -> Dict[str, Any]:
    """
    Calculate theoretical floating-point operations (FLOPs) for a sequence of length T.
    
    Accounts for:
    - Linear GEMMs (QKV, OutProj, SwiGLU Gate/Up/Down, LM Head) using 2*M*K*N FLOPs.
    - Attention quadratic matrix multiplications (Q*K^T, S*V) using 2*T^2*d_model per layer.
    - Non-linear elementwise operations (RoPE, Softmax, Swish, RMSNorm, Residuals).
    """
    T = seq_len
    B = 1
    d = config.d_model
    H = config.num_heads
    d_k = config.head_dim
    d_ff = config.d_ff
    V = config.vocab_size
    N = config.num_layers
    
    # 1. Attention Projections (Q, K, V, O)
    # Q, K, V: 3 * (2 * T * d * d)
    # O: 2 * T * d * d
    attn_linear_layer = 4 * (2 * T * d * d)
    
    # 2. Attention Quadratic Operations
    # Scores: Q * K^T -> H * (2 * T * T * d_k) = 2 * T^2 * d
    # Context: S * V -> H * (2 * T * T * d_k) = 2 * T^2 * d
    # Softmax: ~3 * H * T^2 (max, exp, sum, div)
    attn_quad_layer = 4 * T * T * d + 3 * H * T * T
    
    # 3. SwiGLU FFN
    # Gate, Up: 2 * (2 * T * d * d_ff)
    # Down: 2 * T * d_ff * d
    # Activation: Swish + elementwise multiply: ~4 * T * d_ff
    ffn_layer = 6 * T * d * d_ff + 4 * T * d_ff
    
    # 4. Normalization, RoPE, Residuals
    # RoPE: 6 * T * d (3 FLOPs/elem for 2D rotation of Q and K)
    # RMSNorm: 2 norms per layer * 2 * T * d
    # Residuals: 2 additions per layer * T * d
    misc_layer = 6 * T * d + 4 * T * d + 2 * T * d
    
    layer_flops = attn_linear_layer + attn_quad_layer + ffn_layer + misc_layer
    all_layers_flops = N * layer_flops
    
    # 5. Model Head & Final Norm
    final_norm_flops = 2 * T * d
    lm_head_flops = 2 * T * d * V
    
    total_seq_flops = all_layers_flops + final_norm_flops + lm_head_flops
    flops_per_token = total_seq_flops / T
    
    return {
        "seq_len": T,
        "attn_linear_all_layers": N * attn_linear_layer,
        "attn_quadratic_all_layers": N * attn_quad_layer,
        "ffn_all_layers": N * ffn_layer,
        "misc_all_layers": N * misc_layer,
        "final_norm": final_norm_flops,
        "lm_head": lm_head_flops,
        "total_sequence_flops": total_seq_flops,
        "flops_per_token": flops_per_token,
        "total_mflops": total_seq_flops / 1e6,
        "mflops_per_token": flops_per_token / 1e6,
    }


def get_tensor_forward_contracts(
    config: ChakrConfig,
    batch_size: int = 1,
    seq_len: int = 512
) -> Dict[str, Tuple[int, ...]]:
    """
    Generate the exact tensor shapes produced at every stage of the forward pass.
    """
    B = batch_size
    T = seq_len
    d = config.d_model
    H = config.num_heads
    d_k = config.head_dim
    d_ff = config.d_ff
    V = config.vocab_size
    
    return {
        "input_ids": (B, T),
        "attention_mask": (B, T),
        "embedded_tokens": (B, T, d),
        "layer_input_residual": (B, T, d),
        "layer_norm_1_output": (B, T, d),
        "q_projected_flat": (B, T, d),
        "k_projected_flat": (B, T, d),
        "v_projected_flat": (B, T, d),
        "q_heads_unrotated": (B, H, T, d_k),
        "k_heads_unrotated": (B, H, T, d_k),
        "v_heads": (B, H, T, d_k),
        "q_heads_rotated": (B, H, T, d_k),
        "k_heads_rotated": (B, H, T, d_k),
        "causal_mask": (1, 1, T, T),
        "attention_raw_scores": (B, H, T, T),
        "attention_masked_scores": (B, H, T, T),
        "attention_probabilities": (B, H, T, T),
        "attention_context_heads": (B, H, T, d_k),
        "attention_context_flat": (B, T, d),
        "attention_output_projected": (B, T, d),
        "residual_after_attention": (B, T, d),
        "layer_norm_2_output": (B, T, d),
        "swiglu_gate_proj": (B, T, d_ff),
        "swiglu_up_proj": (B, T, d_ff),
        "swiglu_gate_activated": (B, T, d_ff),
        "swiglu_gated_up": (B, T, d_ff),
        "swiglu_down_proj": (B, T, d),
        "layer_block_output": (B, T, d),
        "final_norm_output": (B, T, d),
        "output_logits": (B, T, V),
    }
