"""
Weight initialization strategies for ChakrView.
"""

import math
import torch
import torch.nn as nn
from chakrview.brain.config import ModelConfig


def initialize_weights(model: nn.Module, config: ModelConfig) -> None:
    """
    Apply mathematically principled weight initialization to the ChakrMicro model:
    - TokenEmbedding: Normal(0, initializer_range)
    - Projections (W_q, W_k, W_v, W_gate, W_up): Normal(0, initializer_range)
    - Residual projections (W_o, W_down): Normal(0, initializer_range / sqrt(2 * n_layers))
    - RMSNorm weights: Constant(1.0)
    
    Args:
        model: The ChakrMicro instance.
        config: The ModelConfig instance.
    """
    std = config.initializer_range
    residual_std = std / math.sqrt(2.0 * config.n_layers)

    with torch.no_grad():
        # Embedding
        if hasattr(model, "embedding") and hasattr(model.embedding, "weight"):
            nn.init.normal_(model.embedding.weight, mean=0.0, std=std)

        # Transformer layers
        if hasattr(model, "layers"):
            for layer in model.layers:
                # Attention projections
                nn.init.normal_(layer.attn.q_proj.weight, mean=0.0, std=std)
                nn.init.normal_(layer.attn.k_proj.weight, mean=0.0, std=std)
                nn.init.normal_(layer.attn.v_proj.weight, mean=0.0, std=std)
                # Out projection (residual branch scaled down)
                nn.init.normal_(layer.attn.out_proj.weight, mean=0.0, std=residual_std)

                # FFN projections
                nn.init.normal_(layer.ffn.gate_proj.weight, mean=0.0, std=std)
                nn.init.normal_(layer.ffn.up_proj.weight, mean=0.0, std=std)
                # Down projection (residual branch scaled down)
                nn.init.normal_(layer.ffn.down_proj.weight, mean=0.0, std=residual_std)

                # Normalization scales
                nn.init.ones_(layer.norm_1.weight)
                nn.init.ones_(layer.norm_2.weight)

        # Final normalization
        if hasattr(model, "final_norm") and hasattr(model.final_norm, "weight"):
            nn.init.ones_(model.final_norm.weight)
