"""
ChakrMicro: The foundational indigenous decoder-only causal language model brain.
"""

from typing import Dict, Any, Optional
import torch
import torch.nn as nn
from chakrview.brain.config import ModelConfig
from chakrview.brain.embeddings import TokenEmbedding
from chakrview.brain.normalization import RMSNorm
from chakrview.brain.block import TransformerBlock
from chakrview.brain.output import LMHead
from chakrview.brain.initialization import initialize_weights


class ChakrMicro(nn.Module):
    """
    ChakrMicro v0.1 Neural Core.
    
    Architecture:
        input_ids [B, T]
        ↓
        TokenEmbedding [B, T, 192]
        ↓
        N × TransformerBlock (Pre-RMSNorm, MHA, RoPE, SwiGLU, Residuals)
        ↓
        Final RMSNorm [B, T, 192]
        ↓
        Tied LMHead [B, T, 4096] (W_out = E^T)
    """
    def __init__(self, config: Optional[ModelConfig] = None) -> None:
        super().__init__()
        if config is None:
            config = ModelConfig()
        self.config = config
        
        self.embedding = TokenEmbedding(config)
        self.layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])
        self.final_norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        
        # Strict weight tying: LMHead shares the exact parameter of embedding
        if config.tie_embeddings:
            self.lm_head = LMHead(self.embedding.weight)
        else:
            self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
            
        # Apply deterministic initialization
        initialize_weights(self, config)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass from token IDs to next-token logits.
        
        Args:
            input_ids: Integer tensor of shape [B, T] where values are in [0, V-1].
            attention_mask: Optional mask tensor of shape [B, T] (1 for valid, 0 for padding).
            
        Returns:
            Logits tensor of shape [B, T, vocab_size].
        """
        B, T = input_ids.shape
        if T > self.config.max_seq_len:
            raise ValueError(
                f"Sequence length {T} exceeds maximum context window {self.config.max_seq_len}."
            )
            
        # 1. Token embeddings: [B, T, d_model]
        x = self.embedding(input_ids)
        
        # 2. Sequential transformer blocks: [B, T, d_model]
        for layer in self.layers:
            x = layer(x, attention_mask=attention_mask)
            
        # 3. Final Pre-RMSNorm: [B, T, d_model]
        x = self.final_norm(x)
        
        # 4. Tied projection to vocabulary logits: [B, T, vocab_size]
        logits = self.lm_head(x)
        return logits

    def count_parameters(self) -> Dict[str, int]:
        """
        Programmatically calculate and report parameters across all subcomponents.
        
        Returns:
            Dictionary with breakdown of parameter counts.
        """
        emb_params = self.embedding.weight.numel()
        
        attn_params = sum(
            layer.attn.q_proj.weight.numel() +
            layer.attn.k_proj.weight.numel() +
            layer.attn.v_proj.weight.numel() +
            layer.attn.out_proj.weight.numel()
            for layer in self.layers
        )
        
        ffn_params = sum(
            layer.ffn.gate_proj.weight.numel() +
            layer.ffn.up_proj.weight.numel() +
            layer.ffn.down_proj.weight.numel()
            for layer in self.layers
        )
        
        norm_params = (
            sum(layer.norm_1.weight.numel() + layer.norm_2.weight.numel() for layer in self.layers) +
            self.final_norm.weight.numel()
        )
        
        layer_block_params = (attn_params + ffn_params + sum(layer.norm_1.weight.numel() + layer.norm_2.weight.numel() for layer in self.layers)) // self.config.n_layers
        
        # Collect unique parameter tensors in memory to respect weight tying
        unique_params = set(self.parameters())
        total_unique = sum(p.numel() for p in unique_params)
        total_trainable = sum(p.numel() for p in unique_params if p.requires_grad)
        
        return {
            "embedding": emb_params,
            "transformer_block_per_layer": layer_block_params,
            "transformer_blocks_total": layer_block_params * self.config.n_layers,
            "attention_per_layer": attn_params // self.config.n_layers,
            "attention_total": attn_params,
            "ffn_per_layer": ffn_params // self.config.n_layers,
            "ffn_total": ffn_params,
            "normalization_total": norm_params,
            "output_head_unique": 0,
            "total_parameters": total_unique,
            "trainable_parameters": total_trainable,
        }
