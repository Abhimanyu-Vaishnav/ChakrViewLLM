"""
Optimizer and Learning Rate Scheduler Builder for ChakrView (Phase 8).

Features:
- Parameter segregation: Decoupled weight decay for 2D weight matrices (embeddings, projections);
  zero weight decay for 1D normalization vectors.
- Duplicate parameter elimination to respect weight tying.
- Cosine learning rate scheduler with linear warmup.
"""

import math
from typing import Tuple, List, Dict, Any
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from chakrview.training.config import TrainingHyperparameters


def get_parameter_groups(
    model: nn.Module, weight_decay: float = 0.01
) -> List[Dict[str, Any]]:
    """
    Split model parameters into decayed (2D weights) and non-decayed (1D norms).
    Ensures tied parameters appear in exactly one group.
    """
    decay_params: List[nn.Parameter] = []
    no_decay_params: List[nn.Parameter] = []
    seen_params = set()

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Avoid duplicate entries for tied parameters (e.g. lm_head and embedding)
        if param in seen_params:
            continue
        seen_params.add(param)

        if param.dim() >= 2:
            decay_params.append(param)
        else:
            no_decay_params.append(param)

    return [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]


def build_optimizer(
    model: nn.Module, config: TrainingHyperparameters
) -> torch.optim.Optimizer:
    """
    Construct optimizer from training hyperparameters.
    """
    param_groups = get_parameter_groups(model, weight_decay=config.weight_decay)
    
    if config.optimizer.lower() == "adamw":
        optimizer = AdamW(
            param_groups,
            lr=config.learning_rate,
            betas=(config.adam_beta1, config.adam_beta2),
            eps=config.adam_eps,
        )
    else:
        raise ValueError(f"Unsupported optimizer: {config.optimizer}")

    return optimizer


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer, config: TrainingHyperparameters
) -> LambdaLR:
    """
    Construct cosine decay learning rate scheduler with linear warmup.
    """
    warmup_steps = config.warmup_steps
    max_steps = config.max_steps
    min_lr_ratio = config.min_learning_rate / config.learning_rate

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, warmup_steps))
        if config.lr_decay_style == "constant":
            return 1.0
        # Cosine decay down to min_lr_ratio
        progress = float(current_step - warmup_steps) / float(
            max(1, max_steps - warmup_steps)
        )
        progress = min(1.0, max(0.0, progress))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    return LambdaLR(optimizer, lr_lambda)
