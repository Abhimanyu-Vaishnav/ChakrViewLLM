"""
Lightweight Deterministic Validation Evaluator for ChakrView (Phase 12).

Computes mean validation cross-entropy loss and perplexity on a fixed subset
of validation batches without gradients or training state mutation.
"""

import math
from typing import Dict, Any, Iterable
import torch
import torch.nn as nn


def evaluate(
    model: nn.Module,
    val_loader: Iterable[Dict[str, torch.Tensor]],
    loss_fn: nn.Module,
    max_batches: int = 5,
    device: str = "cpu",
) -> Dict[str, float]:
    """
    Evaluate model on validation loader subset.
    
    Args:
        model: ChakrMicro instance.
        val_loader: Iterable yielding batch dictionaries.
        loss_fn: CausalLoss instance.
        max_batches: Number of batches to evaluate.
        device: Computation device ("cpu").
        
    Returns:
        Dictionary with val_loss, val_perplexity, and eval_batches.
    """
    was_training = model.training
    model.eval()

    total_loss = 0.0
    batches_evaluated = 0

    with torch.no_grad():
        for batch in val_loader:
            if batches_evaluated >= max_batches:
                break

            input_ids = batch["input_ids"].to(device)
            target_ids = batch["target_ids"].to(device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

            logits = model(input_ids, attention_mask=attention_mask)
            loss = loss_fn(logits, target_ids)

            total_loss += loss.item()
            batches_evaluated += 1

    if was_training:
        model.train()

    if batches_evaluated == 0:
        return {"val_loss": float("nan"), "val_perplexity": float("nan"), "eval_batches": 0}

    mean_loss = total_loss / batches_evaluated
    rounded_loss = round(mean_loss, 4)
    perplexity = math.exp(min(20.0, rounded_loss)) if not math.isnan(rounded_loss) else float("nan")

    return {
        "val_loss": rounded_loss,
        "val_perplexity": round(perplexity, 2),
        "eval_batches": batches_evaluated,
    }
