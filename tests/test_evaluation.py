"""
Tests for Validation Evaluator (Phase 12 & 18).
"""

import math
import pytest
import torch

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.training.loss import CausalLoss
from chakrview.training.evaluator import evaluate


def test_evaluator_deterministic_and_restores_train_mode():
    """Verify evaluator produces deterministic loss and restores model.training flag."""
    cfg = ModelConfig()
    model = ChakrMicro(cfg)
    loss_fn = CausalLoss(ignore_index=2)

    val_batch = {
        "input_ids": torch.tensor([[10, 20, 30, 40], [50, 60, 70, 80]], dtype=torch.long),
        "target_ids": torch.tensor([[20, 30, 40, 2], [60, 70, 80, 2]], dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1, 1, 0], [1, 1, 1, 0]], dtype=torch.long),
    }
    val_loader = [val_batch]

    model.train()
    assert model.training is True

    # First evaluation
    metrics_1 = evaluate(model, val_loader, loss_fn, max_batches=1)
    assert model.training is True  # Mode restored

    # Second evaluation
    metrics_2 = evaluate(model, val_loader, loss_fn, max_batches=1)

    assert metrics_1["val_loss"] == metrics_2["val_loss"]
    assert metrics_1["val_perplexity"] == metrics_2["val_perplexity"]
    assert metrics_1["eval_batches"] == 1
    assert metrics_1["val_loss"] > 0.0

    # Perplexity formula verified: exp(loss)
    expected_ppl = round(math.exp(min(20.0, metrics_1["val_loss"])), 2)
    assert metrics_1["val_perplexity"] == expected_ppl
