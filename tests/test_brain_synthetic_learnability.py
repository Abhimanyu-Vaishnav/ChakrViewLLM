"""
Automated Synthetic Learnability Test for ChakrMicro (Phase 11).

Proves that:
1. ChakrMicro starts at uniform entropy loss (~ln(4096) = 8.318).
2. Optimization over 20 steps of AdamW drops loss below 2.0.
3. Accuracy on the queried association reaches 100%.
"""

import pytest
import torch
import torch.nn.functional as F

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from scripts.run_synthetic_learnability import generate_synthetic_data


def test_associative_recall_learnability():
    torch.manual_seed(42)
    cfg = ModelConfig()
    model = ChakrMicro(cfg)
    model.train()

    inputs, targets = generate_synthetic_data(num_samples=16, seed=42)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-2)

    # Initial step
    logits = model(inputs)[:, -1, :]
    initial_loss = F.cross_entropy(logits, targets).item()
    assert 7.5 < initial_loss < 9.5, f"Unexpected initial loss {initial_loss}"

    # Train for 25 steps
    for _ in range(25):
        optimizer.zero_grad()
        logits = model(inputs)[:, -1, :]
        loss = F.cross_entropy(logits, targets)
        loss.backward()
        optimizer.step()

    final_loss = loss.item()
    preds = torch.argmax(logits.detach(), dim=-1)
    acc = (preds == targets).float().mean().item()

    assert final_loss < 0.5, f"Expected final loss < 0.5, got {final_loss:.4f}"
    assert acc == 1.0, f"Expected 100% accuracy, got {acc * 100:.1f}%"
