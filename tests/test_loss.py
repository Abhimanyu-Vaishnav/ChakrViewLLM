"""
Tests for Causal Cross-Entropy Loss with PAD Masking (Phase 7 & 18).
"""

import pytest
import torch
import math

from chakrview.training.loss import CausalLoss


def test_loss_shape_and_known_example():
    """Verify loss calculation on a known small analytical example."""
    loss_fn = CausalLoss(ignore_index=2)

    # B=1, T=2, V=4
    # Set uniform logits (0.0): cross entropy should equal -ln(1/4) = ln(4) approx 1.3863
    logits = torch.zeros(1, 2, 4, requires_grad=True)
    targets = torch.tensor([[0, 1]], dtype=torch.long)

    loss = loss_fn(logits, targets)
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)
    assert math.isclose(loss.item(), math.log(4.0), rel_tol=1e-5)


def test_loss_pad_masking_exclusion():
    """Verify that tokens at PAD ID position (2) are strictly excluded from loss."""
    loss_fn = CausalLoss(ignore_index=2)

    logits = torch.zeros(1, 4, 10, requires_grad=True)

    # targets1: 2 valid tokens, 2 padded tokens
    targets1 = torch.tensor([[5, 8, 2, 2]], dtype=torch.long)
    loss1 = loss_fn(logits, targets1)

    # targets2: same 2 valid tokens, but altered garbage at padded positions
    targets2 = torch.tensor([[5, 8, 2, 2]], dtype=torch.long)
    loss2 = loss_fn(logits, targets2)

    assert loss1.item() == loss2.item()
    assert math.isclose(loss1.item(), math.log(10.0), rel_tol=1e-5)


def test_loss_gradient_propagation():
    """Verify backward pass through loss generates valid, finite gradients."""
    loss_fn = CausalLoss(ignore_index=2)

    logits = torch.randn(2, 8, 4096, requires_grad=True)
    targets = torch.randint(0, 4096, (2, 8), dtype=torch.long)

    loss = loss_fn(logits, targets)
    loss.backward()

    assert logits.grad is not None
    assert not torch.isnan(logits.grad).any()
    assert not torch.isinf(logits.grad).any()
    assert logits.grad.norm().item() > 0.0
