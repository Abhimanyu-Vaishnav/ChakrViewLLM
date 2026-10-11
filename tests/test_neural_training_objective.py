"""
Tests for Milestone 2: Verification and Correction of Neural Training Objective.
Validates:
1. Causal attention masking (future token leakage prevention).
2. Next-token alignment and shift mechanics.
3. Gradient flow and bounds checking.
4. Error handling for non-finite values, empty sequences, and invalid IDs.
"""

import pytest
import torch
import torch.nn as nn
from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.training.loss import CausalLoss
from chakrview.training.safety import TrainingSafetyChecker


def test_causal_masking_no_future_leakage():
    """
    Verify that altering a future token (at position j > i) does NOT alter the
    logits or hidden representations at position i.
    """
    torch.manual_seed(42)
    config = ModelConfig(vocab_size=4096, d_model=192, n_layers=2, n_heads=4, max_seq_len=64)
    model = ChakrMicro(config)
    model.eval()

    # Original sequence
    seq1 = torch.tensor([[10, 20, 30, 40, 50]], dtype=torch.long)
    # Modified sequence: only position 4 is changed
    seq2 = torch.tensor([[10, 20, 30, 40, 999]], dtype=torch.long)

    with torch.no_grad():
        logits1 = model(seq1)  # [1, 5, 4096]
        logits2 = model(seq2)  # [1, 5, 4096]

    # For positions 0, 1, 2, 3: logits must be exactly identical
    for pos in range(4):
        diff = (logits1[0, pos] - logits2[0, pos]).abs().max().item()
        assert diff == 0.0, f"Future token at pos 4 influenced pos {pos} (max diff={diff})"

    # Pos 4 should be different because its own input is different
    diff_pos4 = (logits1[0, 4] - logits2[0, 4]).abs().max().item()
    assert diff_pos4 > 0.0, "Expected position 4 to differ between sequences"


def test_next_token_target_alignment():
    """
    Verify that target tokens correctly align with input tokens shifted by 1.
    """
    raw_sequence = torch.tensor([[10, 20, 30, 40, 50]], dtype=torch.long)
    input_ids = raw_sequence[:, :-1]   # [10, 20, 30, 40]
    target_ids = raw_sequence[:, 1:]   # [20, 30, 40, 50]

    assert input_ids.shape == (1, 4)
    assert target_ids.shape == (1, 4)
    assert input_ids[0, 0].item() == 10 and target_ids[0, 0].item() == 20
    assert input_ids[0, 3].item() == 40 and target_ids[0, 3].item() == 50


def test_causal_loss_and_padding_exclusion():
    """
    Verify CausalLoss ignores pad_token_id (2) and computes finite scalar cross-entropy.
    """
    loss_fn = CausalLoss(ignore_index=2)
    B, T, V = 2, 4, 10
    torch.manual_seed(42)
    logits = torch.randn(B, T, V, requires_grad=True)
    # Targets with some PAD tokens
    targets = torch.tensor([
        [3, 4, 2, 2],
        [1, 5, 7, 2],
    ], dtype=torch.long)

    loss = loss_fn(logits, targets)
    assert torch.isfinite(loss)
    assert loss.item() > 0.0

    loss.backward()
    assert logits.grad is not None
    # Gradients at pad positions should be zero
    assert logits.grad[0, 2].abs().sum().item() == 0.0
    assert logits.grad[0, 3].abs().sum().item() == 0.0
    assert logits.grad[1, 3].abs().sum().item() == 0.0
    # Gradients at non-pad positions should be non-zero
    assert logits.grad[0, 0].abs().sum().item() > 0.0


def test_training_safety_bounds_and_errors():
    """
    Verify validation of token ranges, non-finite loss, and gradient checks.
    """
    # 1. Invalid token ID check
    invalid_ids = torch.tensor([[10, 5000]], dtype=torch.long)
    from chakrview.training.safety import TrainingSafetyError
    with pytest.raises((ValueError, TrainingSafetyError), match="out of bounds"):
        TrainingSafetyChecker.verify_token_ids(invalid_ids, min_id=0, max_id=4095, allowed_ignore_index=2)

    # 2. Non-finite loss check
    from chakrview.training.safety import NumericalInstabilityError
    inf_loss = torch.tensor(float("inf"))
    with pytest.raises((FloatingPointError, NumericalInstabilityError), match="Infinite|non-finite"):
        TrainingSafetyChecker.check_loss(inf_loss, step=1)

    # 3. Shape mismatch between logits and targets
    loss_fn = CausalLoss(ignore_index=2)
    with pytest.raises(ValueError, match="does not match logits prefix"):
        loss_fn(torch.randn(2, 4, 10), torch.tensor([[1, 2], [3, 4]]))
