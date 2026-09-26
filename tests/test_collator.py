"""
Tests for Batch Collator (Phase 6 & 18).
"""

import pytest
import torch

from chakrview.training.collator import CausalLanguageModelingCollator


def test_collator_stacks_valid_batch():
    """Verify collator produces expected [B, T] shapes and valid masks."""
    collator = CausalLanguageModelingCollator(max_context=512)

    sample1 = {
        "input_ids": torch.tensor([10, 20, 30], dtype=torch.long),
        "target_ids": torch.tensor([20, 30, 40], dtype=torch.long),
        "attention_mask": torch.tensor([1, 1, 1], dtype=torch.long),
    }
    sample2 = {
        "input_ids": torch.tensor([50, 60, 70], dtype=torch.long),
        "target_ids": torch.tensor([60, 70, 80], dtype=torch.long),
        "attention_mask": torch.tensor([1, 1, 1], dtype=torch.long),
    }

    batch = collator([sample1, sample2])
    assert batch["input_ids"].shape == (2, 3)
    assert batch["target_ids"].shape == (2, 3)
    assert batch["attention_mask"].shape == (2, 3)


def test_collator_rejects_sequences_exceeding_max_context():
    """Verify collator explicitly rejects sequences longer than max_context."""
    collator = CausalLanguageModelingCollator(max_context=512)

    long_seq = {
        "input_ids": torch.zeros(513, dtype=torch.long),
        "target_ids": torch.zeros(513, dtype=torch.long),
        "attention_mask": torch.ones(513, dtype=torch.long),
    }

    with pytest.raises(ValueError, match="exceeds configured maximum context"):
        collator([long_seq])


def test_collator_rejects_empty_or_mismatched_batch():
    """Verify collator catches empty batch and mismatched tensor lengths."""
    collator = CausalLanguageModelingCollator()

    with pytest.raises(ValueError, match="empty batch"):
        collator([])

    mismatched = {
        "input_ids": torch.zeros(10, dtype=torch.long),
        "target_ids": torch.zeros(9, dtype=torch.long),  # Mismatch!
        "attention_mask": torch.ones(10, dtype=torch.long),
    }
    with pytest.raises(ValueError, match="Target shape"):
        collator([mismatched])
