"""
Batch 39 Robust SFT Batch Safety Guard.

Implements a fail-closed validation guard for batches before forward/loss computation:
1. Validates that active_tokens > 0 (prevents all-masked / NaN loss).
2. Validates that target token IDs are within vocabulary bounds [0, vocab_size).
3. Validates that sequence length does not exceed max_seq_len.
4. Validates that EOS token is present in supervised targets.
5. Safely rejects or skips invalid batches without propagating non-finite loss or bad gradients.
"""

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


class SFTBatchSafetyGuard:
    """
    Enforces fail-closed safety constraints on SFT training batches.
    Guarantees that invalid, empty-target, out-of-bounds, or overlength batches
    are caught and rejected before causing non-finite loss or gradient corruption.
    """

    def __init__(
        self,
        vocab_size: int = 4096,
        max_seq_len: int = 512,
        ignore_index: int = -100,
        eos_token_id: int = 1,
    ):
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.ignore_index = ignore_index
        self.eos_token_id = eos_token_id

    def validate_batch(
        self,
        input_ids: torch.Tensor,
        target_ids: torch.Tensor,
        active_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Validates batch tensors against fail-closed safety rules.
        Returns a dict with 'is_valid': bool, 'status': str, 'reason': Optional[str].
        """
        # 1. Check tensor dimensions
        if input_ids.dim() != 2 or target_ids.dim() != 2:
            return {
                "is_valid": False,
                "status": "INVALID_SHAPE",
                "reason": f"Expected 2D tensors, got {input_ids.shape} and {target_ids.shape}",
            }

        if input_ids.shape != target_ids.shape:
            return {
                "is_valid": False,
                "status": "SHAPE_MISMATCH",
                "reason": f"Input shape {input_ids.shape} != target shape {target_ids.shape}",
            }

        batch_size, seq_len = input_ids.shape

        # 2. Sequence length check
        if seq_len > self.max_seq_len:
            return {
                "is_valid": False,
                "status": "EXCEEDS_CONTEXT_LENGTH",
                "reason": f"Sequence length {seq_len} exceeds max_seq_len {self.max_seq_len}",
            }

        # 3. Active target token count check (fail-closed against all-masked batches)
        computed_active = int((target_ids != self.ignore_index).sum().item())
        if active_tokens is not None and active_tokens != computed_active:
            return {
                "is_valid": False,
                "status": "ACTIVE_TOKEN_COUNT_MISMATCH",
                "reason": f"Reported active {active_tokens} != computed active {computed_active}",
            }

        if computed_active == 0:
            return {
                "is_valid": False,
                "status": "ALL_TARGETS_MASKED",
                "reason": "All target tokens are masked with ignore_index. Loss would be NaN.",
            }

        # 4. Input vocabulary bounds check
        if (input_ids < 0).any() or (input_ids >= self.vocab_size).any():
            return {
                "is_valid": False,
                "status": "INPUT_VOCAB_OUT_OF_BOUNDS",
                "reason": f"Input token IDs contain values outside [0, {self.vocab_size})",
            }

        # 5. Target vocabulary bounds check (ignoring ignore_index)
        active_mask = target_ids != self.ignore_index
        active_targets = target_ids[active_mask]
        if (active_targets < 0).any() or (active_targets >= self.vocab_size).any():
            return {
                "is_valid": False,
                "status": "TARGET_VOCAB_OUT_OF_BOUNDS",
                "reason": f"Active target token IDs contain values outside [0, {self.vocab_size})",
            }

        # 6. EOS target presence check
        # In causal SFT, every supervised assistant sequence must end with EOS
        has_eos = (active_targets == self.eos_token_id).any().item()
        if not has_eos:
            return {
                "is_valid": False,
                "status": "MISSING_EOS_SUPERVISION",
                "reason": f"Active targets do not contain expected EOS token ID ({self.eos_token_id})",
            }

        return {
            "is_valid": True,
            "status": "VALID",
            "active_tokens": computed_active,
            "reason": None,
        }

    def safe_compute_loss(
        self,
        model: torch.nn.Module,
        input_ids: torch.Tensor,
        target_ids: torch.Tensor,
        active_tokens: Optional[int] = None,
    ) -> Tuple[Optional[torch.Tensor], Dict[str, Any]]:
        """
        Safely computes cross-entropy loss if batch passes validation.
        Returns (loss_tensor, validation_report).
        If invalid, returns (None, validation_report) without running forward pass or producing NaN.
        """
        report = self.validate_batch(input_ids, target_ids, active_tokens=active_tokens)
        if not report["is_valid"]:
            return None, report

        logits = model(input_ids)
        loss = F.cross_entropy(
            logits.view(-1, self.vocab_size),
            target_ids.view(-1),
            ignore_index=self.ignore_index,
        )

        if not math.isfinite(loss.item()):
            return None, {
                "is_valid": False,
                "status": "NON_FINITE_LOSS",
                "reason": f"Computed loss was non-finite: {loss.item()}",
            }

        return loss, report
