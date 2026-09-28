"""
Training Safety & Invariant Verification for ChakrView (Step 22).

Enforces hard mathematical and architectural safety constraints during training:
1. Detects and fails closed on NaN/Inf loss or gradients.
2. Verifies token IDs strictly adhere to vocabulary bounds [0, 4095].
3. Validates frozen neural invariants:
   - Parameters == 3,443,136
   - Vocab size == 4096
   - Context length == 512
   - Special tokens: BOS=0, EOS=1, PAD=2
4. Verifies tokenizer and checkpoint compatibility before training resume.
"""

import math
from typing import Dict, List, Optional, Any, Tuple
import torch
import torch.nn as nn


class TrainingSafetyError(Exception):
    """Base exception for all training safety and invariant violations."""
    pass


class NumericalInstabilityError(TrainingSafetyError):
    """Raised when loss or gradients encounter NaN, Inf, or unrecoverable instability."""
    pass


class InvariantViolationError(TrainingSafetyError):
    """Raised when frozen neural invariants (parameters, vocab, context) are violated."""
    pass


class CheckpointCorruptionError(TrainingSafetyError):
    """Raised when a checkpoint file is incomplete, corrupted, or incompatible."""
    pass


class TrainingSafetyChecker:
    """
    Evaluates safety conditions across training batches, model weights, and optimization steps.
    """

    FROZEN_PARAMS: int = 3_443_136
    FROZEN_VOCAB: int = 4_096
    FROZEN_CONTEXT: int = 512
    FROZEN_BOS: int = 0
    FROZEN_EOS: int = 1
    FROZEN_PAD: int = 2

    @classmethod
    def verify_model_invariants(cls, model: nn.Module) -> Tuple[bool, List[str]]:
        """
        Verify that model conforms strictly to frozen ChakrMicro v0.1 invariants.
        """
        violations: List[str] = []

        # 1. Parameter count check
        total_params = sum(p.numel() for p in model.parameters())
        if total_params != cls.FROZEN_PARAMS:
            violations.append(
                f"Parameter invariant violation: model has {total_params:,} params; "
                f"expected exactly {cls.FROZEN_PARAMS:,}."
            )

        # 2. Config checks if available
        if hasattr(model, "config"):
            cfg = model.config
            if hasattr(cfg, "vocab_size") and cfg.vocab_size != cls.FROZEN_VOCAB:
                violations.append(
                    f"Vocabulary invariant violation: config.vocab_size={cfg.vocab_size}; "
                    f"expected {cls.FROZEN_VOCAB}."
                )
            if hasattr(cfg, "max_seq_len") and cfg.max_seq_len != cls.FROZEN_CONTEXT:
                violations.append(
                    f"Context ceiling violation: config.max_seq_len={cfg.max_seq_len}; "
                    f"expected {cls.FROZEN_CONTEXT}."
                )
            if hasattr(cfg, "bos_token_id") and cfg.bos_token_id != cls.FROZEN_BOS:
                violations.append(f"BOS invariant violation: expected {cls.FROZEN_BOS}.")
            if hasattr(cfg, "eos_token_id") and cfg.eos_token_id != cls.FROZEN_EOS:
                violations.append(f"EOS invariant violation: expected {cls.FROZEN_EOS}.")
            if hasattr(cfg, "pad_token_id") and cfg.pad_token_id != cls.FROZEN_PAD:
                violations.append(f"PAD invariant violation: expected {cls.FROZEN_PAD}.")

        return len(violations) == 0, violations

    @classmethod
    def enforce_model_invariants(cls, model: nn.Module) -> None:
        """Enforce invariants or immediately raise InvariantViolationError."""
        is_valid, violations = cls.verify_model_invariants(model)
        if not is_valid:
            raise InvariantViolationError(
                f"Model failed ChakrMicro frozen invariants: {'; '.join(violations)}"
            )

    @classmethod
    def verify_token_ids(
        cls,
        token_tensor: torch.Tensor,
        min_id: int = 0,
        max_id: int = 4095,
        allowed_ignore_index: Optional[int] = None,
    ) -> None:
        """
        Verify that all token IDs in batch are within the valid vocabulary range.
        """
        if token_tensor.numel() == 0:
            return

        flat = token_tensor.view(-1)
        if allowed_ignore_index is not None:
            flat = flat[flat != allowed_ignore_index]
            if flat.numel() == 0:
                return

        min_val = flat.min().item()
        max_val = flat.max().item()

        if min_val < min_id or max_val > max_id:
            raise TrainingSafetyError(
                f"Token ID range violation: tokens out of bounds [{min_id}, {max_id}]. "
                f"Found min={min_val}, max={max_val}."
            )

    @classmethod
    def check_loss(cls, loss: torch.Tensor | float, step: int = 0) -> float:
        """
        Verify that computed loss is a finite, real number. Fails closed on NaN or Inf.
        """
        loss_val = loss.item() if isinstance(loss, torch.Tensor) else float(loss)

        if math.isnan(loss_val):
            raise NumericalInstabilityError(
                f"Training aborted at step {step}: Loss is NaN (Not a Number)."
            )

        if math.isinf(loss_val):
            raise NumericalInstabilityError(
                f"Training aborted at step {step}: Loss is Infinite ({loss_val})."
            )

        if loss_val > 1000.0:
            raise NumericalInstabilityError(
                f"Training aborted at step {step}: Loss exploded ({loss_val:.2f} > 1000.0)."
            )

        return loss_val

    @classmethod
    def check_gradients(
        cls,
        model: nn.Module,
        step: int = 0,
        max_allowed_norm: float = 500.0,
    ) -> float:
        """
        Verify that model gradients are finite numbers. Fails closed on NaN/Inf.
        Returns total gradient norm.
        """
        total_norm_sq = 0.0
        for name, param in model.named_parameters():
            if param.grad is not None:
                g = param.grad.detach()
                if torch.isnan(g).any():
                    raise NumericalInstabilityError(
                        f"Training aborted at step {step}: Gradient for parameter '{name}' contains NaN."
                    )
                if torch.isinf(g).any():
                    raise NumericalInstabilityError(
                        f"Training aborted at step {step}: Gradient for parameter '{name}' contains Inf."
                    )
                param_norm = torch.norm(g)
                total_norm_sq += (param_norm.item() ** 2)

        total_norm = math.sqrt(total_norm_sq)
        if total_norm > max_allowed_norm:
            raise NumericalInstabilityError(
                f"Training aborted at step {step}: Exploding gradient norm "
                f"({total_norm:.2f} > {max_allowed_norm:.2f})."
            )

        return total_norm

    @classmethod
    def verify_tokenizer_compatibility(cls, expected_fp: str, actual_fp: str) -> None:
        """Verify that dataset and model tokenizer fingerprints match."""
        if expected_fp != actual_fp:
            raise TrainingSafetyError(
                f"Tokenizer incompatibility: dataset built with tokenizer '{expected_fp}', "
                f"but active tokenizer is '{actual_fp}'."
            )

    @classmethod
    def verify_checkpoint_metadata(cls, payload: Dict[str, Any]) -> None:
        """Validate metadata from a saved checkpoint before loading."""
        required_keys = ["step", "model_state_dict", "timestamp"]
        for k in required_keys:
            if k not in payload:
                raise CheckpointCorruptionError(f"Checkpoint corrupted: missing required field '{k}'.")

        if not isinstance(payload["model_state_dict"], dict):
            raise CheckpointCorruptionError("Checkpoint corrupted: model_state_dict is not a dictionary.")

        # Check parameter count in state dict, accounting for tied embeddings (lm_head.weight == embedding.weight)
        unique_tensors = {}
        for k, t in payload["model_state_dict"].items():
            if isinstance(t, torch.Tensor):
                if k == "lm_head.weight" and "embedding.weight" in payload["model_state_dict"]:
                    continue
                unique_tensors[k] = t

        total_ckpt_params = sum(t.numel() for t in unique_tensors.values())
        if total_ckpt_params != cls.FROZEN_PARAMS:
            raise CheckpointCorruptionError(
                f"Checkpoint parameter mismatch: found {total_ckpt_params:,} unique params in checkpoint, "
                f"expected {cls.FROZEN_PARAMS:,}."
            )
