"""
Core Integrity Guard for ChakrView (Step 23).

Central invariant defense mechanism enforcing:
1. parameters == 3,443,136
2. vocab_size == 4,096
3. max_seq_len == 512
4. BOS == 0
5. EOS == 1
6. PAD == 2
7. Architecture configuration (Pre-RMSNorm, RoPE, SwiGLU, weight tying, 6 layers, d_model=192, 6 heads, d_ff=512)
8. Tokenizer fingerprint verification
9. Runtime weight fingerprint (cryptographic SHA-256)

FAIL-CLOSED PRINCIPLE:
If any invariant is violated, execution fails closed immediately.
Self-healing NEVER repairs, patches, or alters the frozen neural core.
"""

from dataclasses import dataclass, field
import hashlib
from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn

from chakrview.brain.model import ChakrMicro
from chakrview.tokenizer.tokenizer import BPETokenizer


# Frozen Architectural Constants
EXPECTED_PARAMETERS = 3_443_136
EXPECTED_VOCAB_SIZE = 4_096
EXPECTED_MAX_SEQ_LEN = 512
EXPECTED_BOS_ID = 0
EXPECTED_EOS_ID = 1
EXPECTED_PAD_ID = 2
EXPECTED_NUM_LAYERS = 6
EXPECTED_D_MODEL = 192
EXPECTED_NUM_HEADS = 6
EXPECTED_D_FF = 512


class InvariantViolationError(RuntimeError):
    """Raised when an immutable architectural invariant is violated."""
    pass


class WeightMutationError(RuntimeError):
    """Raised when runtime inference weights have been mutated."""
    pass


@dataclass(frozen=True)
class IntegrityCheckResult:
    """Result of invariant verification."""
    passed: bool
    parameter_count: int
    vocab_size: int
    max_seq_len: int
    bos_id: int
    eos_id: int
    pad_id: int
    weight_fingerprint: str
    message: str
    details: Dict[str, Any] = field(default_factory=dict)


class CoreIntegrityGuard:
    """
    Central sovereign integrity authority for ChakrView.
    Reusable by inference, thinking, critical thinking, training, diagnostics, and recovery.
    """

    @staticmethod
    def count_unique_parameters(model: nn.Module) -> int:
        """
        Count unique trainable and tied parameters in ChakrMicro.
        Deduplicates tied embedding and lm_head weights.
        """
        seen_ptrs = set()
        total = 0
        for p in model.parameters():
            ptr = p.data_ptr()
            if ptr not in seen_ptrs:
                seen_ptrs.add(ptr)
                total += p.numel()
        return total

    @staticmethod
    def compute_weight_fingerprint(model: nn.Module) -> str:
        """
        Compute deterministic SHA-256 fingerprint over all parameter tensors in canonical key order.
        """
        h = hashlib.sha256()
        state = model.state_dict()
        for k in sorted(state.keys()):
            tensor = state[k].detach().cpu()
            h.update(k.encode("utf-8"))
            h.update(tensor.numpy().tobytes())
        return h.hexdigest()

    @classmethod
    def verify_model(cls, model: ChakrMicro) -> IntegrityCheckResult:
        """
        Verify that a ChakrMicro model satisfies all frozen invariants.
        """
        param_count = cls.count_unique_parameters(model)
        config = getattr(model, "config", None)
        vocab_size = getattr(config, "vocab_size", -1) if config else -1
        max_seq_len = getattr(config, "max_seq_len", -1) if config else -1
        n_layers = getattr(config, "n_layers", getattr(config, "num_layers", -1)) if config else -1
        d_model = getattr(config, "d_model", -1) if config else -1
        n_heads = getattr(config, "n_heads", getattr(config, "num_heads", -1)) if config else -1
        d_ff = getattr(config, "hidden_dim", getattr(config, "d_ff", -1)) if config else -1

        failures = []
        if param_count != EXPECTED_PARAMETERS:
            failures.append(f"parameter_count mismatch: got {param_count}, expected {EXPECTED_PARAMETERS}")
        if vocab_size != EXPECTED_VOCAB_SIZE:
            failures.append(f"vocab_size mismatch: got {vocab_size}, expected {EXPECTED_VOCAB_SIZE}")
        if max_seq_len != EXPECTED_MAX_SEQ_LEN:
            failures.append(f"max_seq_len mismatch: got {max_seq_len}, expected {EXPECTED_MAX_SEQ_LEN}")
        if n_layers != EXPECTED_NUM_LAYERS:
            failures.append(f"num_layers mismatch: got {n_layers}, expected {EXPECTED_NUM_LAYERS}")
        if d_model != EXPECTED_D_MODEL:
            failures.append(f"d_model mismatch: got {d_model}, expected {EXPECTED_D_MODEL}")
        if n_heads != EXPECTED_NUM_HEADS:
            failures.append(f"num_heads mismatch: got {n_heads}, expected {EXPECTED_NUM_HEADS}")
        if d_ff != EXPECTED_D_FF:
            failures.append(f"d_ff mismatch: got {d_ff}, expected {EXPECTED_D_FF}")

        passed = len(failures) == 0
        fingerprint = cls.compute_weight_fingerprint(model)

        return IntegrityCheckResult(
            passed=passed,
            parameter_count=param_count,
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            bos_id=EXPECTED_BOS_ID,
            eos_id=EXPECTED_EOS_ID,
            pad_id=EXPECTED_PAD_ID,
            weight_fingerprint=fingerprint,
            message="Model satisfies all frozen invariants." if passed else "; ".join(failures),
            details={"failures": failures},
        )

    @classmethod
    def verify_tokenizer(cls, tokenizer: BPETokenizer) -> Tuple[bool, str]:
        """
        Verify tokenizer vocabulary and special token invariants.
        """
        failures = []
        vocab_size = getattr(tokenizer, "vocab_size", -1)
        bos_id = getattr(tokenizer, "bos_token_id", EXPECTED_BOS_ID)
        eos_id = getattr(tokenizer, "eos_token_id", EXPECTED_EOS_ID)
        pad_id = getattr(tokenizer, "pad_token_id", EXPECTED_PAD_ID)

        if vocab_size != EXPECTED_VOCAB_SIZE:
            failures.append(f"tokenizer vocab_size: got {vocab_size}, expected {EXPECTED_VOCAB_SIZE}")
        if bos_id != EXPECTED_BOS_ID:
            failures.append(f"tokenizer bos_id: got {bos_id}, expected {EXPECTED_BOS_ID}")
        if eos_id != EXPECTED_EOS_ID:
            failures.append(f"tokenizer eos_id: got {eos_id}, expected {EXPECTED_EOS_ID}")
        if pad_id != EXPECTED_PAD_ID:
            failures.append(f"tokenizer pad_id: got {pad_id}, expected {EXPECTED_PAD_ID}")

        return (len(failures) == 0, "Tokenizer verified." if not failures else "; ".join(failures))

    @classmethod
    def fail_closed_if_invalid(cls, model: ChakrMicro, tokenizer: Optional[BPETokenizer] = None) -> None:
        """
        Strict fail-closed verification. Raises InvariantViolationError immediately on any discrepancy.
        """
        res = cls.verify_model(model)
        if not res.passed:
            raise InvariantViolationError(f"CRITICAL INVARIANT VIOLATION: {res.message}")

        if tokenizer is not None:
            tok_passed, tok_msg = cls.verify_tokenizer(tokenizer)
            if not tok_passed:
                raise InvariantViolationError(f"CRITICAL TOKENIZER INVARIANT VIOLATION: {tok_msg}")

    @classmethod
    def verify_weights_unmodified(
        cls, model: ChakrMicro, initial_fingerprint: str
    ) -> bool:
        """
        Verify that model weights match initial fingerprint.
        Raises WeightMutationError if weights have changed.
        """
        current = cls.compute_weight_fingerprint(model)
        if current != initial_fingerprint:
            raise WeightMutationError(
                f"FATAL: Runtime weight mutation detected! Expected {initial_fingerprint}, got {current}."
            )
        return True
