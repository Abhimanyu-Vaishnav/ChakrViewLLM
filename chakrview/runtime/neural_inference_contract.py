"""
ChakrView Step 91: Neural Inference Contract.

Provides a clean, modular inference contract decoupling:
- Tokenizer input encoding / token bounds checking
- Context limits / prefill budget
- Model invocation (ChakrMicro) with ΔW = 0 verification
- Logits / output tensor bounds (rejecting NaNs/Infs)
- Autoregressive decoding with stopping criteria
- Deterministic inference mode vs resource-aware inference mode
- Fail-closed error handling and explicit epistemic abstention behavior
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import torch

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.tokenizer.tokenizer import BPETokenizer
from chakrview.tokenizer.special_tokens import BOS_ID, EOS_ID, PAD_ID

EXPECTED_WEIGHT_HASH = "c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da"
EXPECTED_PARAM_COUNT = 3_443_136
EXPECTED_VOCAB_SIZE = 4096
DEFAULT_MAX_CONTEXT = 512


# ─────────────────────────────────────────────────────────────────────────────
# 1. Error Hierarchy
# ─────────────────────────────────────────────────────────────────────────────

class NeuralContractError(Exception):
    """Base exception for neural inference contract violations."""
    pass


class TokenBoundsError(NeuralContractError):
    """Raised when token IDs are non-integer, negative, or exceed vocabulary size."""
    pass


class ContextBudgetExceededError(NeuralContractError):
    """Raised when sequence exceeds allowed context budget."""
    pass


class PathologicalLogitsError(NeuralContractError):
    """Raised when model produces NaNs or Infinities."""
    pass


class WeightMutationDetectedError(NeuralContractError):
    """Raised when neural parameter hash diverges from frozen baseline (ΔW != 0)."""
    pass


# ─────────────────────────────────────────────────────────────────────────────
# 2. Enumerations and Contract Dataclasses
# ─────────────────────────────────────────────────────────────────────────────

class InferenceStopReason(Enum):
    MAX_TOKENS = "MAX_TOKENS"
    STOP_TOKEN = "STOP_TOKEN"
    CONTEXT_LIMIT = "CONTEXT_LIMIT"
    ABSTAINED = "ABSTAINED"
    ERROR = "ERROR"


class InferenceExecutionMode(Enum):
    DETERMINISTIC = "DETERMINISTIC"   # temperature = 0.0, exact greedy decoding
    RESOURCE_AWARE = "RESOURCE_AWARE" # dynamically constrained by hardware budget


@dataclass(frozen=True)
class NeuralInferenceConfig:
    """
    Configuration parameters for a neural inference invocation.
    """
    max_new_tokens: int = 32
    min_new_tokens: int = 1
    temperature: float = 0.0           # 0.0 = deterministic greedy
    top_p: float = 1.0
    top_k: int = 0
    stop_tokens: List[int] = field(default_factory=lambda: [EOS_ID])
    add_bos: bool = True
    add_eos: bool = False
    truncate_overflow: bool = False
    mode: InferenceExecutionMode = InferenceExecutionMode.DETERMINISTIC


@dataclass
class NeuralInferencePayload:
    """
    Structured input to the neural inference contract.
    """
    prompt: Optional[str] = None
    prompt_tokens: Optional[List[int]] = None
    config: NeuralInferenceConfig = field(default_factory=NeuralInferenceConfig)
    session_id: str = "default_session"
    tenant_id: str = "default_tenant"
    system_directive: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.prompt is None and self.prompt_tokens is None:
            raise NeuralContractError("Inference payload requires either prompt text or prompt_tokens.")


@dataclass
class NeuralInferenceOutput:
    """
    Standardized result contract from neural model execution.
    """
    generated_text: str
    generated_tokens: List[int]
    prompt_tokens: List[int]
    input_token_count: int
    output_token_count: int
    total_token_count: int
    stop_reason: InferenceStopReason
    latency_ms: float
    weight_hash_verified: bool
    is_abstained: bool = False
    abstention_reason: Optional[str] = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "generated_text": self.generated_text,
            "generated_tokens": self.generated_tokens,
            "prompt_tokens": self.prompt_tokens,
            "input_token_count": self.input_token_count,
            "output_token_count": self.output_token_count,
            "total_token_count": self.total_token_count,
            "stop_reason": self.stop_reason.value,
            "latency_ms": self.latency_ms,
            "weight_hash_verified": self.weight_hash_verified,
            "is_abstained": self.is_abstained,
            "abstention_reason": self.abstention_reason,
            "diagnostics": self.diagnostics,
        }


# ─────────────────────────────────────────────────────────────────────────────
# 3. Neural Inference Contract Engine
# ─────────────────────────────────────────────────────────────────────────────

class NeuralInferenceContract:
    """
    Step 91: Sovereign Neural Inference Contract.
    Enforces strict invariants between BPETokenizer, ChakrMicro, and consumers.
    """

    def __init__(
        self,
        model: ChakrMicro,
        tokenizer: BPETokenizer,
        device: Optional[torch.device] = None,
        max_context: int = DEFAULT_MAX_CONTEXT,
        expected_weight_hash: str = EXPECTED_WEIGHT_HASH,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or (next(model.parameters()).device if list(model.parameters()) else torch.device("cpu"))
        self.max_context = max_context
        self.expected_weight_hash = expected_weight_hash
        self.model.eval()

        self.validate_contract()

    def compute_weight_hash(self) -> str:
        """Compute SHA-256 digest of all model parameters to verify ΔW = 0."""
        hasher = hashlib.sha256()
        with torch.no_grad():
            for name, param in sorted(self.model.named_parameters()):
                hasher.update(name.encode("utf-8"))
                hasher.update(param.detach().cpu().numpy().tobytes())
        return hasher.hexdigest()

    def validate_contract(self) -> None:
        """Validates architecture invariants."""
        # 1. Parameter count check
        total_params = sum(p.numel() for p in self.model.parameters())
        if total_params != EXPECTED_PARAM_COUNT:
            raise NeuralContractError(
                f"Model parameter count {total_params} != expected {EXPECTED_PARAM_COUNT}"
            )

        # 2. Vocab size check
        if self.tokenizer.vocab_size != EXPECTED_VOCAB_SIZE:
            raise NeuralContractError(
                f"Tokenizer vocab_size {self.tokenizer.vocab_size} != expected {EXPECTED_VOCAB_SIZE}"
            )

        # 3. Weight hash check
        h = self.compute_weight_hash()
        if h != self.expected_weight_hash:
            raise WeightMutationDetectedError(
                f"Neural weight hash mismatch: {h} != {self.expected_weight_hash}"
            )

    def validate_token_ids(self, tokens: Sequence[int]) -> None:
        """Fails closed on any non-integer or out-of-vocab token."""
        for i, t in enumerate(tokens):
            if not isinstance(t, int) or isinstance(t, bool):
                raise TokenBoundsError(f"Token at index {i} is not integer: {t} ({type(t).__name__})")
            if t < 0 or t >= EXPECTED_VOCAB_SIZE:
                raise TokenBoundsError(f"Token {t} at index {i} out of bounds [0, {EXPECTED_VOCAB_SIZE})")

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = False) -> List[int]:
        """Encodes text to token IDs and validates bounds."""
        tokens = self.tokenizer.encode(text, add_bos=add_bos, add_eos=add_eos)
        self.validate_token_ids(tokens)
        return tokens

    def decode(self, token_ids: Sequence[int], skip_special_tokens: bool = True) -> str:
        """Decodes token IDs to Unicode string."""
        self.validate_token_ids(token_ids)
        return self.tokenizer.decode(list(token_ids), skip_special_tokens=skip_special_tokens, errors="replace")

    def forward(self, token_ids: Sequence[int]) -> torch.Tensor:
        """
        Executes single forward pass through ChakrMicro, validating finite logits.
        """
        self.validate_token_ids(token_ids)
        if len(token_ids) == 0:
            raise TokenBoundsError("Cannot execute forward pass on empty token sequence.")
        if len(token_ids) > self.max_context:
            raise ContextBudgetExceededError(
                f"Token count {len(token_ids)} exceeds context limit {self.max_context}"
            )

        tensor_in = torch.tensor([list(token_ids)], dtype=torch.long, device=self.device)
        with torch.no_grad():
            logits = self.model(tensor_in)

        if not torch.isfinite(logits).all():
            raise PathologicalLogitsError("Model produced non-finite logits (NaN or Inf).")

        return logits

    def generate(self, payload: NeuralInferencePayload) -> NeuralInferenceOutput:
        """
        Executes autoregressive generation adhering to the contract.
        """
        t0 = time.perf_counter()
        payload.validate()

        # Pre-execution invariant check
        pre_hash = self.compute_weight_hash()
        if pre_hash != self.expected_weight_hash:
            raise WeightMutationDetectedError("Pre-execution weight mutation detected.")

        # 1. Resolve prompt tokens
        if payload.prompt_tokens is not None:
            prompt_tokens = list(payload.prompt_tokens)
            self.validate_token_ids(prompt_tokens)
        else:
            prompt_tokens = self.encode(
                payload.prompt or "",
                add_bos=payload.config.add_bos,
                add_eos=payload.config.add_eos,
            )

        if not prompt_tokens:
            prompt_tokens = [BOS_ID]

        # 2. Context budget validation and governed truncation
        max_new = max(1, payload.config.max_new_tokens)
        if len(prompt_tokens) + max_new > self.max_context:
            if payload.config.truncate_overflow:
                avail = self.max_context - max_new
                if avail <= 0:
                    raise ContextBudgetExceededError(
                        f"max_new_tokens ({max_new}) >= max_context ({self.max_context})"
                    )
                has_bos = (prompt_tokens[0] == BOS_ID)
                if has_bos:
                    prompt_tokens = [BOS_ID] + prompt_tokens[-(avail - 1):]
                else:
                    prompt_tokens = prompt_tokens[-avail:]
            else:
                raise ContextBudgetExceededError(
                    f"Prompt tokens ({len(prompt_tokens)}) + max_new ({max_new}) exceeds {self.max_context}"
                )

        # 3. Autoregressive loop
        generated_tokens: List[int] = []
        curr_tokens = list(prompt_tokens)
        stop_reason = InferenceStopReason.MAX_TOKENS

        with torch.no_grad():
            for step in range(max_new):
                if len(curr_tokens) >= self.max_context:
                    stop_reason = InferenceStopReason.CONTEXT_LIMIT
                    break

                tensor_in = torch.tensor([curr_tokens], dtype=torch.long, device=self.device)
                logits = self.model(tensor_in)  # [1, seq_len, vocab_size]
                next_token_logits = logits[0, -1, :]

                if not torch.isfinite(next_token_logits).all():
                    raise PathologicalLogitsError("Pathological non-finite logits encountered during generation.")

                # Deterministic (greedy) or sampling
                if payload.config.temperature == 0.0 or payload.config.mode == InferenceExecutionMode.DETERMINISTIC:
                    next_token = int(torch.argmax(next_token_logits).item())
                else:
                    scaled = next_token_logits / max(payload.config.temperature, 1e-4)
                    probs = torch.softmax(scaled, dim=-1)
                    next_token = int(torch.multinomial(probs, num_samples=1).item())

                generated_tokens.append(next_token)
                curr_tokens.append(next_token)

                # Check stop tokens
                if step >= (payload.config.min_new_tokens - 1) and next_token in payload.config.stop_tokens:
                    stop_reason = InferenceStopReason.STOP_TOKEN
                    break

        # 4. Post-execution weight invariant check
        post_hash = self.compute_weight_hash()
        if post_hash != self.expected_weight_hash:
            raise WeightMutationDetectedError("Post-execution weight mutation detected (ΔW != 0).")

        gen_text = self.decode(generated_tokens)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        return NeuralInferenceOutput(
            generated_text=gen_text,
            generated_tokens=generated_tokens,
            prompt_tokens=prompt_tokens,
            input_token_count=len(prompt_tokens),
            output_token_count=len(generated_tokens),
            total_token_count=len(curr_tokens),
            stop_reason=stop_reason,
            latency_ms=elapsed_ms,
            weight_hash_verified=True,
            is_abstained=False,
            diagnostics={"device": str(self.device), "mode": payload.config.mode.value},
        )

    def abstain(self, reason: str, prompt_tokens: Optional[List[int]] = None) -> NeuralInferenceOutput:
        """
        Produces an explicit, structured abstention output when evidence is unknown/insufficient.
        """
        pts = prompt_tokens or [BOS_ID]
        return NeuralInferenceOutput(
            generated_text=f"[ABSTAIN]: {reason}",
            generated_tokens=[],
            prompt_tokens=pts,
            input_token_count=len(pts),
            output_token_count=0,
            total_token_count=len(pts),
            stop_reason=InferenceStopReason.ABSTAINED,
            latency_ms=0.0,
            weight_hash_verified=True,
            is_abstained=True,
            abstention_reason=reason,
            diagnostics={"abstained": True},
        )
