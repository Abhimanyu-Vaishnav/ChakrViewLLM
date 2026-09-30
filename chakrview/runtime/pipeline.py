"""
ChakrView End-to-End Neural Inference Pipeline (Step 44).

Unifies Chakr-BPE Tokenizer, Context Assembly, and frozen ChakrMicro Neural Core
into a deterministic, security-governed inference subsystem.

Core Invariants Enforced:
1. ΔW = 0: Model weights strictly immutable across all forward/decode passes.
2. Tokenizer <-> Model Contract: vocab_size == 4096, BOS=0, EOS=1, PAD=2.
3. Token Bounds: All token IDs must be integers in [0, 4096).
4. Sequence Horizon: len(prompt) + max_new_tokens <= 512.
5. Finite Logits: Rejects NaNs and Infinities.
6. Trust Boundaries: External RAG evidence demarcated as passive untrusted data.
7. Zero Secret Exposure: Secret scanning on all prompt and context inputs.
8. Tenant Isolation: Enforces matching tenant_id on all context envelopes.
"""

from dataclasses import dataclass, field, asdict
import hashlib
from pathlib import Path
import re
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import torch

from chakrview.brain.cache import KVCache
from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.cognition.federation.cognitive.models import (
    CognitiveContextEnvelope,
    CognitiveContextOverflowError,
    CognitiveContextTenantViolationError,
    PROHIBITED_CONTEXT_KEYWORDS,
    SecretLeakageInContextError,
)
from chakrview.runtime.hardware import ModelExecutionPlan
from chakrview.runtime.inference import GenerationConfig, StopReason
from chakrview.runtime.sampling import Sampler, SamplingConfig, SamplingProbabilityError
from chakrview.tokenizer.serialization import load_tokenizer_artifacts
from chakrview.tokenizer.special_tokens import BOS_ID, EOS_ID, PAD_ID
from chakrview.tokenizer.tokenizer import BPETokenizer

EXPECTED_WEIGHT_HASH = "c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da"
EXPECTED_VOCAB_SIZE = 4096
MAX_CONTEXT_WINDOW = 512


# ─────────────────────────────────────────────────────────────────────────────
# 1. Pipeline Exception Hierarchy
# ─────────────────────────────────────────────────────────────────────────────

class InferencePipelineError(Exception):
    """Base exception for all neural inference pipeline failures."""
    pass


class TokenizerModelMismatchError(InferencePipelineError):
    """Raised when tokenizer vocabulary does not strictly match model configuration."""
    pass


class MalformedTokenIdError(InferencePipelineError):
    """Raised when token IDs are non-integer, negative, or exceed vocabulary size."""
    pass


class ContextOverflowError(InferencePipelineError):
    """Raised when prompt + generation tokens exceed model context ceiling (512)."""
    pass


class PathologicalLogitsError(InferencePipelineError):
    """Raised when model logits contain NaN or Infinite values."""
    pass


class InvalidGenerationConfigError(InferencePipelineError):
    """Raised when generation parameters are invalid or contradictory."""
    pass


class NeuralWeightMutationError(InferencePipelineError):
    """Raised if model parameter SHA-256 digest mutates during inference (ΔW != 0)."""
    pass


class IncompatibleCheckpointError(InferencePipelineError):
    """Raised when a checkpoint's structure or weights are incompatible with ChakrMicro."""
    pass


# ─────────────────────────────────────────────────────────────────────────────
# 2. Pipeline Request, Context & Result Contracts
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ModelIdentity:
    """
    Cryptographic and architectural identity specification for ChakrMicro (Step 45).
    """
    architecture_name: str = "ChakrMicro"
    version: str = "0.1.0"
    parameter_count: int = 3_443_136
    vocab_size: int = 4096
    max_context_len: int = 512
    weight_hash: str = EXPECTED_WEIGHT_HASH
    tokenizer_checksum: str = "7498d92adeef7c6db98d89a444a7f0e303dd5e7ea4b679a95781a95e6347c617"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class InferenceRequest:
    """
    Standardized request contract for the neural inference pipeline.

    Attributes:
        prompt: Raw user query or instructions text.
        prompt_tokens: Optional explicit list of pre-tokenized integer token IDs.
        context_envelope: Optional Step 42/43 CognitiveContextEnvelope.
        generation_config: Optional GenerationConfig (defaults to greedy decoding).
        tenant_id: Tenant partition identifier.
        session_id: Session identifier.
        add_bos: Prepend BOS token (0) to prompt.
        add_eos: Append EOS token (1) to prompt.
        truncate_if_overflow: If True, explicitly truncate prompt from left to fit budget.
                              If False, fail closed with ContextOverflowError.
    """
    prompt: Optional[str] = None
    prompt_tokens: Optional[List[int]] = None
    context_envelope: Optional[CognitiveContextEnvelope] = None
    generation_config: Optional[GenerationConfig] = None
    tenant_id: str = "default_tenant"
    session_id: str = "default_session"
    add_bos: bool = True
    add_eos: bool = False
    truncate_if_overflow: bool = False

    def __post_init__(self) -> None:
        if self.prompt is None and self.prompt_tokens is None and self.context_envelope is None:
            raise InferencePipelineError("InferenceRequest requires at least prompt, prompt_tokens, or context_envelope.")


@dataclass
class InferenceContext:
    """
    Assembled context ready for tokenizer encoding or model forward pass.
    """
    full_prompt_text: str
    token_ids: List[int]
    tenant_id: str
    session_id: str
    trust_levels: Dict[str, str] = field(default_factory=dict)
    sources: List[Dict[str, Any]] = field(default_factory=list)
    memory_citations: List[str] = field(default_factory=list)


@dataclass
class InferenceResult:
    """
    Complete outcome package from neural inference execution.
    """
    text: str
    token_ids: List[int]
    prompt_tokens: List[int]
    input_token_count: int
    output_token_count: int
    total_token_count: int
    generation_config: GenerationConfig
    stop_reason: StopReason
    latency_ms: float
    model_identity: ModelIdentity = field(default_factory=ModelIdentity)
    reproducibility: Dict[str, Any] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)
    weight_hash_verified: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "token_ids": self.token_ids,
            "prompt_tokens": self.prompt_tokens,
            "input_token_count": self.input_token_count,
            "output_token_count": self.output_token_count,
            "total_token_count": self.total_token_count,
            "generation_config": self.generation_config.to_dict(),
            "stop_reason": self.stop_reason.value,
            "latency_ms": self.latency_ms,
            "model_identity": self.model_identity.to_dict(),
            "reproducibility": self.reproducibility,
            "provenance": self.provenance,
            "weight_hash_verified": self.weight_hash_verified,
        }


# ─────────────────────────────────────────────────────────────────────────────
# 3. Context Builder with Trust & Secret Boundaries
# ─────────────────────────────────────────────────────────────────────────────

class InferenceContextBuilder:
    """
    Builds bounded, sanitized prompt context respecting the 4-tier trust hierarchy:
    LOCAL_VERIFIED_MEMORY > FEDERATED_VERIFIED_MEMORY > RETRIEVED_EXTERNAL_KNOWLEDGE > UNTRUSTED_INPUT
    """

    @staticmethod
    def scan_for_secrets(text: str) -> None:
        """Scan text for prohibited credential patterns; fail closed on match."""
        lower_text = text.lower()
        for keyword in PROHIBITED_CONTEXT_KEYWORDS:
            if keyword in lower_text:
                raise SecretLeakageInContextError(
                    f"Prohibited credential keyword '{keyword}' detected in inference input."
                )

    @classmethod
    def build_context(
        cls,
        request: InferenceRequest,
        tokenizer: BPETokenizer,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Assemble text, enforcing tenant isolation, secret scanning, and evidence demarcation.
        """
        sources: List[Dict[str, Any]] = []
        memory_citations: List[str] = []
        trust_levels: Dict[str, str] = {}
        sections: List[str] = []

        # 1. Inspect Context Envelope if provided
        if request.context_envelope is not None:
            # Enforce tenant isolation
            request.context_envelope.validate_for_tenant(request.tenant_id)
            # Enforce envelope invariants
            request.context_envelope.validate()

            # Local / Federated Verified Memory items
            if request.context_envelope.context_items:
                sections.append("--- VERIFIED COGNITIVE MEMORY ---")
                for item in request.context_envelope.context_items:
                    cls.scan_for_secrets(item)
                    sections.append(item)
                    memory_citations.append(item)
                sections.append("--- END VERIFIED COGNITIVE MEMORY ---\n")
                trust_levels["memory"] = "LOCAL_VERIFIED_MEMORY"

            # External Evidence items (Demarcated as passive data)
            if request.context_envelope.evidence_items:
                sections.append("--- RETRIEVED EXTERNAL EVIDENCE (UNTRUSTED PASSIVE DATA) ---")
                for ev in request.context_envelope.evidence_items:
                    cls.scan_for_secrets(ev)
                    sections.append(ev)
                    sources.append({"content": ev, "trust_level": "RETRIEVED_EXTERNAL"})
                sections.append("--- END RETRIEVED EXTERNAL EVIDENCE ---\n")
                trust_levels["evidence"] = "RETRIEVED_EXTERNAL_KNOWLEDGE"

        # 2. Append User Prompt / Instructions
        if request.prompt is not None:
            cls.scan_for_secrets(request.prompt)
            sections.append(request.prompt.strip())
            trust_levels["prompt"] = "UNTRUSTED_INPUT"

        full_text = "\n".join(sections).strip()
        cls.scan_for_secrets(full_text)

        metadata = {
            "sources": sources,
            "memory_citations": memory_citations,
            "trust_levels": trust_levels,
        }
        return full_text, metadata


# ─────────────────────────────────────────────────────────────────────────────
# 4. End-to-End Neural Inference Engine
# ─────────────────────────────────────────────────────────────────────────────

class InferenceEngine:
    """
    Indigenous End-to-End Neural Inference Engine for ChakrView.

    Coordinates BPETokenizer, InferenceContextBuilder, frozen ChakrMicro neural core,
    KV caching, logits validation, and deterministic autoregressive generation.
    """

    def __init__(
        self,
        model: Optional[ChakrMicro] = None,
        tokenizer: Optional[BPETokenizer] = None,
        execution_plan: Optional[ModelExecutionPlan] = None,
        sampler: Optional[Sampler] = None,
        expected_weight_hash: str = EXPECTED_WEIGHT_HASH,
    ) -> None:
        if model is None:
            torch.manual_seed(42)
            self.model = ChakrMicro(ModelConfig())
        else:
            self.model = model
        self.model.eval()

        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            root_dir = Path(__file__).resolve().parents[2]
            tok_dir = root_dir / "data" / "experiments" / "vocab_4096"
            if not tok_dir.exists():
                tok_dir = root_dir / "data" / "tokenizer_experiments" / "v4096"
            self.tokenizer, _ = load_tokenizer_artifacts(tok_dir)

        self.execution_plan = execution_plan
        self.sampler = sampler or Sampler()
        self.expected_weight_hash = expected_weight_hash

        self.device = torch.device(execution_plan.device) if execution_plan else torch.device("cpu")
        self.max_context = (
            execution_plan.max_context_len if execution_plan else self.model.config.max_seq_len
        )

        # Validate contract on initialization
        self.validate_contract()

        # Model Identity Contract (Step 45)
        self.model_identity = ModelIdentity(
            parameter_count=self.model.count_parameters()["total_parameters"],
            max_context_len=self.max_context,
            weight_hash=self.expected_weight_hash,
        )

        # Initialize KV Cache
        self.kv_cache = KVCache(
            num_layers=self.model.config.n_layers,
            max_seq_len=self.max_context,
            device=self.device,
            dtype=torch.float32,
        )

    def compute_weight_hash(self) -> str:
        """Compute SHA-256 digest of all named parameters for ΔW = 0 verification."""
        hasher = hashlib.sha256()
        with torch.no_grad():
            for name, param in sorted(self.model.named_parameters()):
                hasher.update(name.encode("utf-8"))
                hasher.update(param.detach().cpu().numpy().tobytes())
        return hasher.hexdigest()

    def validate_contract(self) -> None:
        """
        Validate strict Tokenizer <-> Model contract:
        1. Model vocab_size == 4096
        2. Tokenizer vocab_size == 4096
        3. Pre-flight weight hash == EXPECTED_WEIGHT_HASH
        """
        if self.model.config.vocab_size != EXPECTED_VOCAB_SIZE:
            raise TokenizerModelMismatchError(
                f"Model config vocab_size {self.model.config.vocab_size} != expected {EXPECTED_VOCAB_SIZE}"
            )
        if self.tokenizer.vocab_size != EXPECTED_VOCAB_SIZE:
            raise TokenizerModelMismatchError(
                f"Tokenizer vocab_size {self.tokenizer.vocab_size} != expected {EXPECTED_VOCAB_SIZE}"
            )

        cur_hash = self.compute_weight_hash()
        if cur_hash != self.expected_weight_hash:
            raise NeuralWeightMutationError(
                f"Pre-flight neural weight hash mismatch: {cur_hash[:16]}… != {self.expected_weight_hash[:16]}…"
            )

    def validate_token_ids(self, tokens: List[int]) -> None:
        """
        Assert that all token IDs are integers in [0, vocab_size).
        Fails closed on any malformed or out-of-bounds token.
        """
        for i, t in enumerate(tokens):
            if not isinstance(t, int) or isinstance(t, bool):
                raise MalformedTokenIdError(f"Token at index {i} is not an integer: {type(t).__name__} ({t})")
            if t < 0 or t >= EXPECTED_VOCAB_SIZE:
                raise MalformedTokenIdError(
                    f"Token ID {t} at index {i} out of bounds [0, {EXPECTED_VOCAB_SIZE})"
                )

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = False) -> List[int]:
        """Encode text to token IDs and validate bounds."""
        tokens = self.tokenizer.encode(text, add_bos=add_bos, add_eos=add_eos)
        self.validate_token_ids(tokens)
        return tokens

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs back to exact Unicode text."""
        self.validate_token_ids(token_ids)
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens, errors="replace")

    def forward(self, token_ids: List[int]) -> torch.Tensor:
        """
        Execute single-batch forward pass through ChakrMicro with validation.

        Args:
            token_ids: List of integer token IDs.

        Returns:
            Logits tensor of shape [1, len(token_ids), 4096].
        """
        self.validate_token_ids(token_ids)
        if len(token_ids) == 0:
            raise MalformedTokenIdError("Cannot execute forward pass on empty token sequence.")
        if len(token_ids) > self.max_context:
            raise ContextOverflowError(
                f"Sequence length {len(token_ids)} exceeds maximum context {self.max_context}"
            )

        pre_hash = self.compute_weight_hash()
        if pre_hash != self.expected_weight_hash:
            raise NeuralWeightMutationError("Pre-forward weight mutation detected.")

        tensor_in = torch.tensor([token_ids], dtype=torch.long, device=self.device)
        with torch.no_grad():
            logits = self.model(tensor_in)

        # Validate logits
        if not torch.isfinite(logits).all():
            raise PathologicalLogitsError("Model emitted non-finite logits (NaN or Inf).")
        if logits.shape != (1, len(token_ids), EXPECTED_VOCAB_SIZE):
            raise InferencePipelineError(
                f"Unexpected logits shape: {logits.shape}, expected (1, {len(token_ids)}, {EXPECTED_VOCAB_SIZE})"
            )

        post_hash = self.compute_weight_hash()
        if post_hash != self.expected_weight_hash:
            raise NeuralWeightMutationError("Post-forward weight mutation detected (ΔW != 0).")

        return logits

    def execute(self, request: InferenceRequest) -> InferenceResult:
        """
        Execute full end-to-end inference request:
        1. Context assembly & secret scanning
        2. Tokenizer encoding & bounds validation
        3. Token budget calculation & overflow enforcement
        4. Model prefill & autoregressive generation with KV caching
        5. Logits validation & deterministic sampling
        6. Lossless text decoding & ΔW = 0 verification
        """
        t_start = time.perf_counter()

        # 1. Pre-flight verification
        pre_hash = self.compute_weight_hash()
        if pre_hash != self.expected_weight_hash:
            raise NeuralWeightMutationError("Pre-flight weight mutation detected.")

        # 2. Context Assembly & Tokenization
        provenance: Dict[str, Any] = {}
        if request.prompt_tokens is not None:
            # Explicit token IDs provided
            prompt_tokens = list(request.prompt_tokens)
            self.validate_token_ids(prompt_tokens)
        else:
            # Assemble text from prompt and context envelope
            full_text, provenance = InferenceContextBuilder.build_context(request, self.tokenizer)
            if not full_text:
                prompt_tokens = [BOS_ID] if request.add_bos else []
            else:
                prompt_tokens = self.encode(
                    full_text,
                    add_bos=request.add_bos,
                    add_eos=request.add_eos,
                )

        if not prompt_tokens:
            prompt_tokens = [BOS_ID]

        # 3. Generation configuration & Token budget
        config = request.generation_config or GenerationConfig(
            max_new_tokens=32,
            sampling=SamplingConfig(temperature=0.0),  # Greedy baseline
        )
        max_new = config.max_new_tokens
        if max_new <= 0:
            raise InvalidGenerationConfigError(f"max_new_tokens must be positive, got {max_new}")

        total_requested = len(prompt_tokens) + max_new
        if total_requested > self.max_context:
            if request.truncate_if_overflow:
                # Governed truncation: keep BOS if present, truncate leftmost tokens
                available_for_prompt = self.max_context - max_new
                if available_for_prompt <= 0:
                    raise ContextOverflowError(
                        f"max_new_tokens ({max_new}) >= max_context ({self.max_context}); cannot fit prompt."
                    )
                has_bos = (prompt_tokens[0] == BOS_ID)
                if has_bos:
                    prompt_tokens = [BOS_ID] + prompt_tokens[-(available_for_prompt - 1):]
                else:
                    prompt_tokens = prompt_tokens[-available_for_prompt:]
            else:
                raise ContextOverflowError(
                    f"Prompt tokens ({len(prompt_tokens)}) + max_new_tokens ({max_new}) = "
                    f"{total_requested} exceeds context ceiling {self.max_context}. "
                    f"Set truncate_if_overflow=True to enable governed prompt truncation."
                )

        # 4. Reset KV cache and execute Prefill
        self.kv_cache.reset()
        prompt_tensor = torch.tensor([prompt_tokens], dtype=torch.long, device=self.device)

        with torch.no_grad():
            prefill_logits, _ = self.model.prefill(prompt_tensor, kv_cache=self.kv_cache)
            if not torch.isfinite(prefill_logits).all():
                raise PathologicalLogitsError("Non-finite logits detected during prompt prefill.")
            latest_logits = prefill_logits[0, -1, :]

        # 5. Autoregressive Decoding Loop
        generated_tokens: List[int] = []
        stop_reason = StopReason.MAX_TOKENS
        stop_tokens = set(config.stop_token_ids) if config.stop_token_ids else {EOS_ID}
        min_new = getattr(config, "min_new_tokens", 0)

        for step in range(max_new):
            # Check context limits
            if self.kv_cache.sequence_length >= self.max_context:
                stop_reason = StopReason.CONTEXT_LIMIT
                break

            # Sample next token with signature compatibility
            try:
                next_token_id = self.sampler.sample(
                    logits=latest_logits,
                    generated_tokens=generated_tokens,
                    config=config.sampling,
                    step=step,
                    strict_safety=True,
                )
            except TypeError:
                next_token_id = self.sampler.sample(
                    logits=latest_logits,
                    generated_tokens=generated_tokens,
                    config=config.sampling,
                    step=step,
                )

            # Check stop condition
            if next_token_id in stop_tokens:
                if step < min_new:
                    # Enforce min_new_tokens: mask stop tokens and resample
                    masked_logits = latest_logits.clone()
                    for st in stop_tokens:
                        masked_logits[st] = -1e9
                    try:
                        next_token_id = self.sampler.sample(
                            logits=masked_logits,
                            generated_tokens=generated_tokens,
                            config=config.sampling,
                            step=step,
                            strict_safety=True,
                        )
                    except TypeError:
                        next_token_id = self.sampler.sample(
                            logits=masked_logits,
                            generated_tokens=generated_tokens,
                            config=config.sampling,
                            step=step,
                        )
                    self.validate_token_ids([next_token_id])
                    generated_tokens.append(next_token_id)
                else:
                    self.validate_token_ids([next_token_id])
                    generated_tokens.append(next_token_id)
                    stop_reason = StopReason.EOS
                    break
            else:
                self.validate_token_ids([next_token_id])
                generated_tokens.append(next_token_id)

            # Single-token decode next step
            next_tensor = torch.tensor([[next_token_id]], dtype=torch.long, device=self.device)
            with torch.no_grad():
                step_logits = self.model.decode_next(next_tensor, kv_cache=self.kv_cache)
                if not torch.isfinite(step_logits).all():
                    raise PathologicalLogitsError("Non-finite logits detected during incremental decode.")
                latest_logits = step_logits[0, -1, :]

        # 6. Post-flight immutability verification
        post_hash = self.compute_weight_hash()
        if post_hash != self.expected_weight_hash:
            raise NeuralWeightMutationError("Post-inference weight mutation detected (ΔW != 0).")

        t_end = time.perf_counter()
        latency_ms = (t_end - t_start) * 1000.0

        # 7. Decode text output
        generated_text = self.decode(generated_tokens, skip_special_tokens=True)

        model_identity = ModelIdentity(
            architecture_name="ChakrMicro",
            version="0.1.0",
            parameter_count=sum(p.numel() for p in self.model.parameters()),
            vocab_size=self.model.config.vocab_size,
            max_context_len=self.max_context,
            weight_hash=post_hash,
        )

        reproducibility = {
            "seed": config.sampling.seed,
            "is_greedy": config.sampling.is_greedy,
            "strategy": config.sampling.strategy.value,
            "temperature": config.sampling.temperature,
            "top_k": config.sampling.top_k,
            "top_p": config.sampling.top_p,
            "repetition_penalty": config.sampling.repetition_penalty,
            "min_new_tokens": min_new,
            "max_new_tokens": max_new,
        }

        return InferenceResult(
            text=generated_text,
            token_ids=generated_tokens,
            prompt_tokens=prompt_tokens,
            input_token_count=len(prompt_tokens),
            output_token_count=len(generated_tokens),
            total_token_count=len(prompt_tokens) + len(generated_tokens),
            generation_config=config,
            stop_reason=stop_reason,
            latency_ms=latency_ms,
            model_identity=model_identity,
            reproducibility=reproducibility,
            provenance=provenance,
            weight_hash_verified=True,
        )

    def generate_text(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        tenant_id: str = "default_tenant",
        session_id: str = "default_session",
        truncate_if_overflow: bool = False,
    ) -> str:
        """Convenience method returning decoded generated text directly."""
        req = InferenceRequest(
            prompt=prompt,
            generation_config=config,
            tenant_id=tenant_id,
            session_id=session_id,
            truncate_if_overflow=truncate_if_overflow,
        )
        res = self.execute(req)
        return res.text


# ─────────────────────────────────────────────────────────────────────────────
# 5. Checkpoint & Model Identity Validation Functions
# ─────────────────────────────────────────────────────────────────────────────

def validate_checkpoint_compatibility(
    checkpoint: Dict[str, Any],
    expected_config: Optional[ModelConfig] = None,
) -> bool:
    """
    Validate that checkpoint dictionary is strictly compatible with ChakrMicro.

    Checks:
    1. Must be a dict containing 'model_state_dict'.
    2. Must contain all required ChakrMicro parameter keys.
    3. Parameter tensor shapes must match ModelConfig.
    4. Total parameter count must match exactly 3,443,136.

    Raises:
        IncompatibleCheckpointError on any discrepancy.
    """
    if not isinstance(checkpoint, dict):
        raise IncompatibleCheckpointError(f"Checkpoint must be a dict, got {type(checkpoint).__name__}")

    state_dict = checkpoint.get("model_state_dict")
    if state_dict is None:
        raise IncompatibleCheckpointError("Checkpoint missing 'model_state_dict' key.")

    cfg = expected_config or ModelConfig()
    ref_model = ChakrMicro(cfg)
    ref_dict = ref_model.state_dict()

    # Verify all expected keys are present
    missing_keys = set(ref_dict.keys()) - set(state_dict.keys())
    if missing_keys:
        raise IncompatibleCheckpointError(f"Checkpoint missing required parameter keys: {sorted(missing_keys)}")

    unexpected_keys = set(state_dict.keys()) - set(ref_dict.keys())
    if unexpected_keys:
        raise IncompatibleCheckpointError(f"Checkpoint contains unexpected parameter keys: {sorted(unexpected_keys)}")

    # Verify tensor shapes and count parameters
    total_params = 0
    for key, ref_tensor in ref_dict.items():
        ckpt_tensor = state_dict[key]
        if not isinstance(ckpt_tensor, torch.Tensor):
            raise IncompatibleCheckpointError(f"Parameter '{key}' in checkpoint is not a torch.Tensor")
        if ckpt_tensor.shape != ref_tensor.shape:
            raise IncompatibleCheckpointError(
                f"Shape mismatch for parameter '{key}': checkpoint has {ckpt_tensor.shape}, "
                f"expected {ref_tensor.shape}"
            )
        if key != "lm_head.weight":
            total_params += ckpt_tensor.numel()

    if total_params != 3_443_136:
        raise IncompatibleCheckpointError(
            f"Checkpoint total unique parameters {total_params} != expected 3443136"
        )

    return True


def load_and_validate_checkpoint(
    checkpoint_path: Union[str, Path],
    model: ChakrMicro,
    expected_config: Optional[ModelConfig] = None,
) -> Dict[str, Any]:
    """
    Safely load a checkpoint from disk and validate architectural compatibility.

    Raises:
        FileNotFoundError if checkpoint path does not exist.
        IncompatibleCheckpointError if weights/shapes are incompatible.
    """
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {path}")

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    validate_checkpoint_compatibility(checkpoint, expected_config=expected_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint

