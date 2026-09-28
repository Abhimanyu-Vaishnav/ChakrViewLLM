"""
Interactive Cognitive Inference Engine for ChakrView (Step 10).

Provides production-grade autoregressive generation substrate:
- Persistent Key-Value (KV) cache for O(N) single-token incremental decoding
- Token-by-token streaming generator
- Multi-strategy sampling (Greedy, Temperature, Top-K, Top-P, Repetition Penalty)
- Context boundary enforcement (T <= 512, overflow policies)
- Hardware-plan integration (CPU multi-threading, memory limits)
- Numerical safety (finite logit checks, bounds validation)
- Privacy-first structured observability (zero prompt persistence)
- Strict security boundary (pure compute, zero OS/shell authority)
"""

from dataclasses import dataclass, field, asdict
from enum import Enum
import time
from typing import Dict, Iterator, List, Optional, Set, Tuple, Union, Any
import torch

from chakrview.brain.cache import KVCache, KVCacheOverflowError
from chakrview.brain.model import ChakrMicro
from chakrview.runtime.hardware import ModelExecutionPlan, PrecisionType
from chakrview.runtime.sampling import SamplingConfig, Sampler
from chakrview.tokenizer.tokenizer import BPETokenizer


class StopReason(str, Enum):
    """Reason for termination of autoregressive generation."""
    EOS = "eos"
    MAX_TOKENS = "max_tokens"
    CONTEXT_LIMIT = "context_limit"
    USER_STOP = "user_stop"
    ERROR = "error"


@dataclass
class StreamToken:
    """
    An incremental token emitted during streaming autoregressive generation.
    
    Attributes:
        token_id: Integer vocabulary token ID.
        text: Lossless UTF-8 decoded text fragment for this token.
        cumulative_token_count: Number of generated tokens emitted so far.
        is_final: True if this token concludes the sequence.
        stop_reason: Termination reason if is_final is True.
    """
    token_id: int
    text: str
    cumulative_token_count: int
    is_final: bool = False
    stop_reason: Optional[StopReason] = None


@dataclass
class GenerationConfig:
    """
    Runtime controls for autoregressive generation.
    
    Attributes:
        max_new_tokens: Maximum number of tokens to generate.
        sampling: Sampling strategy parameters.
        stop_token_ids: List of token IDs that trigger sequence termination (default: [1] for EOS).
        context_window: Hard upper bound on sequence length (<= 512).
        context_overflow_policy: Policy when prompt + generation reaches context_window
                                 ("stop", "truncate", "sliding_window").
        stream_interval: Emit stream event every N tokens.
    """
    max_new_tokens: int = 64
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    stop_token_ids: List[int] = field(default_factory=lambda: [1])  # EOS = 1
    context_window: int = 512
    context_overflow_policy: str = "stop"
    stream_interval: int = 1

    def __post_init__(self) -> None:
        if self.max_new_tokens <= 0:
            raise ValueError(f"max_new_tokens must be positive, got {self.max_new_tokens}")
        if not (1 <= self.context_window <= 512):
            raise ValueError(f"context_window must be in [1, 512], got {self.context_window}")
        if self.context_overflow_policy not in ("stop", "truncate", "sliding_window"):
            raise ValueError(
                f"Invalid context_overflow_policy: '{self.context_overflow_policy}'. "
                "Must be 'stop', 'truncate', or 'sliding_window'."
            )

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["sampling"] = self.sampling.to_dict()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GenerationConfig":
        data = dict(data)
        if "sampling" in data and isinstance(data["sampling"], dict):
            data["sampling"] = SamplingConfig.from_dict(data["sampling"])
        return cls(**data)


@dataclass
class InferenceMetrics:
    """
    Structured performance, hardware, and latency accounting for an inference run.
    
    Privacy-first: Does NOT persist prompt text.
    """
    model_version: str
    tokenizer_checksum: str
    prompt_tokens: int
    generated_tokens: int
    total_tokens: int
    prefill_latency_ms: float
    decode_latency_ms: float
    total_latency_ms: float
    throughput_tokens_per_sec: float
    first_token_latency_ms: float
    stop_reason: StopReason
    cache_memory_bytes: int
    device: str

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["stop_reason"] = self.stop_reason.value
        return data


@dataclass
class GenerationResult:
    """Complete output package from an inference generation."""
    text: str
    token_ids: List[int]
    metrics: InferenceMetrics
    stop_reason: StopReason

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "token_ids": self.token_ids,
            "metrics": self.metrics.to_dict(),
            "stop_reason": self.stop_reason.value,
        }


class InferenceSession:
    """
    Interactive, stateful inference session coordinating model, tokenizer, and KV cache.
    
    Architectural Boundaries:
    - Pure computation engine: zero OS/filesystem/shell authority
    - Thread-safe within a single execution session
    - Reusable across applications (chat, tools, coding, reasoning, agents)
    """

    def __init__(
        self,
        model: ChakrMicro,
        tokenizer: BPETokenizer,
        execution_plan: Optional[ModelExecutionPlan] = None,
        sampler: Optional[Sampler] = None,
        model_version: str = "chakrmicro-v0.1",
        tokenizer_checksum: str = "7498d92adeef7c6db98d89a444a7f0e303dd5e7ea4b679a95781a95e6347c617",
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.execution_plan = execution_plan
        self.sampler = sampler or Sampler()
        self.model_version = model_version
        self.tokenizer_checksum = tokenizer_checksum

        # Set evaluation mode
        self.model.eval()

        # Determine target device and context ceiling
        self.device = torch.device(execution_plan.device) if execution_plan else torch.device("cpu")
        self.max_context = execution_plan.max_context_len if execution_plan else self.model.config.max_seq_len
        
        # Configure PyTorch CPU threading if specified
        if execution_plan and execution_plan.device == "cpu" and execution_plan.thread_count > 0:
            torch.set_num_threads(execution_plan.thread_count)

        # Initialize KV Cache
        self.kv_cache: KVCache = KVCache(
            num_layers=self.model.config.n_layers,
            max_seq_len=self.max_context,
            device=self.device,
            dtype=torch.float32,
        )

        self._active_prompt_ids: List[int] = []
        self._generated_token_ids: List[int] = []
        self._latest_logits: Optional[torch.Tensor] = None

    def reset(self) -> None:
        """Reset KV cache and session state for next generation."""
        self.kv_cache.reset()
        self._active_prompt_ids.clear()
        self._generated_token_ids.clear()
        self._latest_logits = None

    def prefill(self, prompt: str, add_bos: bool = True) -> Tuple[List[int], float]:
        """
        Encode prompt and execute initial parallel prefill through ChakrMicro.
        
        Args:
            prompt: Input text sequence.
            add_bos: Whether to prepend BOS token (default: True).
            
        Returns:
            Tuple of (prompt_token_ids, prefill_latency_ms).
        """
        self.reset()
        t0 = time.perf_counter()

        # 1. Encode prompt
        token_ids = self.tokenizer.encode(prompt, add_bos=add_bos, add_eos=False)
        if not token_ids:
            # Fallback if empty: inject BOS
            token_ids = [0]

        # 2. Enforce context capacity
        if len(token_ids) >= self.max_context:
            # Prompt exceeds maximum sequence length
            token_ids = token_ids[-self.max_context + 1:]

        self._active_prompt_ids = token_ids
        prompt_tensor = torch.tensor([token_ids], dtype=torch.long, device=self.device)

        # 3. Model Prefill forward pass
        with torch.no_grad():
            logits, _ = self.model.prefill(prompt_tensor, kv_cache=self.kv_cache)
            # Store last token's logits for subsequent sampling
            self._latest_logits = logits[0, -1, :]

        latency_ms = (time.perf_counter() - t0) * 1000.0
        return token_ids, latency_ms

    def stream(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        add_bos: bool = True,
    ) -> Iterator[StreamToken]:
        """
        Stream autoregressively generated tokens one by one.
        
        Args:
            prompt: Input text prompt.
            config: Optional generation configuration override.
            add_bos: Whether to prepend BOS token.
            
        Yields:
            StreamToken for each emitted token until sequence termination.
        """
        cfg = config or GenerationConfig()
        prompt_ids, prefill_latency = self.prefill(prompt, add_bos=add_bos)

        stop_token_set = set(cfg.stop_token_ids)
        cumulative_count = 0

        while cumulative_count < cfg.max_new_tokens:
            current_seq_len = self.kv_cache.sequence_length

            # 1. Check Context Limit
            if current_seq_len >= cfg.context_window or current_seq_len >= self.max_context:
                if cfg.context_overflow_policy == "stop":
                    yield StreamToken(
                        token_id=0,
                        text="",
                        cumulative_token_count=cumulative_count,
                        is_final=True,
                        stop_reason=StopReason.CONTEXT_LIMIT,
                    )
                    return
                elif cfg.context_overflow_policy == "sliding_window":
                    # Evict oldest tokens (sliding window)
                    keep_len = self.max_context // 2
                    self.kv_cache.truncate(keep_len)
                else:  # "truncate"
                    yield StreamToken(
                        token_id=0,
                        text="",
                        cumulative_token_count=cumulative_count,
                        is_final=True,
                        stop_reason=StopReason.CONTEXT_LIMIT,
                    )
                    return

            # 2. Sample next token
            next_token_id = self.sampler.sample(
                logits=self._latest_logits,
                generated_tokens=self._generated_token_ids,
                config=cfg.sampling,
                step=cumulative_count,
            )

            # 3. Decode token to UTF-8 text fragment
            text_fragment = self.tokenizer.decode([next_token_id])
            self._generated_token_ids.append(next_token_id)
            cumulative_count += 1

            # 4. Check for Stop Token (EOS)
            if next_token_id in stop_token_set:
                yield StreamToken(
                    token_id=next_token_id,
                    text=text_fragment,
                    cumulative_token_count=cumulative_count,
                    is_final=True,
                    stop_reason=StopReason.EOS,
                )
                return

            # 5. Check for Max Tokens Reached
            if cumulative_count >= cfg.max_new_tokens:
                yield StreamToken(
                    token_id=next_token_id,
                    text=text_fragment,
                    cumulative_token_count=cumulative_count,
                    is_final=True,
                    stop_reason=StopReason.MAX_TOKENS,
                )
                return

            # Emit normal continuation token
            yield StreamToken(
                token_id=next_token_id,
                text=text_fragment,
                cumulative_token_count=cumulative_count,
                is_final=False,
            )

            # 6. Single-token incremental decode for next step
            next_tensor = torch.tensor([[next_token_id]], dtype=torch.long, device=self.device)
            with torch.no_grad():
                step_logits = self.model.decode_next(next_tensor, kv_cache=self.kv_cache)
                self._latest_logits = step_logits[0, -1, :]

    def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        add_bos: bool = True,
    ) -> GenerationResult:
        """
        Execute full autoregressive generation and return aggregated GenerationResult.
        """
        cfg = config or GenerationConfig()
        t_start = time.perf_counter()

        prompt_ids, prefill_latency_ms = self.prefill(prompt, add_bos=add_bos)
        t_prefill_done = time.perf_counter()

        tokens_generated: List[int] = []
        text_fragments: List[str] = []
        stop_reason = StopReason.MAX_TOKENS
        first_token_latency_ms = 0.0

        for token in self.stream(prompt, config=cfg, add_bos=add_bos):
            if token.cumulative_token_count == 1:
                first_token_latency_ms = (time.perf_counter() - t_start) * 1000.0

            if token.token_id != 0 or not token.is_final:
                tokens_generated.append(token.token_id)
                text_fragments.append(token.text)

            if token.is_final:
                stop_reason = token.stop_reason or StopReason.MAX_TOKENS
                break

        t_end = time.perf_counter()
        total_latency_ms = (t_end - t_start) * 1000.0
        decode_latency_ms = (t_end - t_prefill_done) * 1000.0

        num_gen = len(tokens_generated)
        throughput = (num_gen / (decode_latency_ms / 1000.0)) if decode_latency_ms > 0 else 0.0

        metrics = InferenceMetrics(
            model_version=self.model_version,
            tokenizer_checksum=self.tokenizer_checksum,
            prompt_tokens=len(prompt_ids),
            generated_tokens=num_gen,
            total_tokens=len(prompt_ids) + num_gen,
            prefill_latency_ms=prefill_latency_ms,
            decode_latency_ms=decode_latency_ms,
            total_latency_ms=total_latency_ms,
            throughput_tokens_per_sec=throughput,
            first_token_latency_ms=first_token_latency_ms,
            stop_reason=stop_reason,
            cache_memory_bytes=self.kv_cache.total_memory_bytes,
            device=str(self.device),
        )

        return GenerationResult(
            text="".join(text_fragments),
            token_ids=tokens_generated,
            metrics=metrics,
            stop_reason=stop_reason,
        )
