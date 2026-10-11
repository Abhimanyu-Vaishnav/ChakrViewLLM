"""
Neural Inference Engine for ChakrView (Step 20).

Executes autoregressive neural inference through ChakrMicro v0.1 while strictly
guaranteeing:
1. ZERO RUNTIME WEIGHT UPDATES: Model remains strictly eval() and read-only.
2. CONTEXT CEILING: Sequence length <= 512 tokens.
3. HONEST UNCERTAINTY: Real mathematical metrics (entropy, top-2 margin) when
   computed, or explicitly marked unavailable. No fabricated confidence scores.
4. SOVEREIGN EXECUTION: Zero cloud, external LLM, or GPU dependencies.
"""

import math
import time
from typing import Dict, List, Optional, Any, Tuple
import torch
import torch.nn.functional as F

from chakrview.brain.cache import KVCache
from chakrview.brain.model import ChakrMicro
from chakrview.tokenizer.tokenizer import BPETokenizer
from chakrview.runtime.sampling import SamplingConfig, Sampler
from chakrview.intelligence.contracts import (
    NeuralInferenceRequest,
    NeuralInferenceResult,
    UncertaintyMetric,
)


class NeuralInferenceEngine:
    """
    Sovereign neural inference engine interfacing with the frozen ChakrMicro generative core.
    """

    def __init__(
        self,
        model: ChakrMicro,
        tokenizer: BPETokenizer,
        model_version: str = "chakrmicro-v0.1",
        device: Optional[torch.device] = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.model_version = model_version
        self.device = device or torch.device("cpu")
        self.sampler = Sampler()

        # Enforce evaluation mode permanently
        self.model.eval()

        # Allocate reusable KV Cache bounded by frozen context length (512)
        self.kv_cache = KVCache(
            num_layers=self.model.config.n_layers,
            max_seq_len=self.model.config.max_seq_len,
            device=self.device,
            dtype=torch.float32,
        )

        # Baseline model weight hash/fingerprint to verify weights remain untouched
        self._initial_param_fingerprint = self._compute_weights_fingerprint()

    def _compute_weights_fingerprint(self) -> float:
        """Compute sum of final norm weights as a fast immutability sanity check."""
        with torch.no_grad():
            if hasattr(self.model, "final_norm"):
                return float(self.model.final_norm.weight.sum().item())
            elif hasattr(self.model, "ln_f"):
                return float(self.model.ln_f.weight.sum().item())
            return float(next(self.model.parameters()).sum().item())

    def verify_weights_unmodified(self) -> bool:
        """Sanity check that runtime execution has not modified model weights."""
        current_fp = self._compute_weights_fingerprint()
        return math.isclose(current_fp, self._initial_param_fingerprint, abs_tol=1e-7)

    def infer(self, request: NeuralInferenceRequest) -> NeuralInferenceResult:
        """
        Execute bounded autoregressive generation for the given request.
        """
        t_start = time.perf_counter()

        # 1. Reset KV Cache
        self.kv_cache.reset()

        # 2. Tokenize prompt if token IDs not directly provided
        if request.prompt_tokens is not None:
            prompt_tokens = list(request.prompt_tokens)
        else:
            prompt_tokens = self.tokenizer.encode(request.prompt_text, add_bos=True, add_eos=False)
            if not prompt_tokens:
                prompt_tokens = [0]  # BOS fallback

        # 3. Enforce context window ceiling (prompt + generation <= 512)
        max_context = self.model.config.max_seq_len  # 512
        max_gen = min(request.max_new_tokens, max_context - len(prompt_tokens))
        if max_gen <= 0:
            # Prompt filled entire context window; truncate prompt from left to leave space
            keep_len = max_context - max(16, request.max_new_tokens)
            prompt_tokens = prompt_tokens[-keep_len:]
            max_gen = max_context - len(prompt_tokens)

        # 4. Prefill pass
        prompt_tensor = torch.tensor([prompt_tokens], dtype=torch.long, device=self.device)
        with torch.no_grad():
            logits, _ = self.model.prefill(prompt_tensor, kv_cache=self.kv_cache)
            last_logits = logits[0, -1, :]

        # 5. Autoregressive decoding loop
        generated_tokens: List[int] = []
        stop_token_set = set(request.stop_token_ids)
        stop_reason = "max_tokens"

        entropy_values: List[float] = []
        margin_values: List[float] = []

        sampling_cfg = SamplingConfig(
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p,
        )

        for step in range(max_gen):
            if self.kv_cache.sequence_length >= max_context:
                stop_reason = "context_limit"
                break

            # If uncertainty calculation requested, compute real mathematical metrics from logits
            if request.compute_uncertainty:
                probs = F.softmax(last_logits, dim=-1)
                # Shannon entropy in nats: -sum(p * log(p + eps))
                log_probs = F.log_softmax(last_logits, dim=-1)
                entropy = float(-(probs * log_probs).sum().item())
                entropy_values.append(entropy)

                # Top-1 vs Top-2 probability margin
                top2_probs, _ = torch.topk(probs, k=2)
                margin = float((top2_probs[0] - top2_probs[1]).item())
                margin_values.append(margin)

            # Sample next token
            next_token_id = self.sampler.sample(
                logits=last_logits,
                generated_tokens=generated_tokens,
                config=sampling_cfg,
                step=step,
            )

            generated_tokens.append(next_token_id)

            # Check stop criteria
            if next_token_id in stop_token_set:
                stop_reason = "eos"
                break

            # Incremental decode for next step
            next_tensor = torch.tensor([[next_token_id]], dtype=torch.long, device=self.device)
            with torch.no_grad():
                step_logits = self.model.decode_next(next_tensor, kv_cache=self.kv_cache)
                last_logits = step_logits[0, -1, :]

        # 6. Uncertainty evaluation
        if request.compute_uncertainty and entropy_values:
            avg_entropy = sum(entropy_values) / len(entropy_values)
            avg_margin = sum(margin_values) / len(margin_values)
            uncertainty = UncertaintyMetric(
                entropy=avg_entropy,
                top_token_margin=avg_margin,
                is_calibrated=False,  # Raw next-token logits are not temperature/conformal calibrated
                is_available=True,
                notes="Calculated from autoregressive step logits; uncalibrated.",
            )
        else:
            uncertainty = UncertaintyMetric.unavailable()

        # 7. Decode text
        decoded_text = self.tokenizer.decode(generated_tokens, errors="replace")
        latency_ms = (time.perf_counter() - t_start) * 1000.0

        # Verify weights were not modified
        assert self.verify_weights_unmodified(), "Architectural Violation: Model weights modified during inference!"

        return NeuralInferenceResult(
            text=decoded_text,
            token_ids=generated_tokens,
            prompt_tokens_count=len(prompt_tokens),
            generated_tokens_count=len(generated_tokens),
            total_tokens_count=len(prompt_tokens) + len(generated_tokens),
            latency_ms=latency_ms,
            stop_reason=stop_reason,
            uncertainty=uncertainty,
            model_version=self.model_version,
            context_provenance=dict(request.context_provenance),
            weights_modified=False,
        )
