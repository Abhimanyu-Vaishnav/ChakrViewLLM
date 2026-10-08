"""
Sampling Subsystem for ChakrView (Step 10).

Provides deterministic and stochastic token selection strategies:
- Greedy decoding (temperature = 0.0)
- Temperature scaling
- Top-K truncation
- Top-P (nucleus) filtering
- Repetition penalty
- Minimum probability threshold
- Deterministic seeding via independent PyTorch generators
"""

from dataclasses import dataclass, field, asdict
from enum import Enum
import math
from typing import Dict, List, Optional, Set, Union
import torch


class SamplingStrategy(str, Enum):
    """Categorization of token sampling strategy."""
    GREEDY = "greedy"
    TEMPERATURE = "temperature"
    TOP_K = "top_k"
    TOP_P = "top_p"
    HYBRID = "hybrid"


class SamplingProbabilityError(ValueError):
    """Raised when sampling encounters a degenerate or unnormalizable probability distribution."""
    pass


@dataclass
class SamplingConfig:
    """
    Configuration parameters for autoregressive sampling.
    
    Attributes:
        temperature: Logit scaling factor (0.0 for greedy, >0.0 for stochastic).
        top_k: Keep only top K highest-probability tokens (<= 0 to disable).
        top_p: Nucleus threshold cumulative probability (1.0 to disable).
        repetition_penalty: Discount factor applied to previously emitted tokens ([1.0, 10.0]; 1.0 disables).
        min_prob: Minimum token probability threshold (0.0 to disable).
        seed: Random seed for deterministic stochastic sampling.
    """
    temperature: float = 0.0
    top_k: int = 0
    top_p: float = 1.0
    repetition_penalty: float = 1.0
    min_prob: float = 0.0
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        if self.temperature < 0.0:
            raise ValueError(f"temperature must be non-negative, got {self.temperature}")
        if self.top_k < 0:
            raise ValueError(f"top_k must be non-negative, got {self.top_k}")
        if not (0.0 < self.top_p <= 1.0):
            raise ValueError(f"top_p must be in (0.0, 1.0], got {self.top_p}")
        if self.repetition_penalty < 1.0:
            raise ValueError(f"repetition_penalty must be >= 1.0, got {self.repetition_penalty}")
        if self.repetition_penalty > 10.0:
            raise ValueError(f"repetition_penalty must be <= 10.0, got {self.repetition_penalty}")
        if not (0.0 <= self.min_prob < 1.0):
            raise ValueError(f"min_prob must be in [0.0, 1.0), got {self.min_prob}")

    @property
    def is_greedy(self) -> bool:
        return self.temperature == 0.0

    @property
    def strategy(self) -> SamplingStrategy:
        if self.is_greedy:
            return SamplingStrategy.GREEDY
        if self.top_k > 0 and self.top_p < 1.0:
            return SamplingStrategy.HYBRID
        if self.top_k > 0:
            return SamplingStrategy.TOP_K
        if self.top_p < 1.0:
            return SamplingStrategy.TOP_P
        return SamplingStrategy.TEMPERATURE

    def to_dict(self) -> Dict[str, Union[float, int, str, None]]:
        data = asdict(self)
        data["strategy"] = self.strategy.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Union[float, int, str, None]]) -> "SamplingConfig":
        clean = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**clean)


def apply_repetition_penalty(
    logits: torch.Tensor,
    tokens: List[int],
    penalty: float,
) -> torch.Tensor:
    """
    Apply multiplicative repetition penalty to logits of seen tokens.
    
    Formula (Keskar et al., 2019):
        logit = logit / penalty if logit > 0 else logit * penalty
    """
    if penalty <= 1.0 or not tokens:
        return logits

    unique_tokens = list(set(tokens))
    filtered_logits = logits.clone()
    for tok_id in unique_tokens:
        if 0 <= tok_id < filtered_logits.shape[-1]:
            val = filtered_logits[..., tok_id]
            filtered_logits[..., tok_id] = torch.where(val > 0, val / penalty, val * penalty)
    return filtered_logits


def apply_top_k(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    """Filter logits keeping only the top K highest values."""
    if top_k <= 0 or top_k >= logits.shape[-1]:
        return logits

    v, _ = torch.topk(logits, min(top_k, logits.shape[-1]))
    min_val = v[..., -1, None]
    return torch.where(logits < min_val, torch.full_like(logits, -float("inf")), logits)


def apply_top_p(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """
    Nucleus sampling: keep top tokens whose cumulative probability reaches top_p.
    """
    if top_p >= 1.0:
        return logits

    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

    # Remove tokens with cumulative probability above top_p (shift right by 1 to keep first)
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False

    indices_to_remove = sorted_indices_to_remove.scatter(
        dim=-1, index=sorted_indices, src=sorted_indices_to_remove
    )
    return torch.where(indices_to_remove, torch.full_like(logits, -float("inf")), logits)


def apply_min_prob(logits: torch.Tensor, min_prob: float) -> torch.Tensor:
    """Filter out tokens whose softmax probability falls below min_prob threshold."""
    if min_prob <= 0.0:
        return logits

    probs = torch.softmax(logits, dim=-1)
    mask = probs < min_prob
    # Ensure at least the argmax token is preserved
    argmax_idx = torch.argmax(logits, dim=-1, keepdim=True)
    mask.scatter_(dim=-1, index=argmax_idx, value=False)
    return torch.where(mask, torch.full_like(logits, -float("inf")), logits)


class Sampler:
    """
    Thread-safe, deterministic token sampler supporting greedy and stochastic modes.
    """
    def __init__(self, default_config: Optional[SamplingConfig] = None) -> None:
        self.default_config = default_config or SamplingConfig()

    def sample(
        self,
        logits: torch.Tensor,
        generated_tokens: Optional[List[int]] = None,
        config: Optional[SamplingConfig] = None,
        step: int = 0,
        strict_safety: bool = False,
    ) -> int:
        """
        Sample the next token ID from next-token logits.
        
        Args:
            logits: 1D [vocab_size] or 2D [1, vocab_size] logits tensor.
            generated_tokens: List of previously generated token IDs for repetition penalty.
            config: Optional sampling configuration override.
            step: Step index used for deterministic pseudorandom generator seed progression.
            strict_safety: If True, raise SamplingProbabilityError on degenerate distribution.
            
        Returns:
            Integer next-token ID in [0, vocab_size - 1].
        """
        cfg = config or self.default_config
        
        # Flatten to 1D
        if logits.dim() == 2:
            if logits.shape[0] != 1:
                raise ValueError(f"Sampler expects single batch, got {logits.shape}")
            logits = logits[0]
        elif logits.dim() != 1:
            raise ValueError(f"Sampler expects 1D or [1, V] logits, got shape {logits.shape}")

        # Check for numerical corruption
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            raise ValueError("Logits contain NaN or Inf values. Generation aborted for numerical safety.")

        # 1. Greedy path
        if cfg.is_greedy:
            if cfg.repetition_penalty > 1.0 and generated_tokens:
                penalized = apply_repetition_penalty(logits, generated_tokens, cfg.repetition_penalty)
                return int(torch.argmax(penalized).item())
            return int(torch.argmax(logits).item())

        # 2. Stochastic path
        scaled = logits
        if cfg.repetition_penalty > 1.0 and generated_tokens:
            scaled = apply_repetition_penalty(scaled, generated_tokens, cfg.repetition_penalty)

        scaled = scaled / cfg.temperature

        if cfg.top_k > 0:
            scaled = apply_top_k(scaled, cfg.top_k)

        if cfg.top_p < 1.0:
            scaled = apply_top_p(scaled, cfg.top_p)

        if cfg.min_prob > 0.0:
            scaled = apply_min_prob(scaled, cfg.min_prob)

        probs = torch.softmax(scaled, dim=-1)

        # Handle potential zero-probability degeneracies
        if torch.isnan(probs).any() or probs.sum() <= 0:
            if strict_safety:
                raise SamplingProbabilityError(
                    "Degenerate probability distribution (sum <= 0 or NaNs) encountered in Sampler."
                )
            # Fallback to greedy on original logits
            return int(torch.argmax(logits).item())

        # Deterministic RNG generator if seed is set
        generator = None
        if cfg.seed is not None:
            generator = torch.Generator(device=logits.device)
            # Offset by step so sequential tokens have distinct deterministic samples
            generator.manual_seed((cfg.seed + step) % (2**31 - 1))

        sampled_id = torch.multinomial(probs, num_samples=1, generator=generator).item()
        return int(sampled_id)
