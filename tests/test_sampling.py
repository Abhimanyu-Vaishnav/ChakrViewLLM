"""
Unit tests for Step 10 Sampling Subsystem (chakrview.runtime.sampling).
"""

import pytest
import torch
from chakrview.runtime.sampling import (
    SamplingConfig,
    SamplingStrategy,
    Sampler,
    apply_repetition_penalty,
    apply_top_k,
    apply_top_p,
    apply_min_prob,
)


def test_sampling_config_validation():
    cfg = SamplingConfig(temperature=0.7, top_k=50, top_p=0.9, repetition_penalty=1.1)
    assert cfg.strategy == SamplingStrategy.HYBRID

    greedy_cfg = SamplingConfig(temperature=0.0)
    assert greedy_cfg.is_greedy is True
    assert greedy_cfg.strategy == SamplingStrategy.GREEDY

    # Invalid temperature
    with pytest.raises(ValueError, match="temperature must be non-negative"):
        SamplingConfig(temperature=-0.5)

    # Invalid top_p
    with pytest.raises(ValueError, match="top_p must be in"):
        SamplingConfig(top_p=1.5)

    # Invalid repetition penalty
    with pytest.raises(ValueError, match="repetition_penalty must be >= 1.0"):
        SamplingConfig(repetition_penalty=0.8)


def test_greedy_sampling():
    sampler = Sampler()
    logits = torch.tensor([1.2, 0.5, 4.8, 2.1, -1.0])
    token_id = sampler.sample(logits, config=SamplingConfig(temperature=0.0))
    assert token_id == 2  # argmax is index 2 (value 4.8)


def test_repetition_penalty():
    logits = torch.tensor([2.0, 5.0, 4.0, 1.0])
    # Token 1 has highest logit (5.0). Penalize it heavily (penalty = 3.0)
    penalized = apply_repetition_penalty(logits, tokens=[1], penalty=3.0)
    # Logit for token 1 was 5.0 -> 5.0 / 3.0 = 1.6667
    assert penalized[1] < penalized[2]  # Token 2 (4.0) is now higher than token 1

    # Unseen tokens remain unchanged
    assert penalized[0] == logits[0]
    assert penalized[3] == logits[3]


def test_top_k_filtering():
    logits = torch.tensor([1.0, 5.0, 3.0, 2.0, 4.0])
    # Top 2 keeps values 5.0 (idx 1) and 4.0 (idx 4)
    filtered = apply_top_k(logits, top_k=2)
    assert filtered[1] == 5.0
    assert filtered[4] == 4.0
    assert filtered[0] == -float("inf")
    assert filtered[2] == -float("inf")
    assert filtered[3] == -float("inf")


def test_top_p_filtering():
    # Make token 0 very dominant
    logits = torch.tensor([10.0, 2.0, 1.0, 0.0])
    filtered = apply_top_p(logits, top_p=0.8)
    assert filtered[0] == 10.0
    # Lower tokens filtered to -inf
    assert filtered[3] == -float("inf")


def test_deterministic_seed():
    sampler = Sampler()
    logits = torch.tensor([1.0, 2.0, 1.5, 2.2, 1.8])
    cfg = SamplingConfig(temperature=0.8, seed=12345)

    samples_run1 = [sampler.sample(logits, config=cfg, step=i) for i in range(10)]
    samples_run2 = [sampler.sample(logits, config=cfg, step=i) for i in range(10)]

    assert samples_run1 == samples_run2, "Sampling with same seed must be deterministic"


def test_nan_inf_protection():
    sampler = Sampler()
    corrupt_nan = torch.tensor([1.0, float("nan"), 3.0])
    with pytest.raises(ValueError, match="NaN or Inf"):
        sampler.sample(corrupt_nan)

    corrupt_inf = torch.tensor([1.0, float("inf"), 3.0])
    with pytest.raises(ValueError, match="NaN or Inf"):
        sampler.sample(corrupt_inf)
