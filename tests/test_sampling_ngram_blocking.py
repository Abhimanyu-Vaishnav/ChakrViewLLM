"""
Regression tests for repetition penalty and sliding-window n-gram blocking in Sampling subsystem.
"""
import pytest
import torch
from chakrview.runtime.sampling import (
    SamplingConfig,
    Sampler,
    apply_repetition_penalty,
    apply_no_repeat_ngram,
)

def test_apply_no_repeat_ngram_blocks_cyclic_sequence():
    # Sequence: 10, 20, 30, 10, 20 -> next token in cycle would be 30
    logits = torch.zeros(50)
    logits[30] = 5.0
    logits[40] = 2.0
    
    tokens = [10, 20, 30, 10, 20]
    filtered = apply_no_repeat_ngram(logits, tokens, ngram_size=3)
    
    assert filtered[30].item() == -float("inf")
    assert filtered[40].item() == 2.0
    assert torch.argmax(filtered).item() == 40


def test_sampler_with_no_repeat_ngram_breaks_greedy_loop():
    sampler = Sampler()
    cfg = SamplingConfig(temperature=0.0, no_repeat_ngram_size=2)
    
    # Always prefer token 5 if unmasked
    logits = torch.zeros(10)
    logits[5] = 10.0
    logits[6] = 5.0
    
    # First step
    tok1 = sampler.sample(logits, generated_tokens=[], config=cfg)
    assert tok1 == 5
    
    # Second step with generated [5]: token 5 follows 5, with bigram size=2, if 5 already generated,
    # wait: bigram prefix is [5]. Has [5] appeared before with a follow token? No prior bigrams yet.
    # tokens = [5] -> prefix is (5). History has no tokens after 5 yet.
    tok2 = sampler.sample(logits, generated_tokens=[5], config=cfg)
    assert tok2 == 5
    
    # Third step with generated [5, 5]: prefix is (5). History has (5) -> 5 at indices 0->1.
    # So 5 is banned!
    tok3 = sampler.sample(logits, generated_tokens=[5, 5], config=cfg)
    assert tok3 == 6  # Loop broken!


def test_sampling_config_validation():
    with pytest.raises(ValueError, match="no_repeat_ngram_size must be >= 0"):
        SamplingConfig(no_repeat_ngram_size=-1)