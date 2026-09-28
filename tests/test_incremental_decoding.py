"""
Unit tests comparing full-forward vs incremental KV-cache decoding (Step 10).
"""

import pytest
import torch
from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.brain.cache import KVCache


def test_prefill_and_decode_next_shapes():
    model = ChakrMicro(ModelConfig())
    model.eval()

    prompt = torch.tensor([[10, 20, 30, 40]], dtype=torch.long)
    prefill_logits, cache = model.prefill(prompt)

    assert prefill_logits.shape == (1, 4, 4096)
    assert cache.sequence_length == 4

    # Decode next token
    next_tok = torch.tensor([[50]], dtype=torch.long)
    next_logits = model.decode_next(next_tok, cache)

    assert next_logits.shape == (1, 1, 4096)
    assert cache.sequence_length == 5


def test_incremental_vs_full_forward_equivalence_multiple_prompts():
    """
    Core Invariant:
    Cached incremental decoding must match full forward pass logits within numerical tolerance (< 1e-4).
    """
    torch.manual_seed(42)
    model = ChakrMicro(ModelConfig())
    model.eval()

    test_prompts = [
        torch.tensor([[5, 12, 100]], dtype=torch.long),
        torch.tensor([[200, 350, 1000, 2500, 3000]], dtype=torch.long),
        torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]], dtype=torch.long),
    ]

    for prompt in test_prompts:
        # 1. Prefill
        prefill_out, cache = model.prefill(prompt)
        full_out_prompt = model(prompt)
        prefill_diff = torch.max(torch.abs(prefill_out - full_out_prompt)).item()
        assert prefill_diff < 1e-5, f"Prefill discrepancy {prefill_diff} exceeds tolerance"

        # 2. Sequential decode 5 tokens
        current_seq = prompt.clone()
        for step in range(5):
            # Target next token
            next_token_id = (step * 73 + 17) % 4096
            next_token = torch.tensor([[next_token_id]], dtype=torch.long)

            # Cached decode
            cached_logits = model.decode_next(next_token, kv_cache=cache)[:, 0, :]

            # Full forward
            current_seq = torch.cat([current_seq, next_token], dim=1)
            full_logits = model(current_seq)[:, -1, :]

            delta = torch.max(torch.abs(cached_logits - full_logits)).item()
            assert delta < 1e-4, f"Step {step} discrepancy {delta} exceeds 1e-4 tolerance"
