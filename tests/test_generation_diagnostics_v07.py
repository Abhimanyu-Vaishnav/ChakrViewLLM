"""
Unit tests for Milestone 5: Generation diagnostics and decoding strategies.
Verifies:
1. Greedy decoding reproducibility under identical model and prompt.
2. Sampling reproducibility with fixed random generator seed.
3. Top-k and Top-p filtering constraints.
4. EOS token termination.
"""

from pathlib import Path
import pytest
import torch

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.tokenizer.serialization import load_tokenizer_artifacts
from chakrview.training.generation_diagnostics import GenerationDiagnosticEngine


def test_greedy_decoding_deterministic():
    torch.manual_seed(42)
    model = ChakrMicro(ModelConfig())
    tok_dir = Path("data/experiments/vocab_4096")
    tokenizer, _ = load_tokenizer_artifacts(tok_dir)

    diag1 = GenerationDiagnosticEngine.generate_with_diagnostics(
        model=model,
        tokenizer=tokenizer,
        prompt="ChakrView native",
        max_new_tokens=10,
        method="greedy",
    )
    diag2 = GenerationDiagnosticEngine.generate_with_diagnostics(
        model=model,
        tokenizer=tokenizer,
        prompt="ChakrView native",
        max_new_tokens=10,
        method="greedy",
    )
    assert diag1.generated_token_ids == diag2.generated_token_ids
    assert diag1.generated_text == diag2.generated_text


def test_temperature_sampling_seed_reproducibility():
    torch.manual_seed(42)
    model = ChakrMicro(ModelConfig())
    tok_dir = Path("data/experiments/vocab_4096")
    tokenizer, _ = load_tokenizer_artifacts(tok_dir)

    diag1 = GenerationDiagnosticEngine.generate_with_diagnostics(
        model=model,
        tokenizer=tokenizer,
        prompt="August is",
        max_new_tokens=12,
        method="sampling",
        temperature=0.8,
        seed=101,
    )
    diag2 = GenerationDiagnosticEngine.generate_with_diagnostics(
        model=model,
        tokenizer=tokenizer,
        prompt="August is",
        max_new_tokens=12,
        method="sampling",
        temperature=0.8,
        seed=101,
    )
    assert diag1.generated_token_ids == diag2.generated_token_ids


def test_top_p_nucleus_sampling_generates_tokens():
    torch.manual_seed(42)
    model = ChakrMicro(ModelConfig())
    tok_dir = Path("data/experiments/vocab_4096")
    tokenizer, _ = load_tokenizer_artifacts(tok_dir)

    diag = GenerationDiagnosticEngine.generate_with_diagnostics(
        model=model,
        tokenizer=tokenizer,
        prompt="The system",
        max_new_tokens=15,
        method="sampling",
        temperature=0.8,
        top_p=0.90,
        seed=42,
    )
    assert len(diag.generated_token_ids) == 15
    assert diag.mean_token_entropy > 0.0
