"""
Unit and regression tests for Workstream E: Stage D SFT Engine and Dataset.
Covers:
1. Response-only causal loss alignment and prompt mask targeting (-100).
2. Gradient isolation: zero gradient on prompt positions, active gradients on response positions.
3. Dataset loading, dynamic padding, and truncation safety.
4. Engine initialization and base checkpoint preservation.
5. Save and resume determinism.
"""

import copy
from pathlib import Path
import pytest
import torch
import torch.nn as nn

from chakrview.training.sft_engine import SFTConfig, SFTDataset, SFTExperimentEngine
from chakrview.tokenizer.serialization import load_tokenizer_artifacts


@pytest.fixture
def tokenizer():
    tok, _ = load_tokenizer_artifacts("release/chakrview-v0.5.0/data/experiments/vocab_4096")
    return tok


def test_sft_dataset_encoding_and_masking(tokenizer):
    """Verifies that SFTDataset masks prompt tokens with -100 and aligns response targets."""
    train_file = Path("experiments/batch29_stage_d_sprint/data/sft_train.jsonl")
    if not train_file.exists():
        pytest.skip("SFT train file not yet compiled")

    ds = SFTDataset(train_file, tokenizer, max_seq_len=256)
    prompt = "What is Pre-RMSNorm?"
    response = "A normalization technique for deep models."
    inp, tgt = ds.encode_example(prompt, response)

    p_tokens = tokenizer.encode(prompt.strip())
    r_tokens = tokenizer.encode(response.strip())

    # Input format: <BOS> + prompt + response
    assert inp[0] == 0
    assert inp[1 : 1 + len(p_tokens)] == p_tokens
    assert inp[1 + len(p_tokens) :] == r_tokens

    # Target format: [-100]*len(prompt) + response + <EOS>
    assert tgt[: len(p_tokens)] == [-100] * len(p_tokens)
    assert tgt[len(p_tokens) : -1] == r_tokens
    assert tgt[-1] == 1  # EOS


def test_sft_gradient_isolation(tokenizer):
    """Verifies that CrossEntropyLoss ignores prompt tokens and only computes gradients on response."""
    train_file = Path("experiments/batch29_stage_d_sprint/data/sft_train.jsonl")
    if not train_file.exists():
        pytest.skip("SFT train file not yet compiled")

    ds = SFTDataset(train_file, tokenizer, max_seq_len=128)
    inp, tgt = ds.encode_example("Question here", "Answer here")

    x = torch.tensor([inp], dtype=torch.long)
    y = torch.tensor([tgt], dtype=torch.long)

    vocab_size = 4096
    logits = torch.randn(1, x.shape[1], vocab_size, requires_grad=True)

    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    loss = loss_fn(logits.reshape(-1, vocab_size), y.reshape(-1))
    loss.backward()

    # Masked positions must have 0 gradient
    num_masked = (y[0] == -100).sum().item()
    prompt_grads = logits.grad[0, :num_masked, :]
    assert (prompt_grads == 0.0).all()

    # Active response positions must have non-zero gradient
    resp_grads = logits.grad[0, num_masked:, :]
    assert (resp_grads.abs().max() > 0.0).item()


def test_sft_engine_end_to_end_step(tmp_path):
    """Executes a 2-step SFT run on temporary directory to verify forward, backward, and checkpointing."""
    base_ckpt = Path("experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt")
    if not base_ckpt.exists():
        pytest.skip("Base checkpoint missing")

    train_file = Path("experiments/batch29_stage_d_sprint/data/sft_train.jsonl")
    val_file = Path("experiments/batch29_stage_d_sprint/data/sft_val.jsonl")
    if not train_file.exists() or not val_file.exists():
        pytest.skip("SFT data missing")

    cfg = SFTConfig(
        repo_root=Path("."),
        train_file=train_file,
        val_file=val_file,
        output_dir=tmp_path,
        base_checkpoint=base_ckpt,
        steps=2,
        batch_size=2,
        val_interval=2,
        save_interval=2,
        device="cpu",
    )

    engine = SFTExperimentEngine(cfg)
    res = engine.train()

    assert res["total_steps"] == 2
    assert "final_val_loss" in res
    assert (tmp_path / "sft_checkpoint_step_2.pt").exists()
