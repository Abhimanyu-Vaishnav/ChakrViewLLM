"""
Unit tests for SFT gradient accumulation and prompt-masked loss gradient dynamics.

Verifies:
1. Micro-batch step vs optimizer step distinction.
2. Gradient scaling factor 1 / grad_accum_steps.
3. Loss masking gradient verification (prompt positions receive zero gradient, response positions receive non-zero gradient).
"""

from pathlib import Path
import pytest
import torch
import torch.nn as nn
import torch.optim as optim

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.training.sft_engine import SFTConfig, SFTExperimentEngine


def test_gradient_accumulation_step_scaling(tmp_path):
    """Verifies that running SFT with grad_accum_steps > 1 performs the expected number of optimizer updates."""
    train_file = tmp_path / "train.jsonl"
    val_file = tmp_path / "val.jsonl"

    data = [
        {"prompt": f"Explain concept {i}", "response": f"Concept {i} details."}
        for i in range(10)
    ]
    with open(train_file, "w", encoding="utf-8") as f:
        for d in data:
            f.write(f'{{"prompt": "{d["prompt"]}", "response": "{d["response"]}"}}\n')
    with open(val_file, "w", encoding="utf-8") as f:
        for d in data[:2]:
            f.write(f'{{"prompt": "{d["prompt"]}", "response": "{d["response"]}"}}\n')

    out_dir = tmp_path / "sft_out"

    cfg = SFTConfig(
        repo_root=Path("."),
        train_file=train_file,
        val_file=val_file,
        output_dir=out_dir,
        base_checkpoint=Path("experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt"),
        steps=4,
        batch_size=2,
        grad_accum_steps=2,
        val_interval=2,
        save_interval=2,
    )

    engine = SFTExperimentEngine(cfg)
    metrics = engine.train()

    assert metrics["total_steps"] == 4
    assert len(engine.history) == 4
    for record in engine.history:
        assert record["grad_accum_steps"] == 2
        assert record["train_loss"] > 0.0
        assert record["grad_norm"] >= 0.0


def test_prompt_mask_gradient_exactness():
    """Confirms mathematically that target position -100 yields exactly zero gradient into prompt logits."""
    # Tiny linear projection as mock model head
    vocab_size = 16
    embed_dim = 8
    seq_len = 6

    head = nn.Linear(embed_dim, vocab_size, bias=False)
    hidden_states = torch.randn(1, seq_len, embed_dim, requires_grad=True)

    # Prompt tokens are first 3 positions: targets are -100
    # Response tokens are positions 3, 4, 5
    targets = torch.tensor([[-100, -100, -100, 4, 8, 12]], dtype=torch.long)

    logits = head(hidden_states)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    loss = loss_fn(logits.view(-1, vocab_size), targets.view(-1))
    loss.backward()

    # Gradients on hidden_states for positions 0, 1, 2 must be exactly 0
    assert torch.all(hidden_states.grad[0, :3, :] == 0.0), "Prompt token positions received non-zero gradients!"
    # Gradients on response positions must be non-zero
    assert torch.any(hidden_states.grad[0, 3:, :] != 0.0), "Response positions should have active gradients!"


def test_mask_prompt_option_and_target_alignment():
    """
    Verifies that mask_prompt=True masks prompt tokens with -100, while
    mask_prompt=False supervises prompt tokens.
    Verifies that both preserve:
    - input sequence: [BOS] + prompt_tokens + response_tokens
    - response targets: response_tokens + [EOS]
    - alignment: input[len(p)] -> target[len(p)] == response[0]
    """
    from chakrview.training.prompt_contract import build_causal_sequence

    prompt_toks = [10, 20, 30]
    resp_toks = [40, 50]

    # Arm 2: mask_prompt=True (Default)
    inp_masked, tgt_masked = build_causal_sequence(prompt_toks, resp_toks, mask_prompt=True)
    assert inp_masked == [0, 10, 20, 30, 40, 50]
    assert tgt_masked == [-100, -100, -100, 40, 50, 1]
    # First response token is supervised at position len(prompt_toks)
    assert tgt_masked[len(prompt_toks)] == resp_toks[0]

    # Arm 1: mask_prompt=False (Full Sequence Supervision)
    inp_unmasked, tgt_unmasked = build_causal_sequence(prompt_toks, resp_toks, mask_prompt=False)
    assert inp_unmasked == [0, 10, 20, 30, 40, 50]
    assert tgt_unmasked == [10, 20, 30, 40, 50, 1]
    # In unmasked mode, BOS (position 0) predicts prompt_toks[0]
    assert tgt_unmasked[0] == prompt_toks[0]
    # Final prompt token (position 3) predicts response_toks[0]
    assert tgt_unmasked[len(prompt_toks)] == resp_toks[0]
    # Final response token (position 4) predicts EOS (1)
    assert tgt_unmasked[-1] == 1


def test_unmasked_prompt_gradient_flow():
    """Confirms mathematically that mask_prompt=False causes gradients to flow into prompt positions."""
    vocab_size = 16
    embed_dim = 8
    seq_len = 6

    head = nn.Linear(embed_dim, vocab_size, bias=False)
    hidden_states = torch.randn(1, seq_len, embed_dim, requires_grad=True)

    # All 6 positions supervised: prompt targets + response targets + EOS
    targets = torch.tensor([[2, 3, 4, 5, 6, 1]], dtype=torch.long)

    logits = head(hidden_states)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    loss = loss_fn(logits.view(-1, vocab_size), targets.view(-1))
    loss.backward()

    # All positions (including prompt positions 0, 1, 2) must have active gradients
    assert torch.all(hidden_states.grad[0, :3, :] != 0.0), "Prompt token positions should have active gradients when unmasked!"
    assert torch.all(hidden_states.grad[0, 3:, :] != 0.0), "Response token positions must have active gradients!"


def test_token_weighted_gradient_accumulation_matches_effective_batch():
    """
    Verifies that accumulating micro-batch losses weighted by active token counts
    yields gradients mathematically identical to computing loss over the full effective batch.
    """
    embed_dim = 8
    vocab_size = 12

    # Shared model layer
    head_eff = nn.Linear(embed_dim, vocab_size, bias=False)
    head_accum = nn.Linear(embed_dim, vocab_size, bias=False)
    head_accum.load_state_dict(head_eff.state_dict())

    # Micro-batch 1: 4 tokens (e.g. 1 prompt token masked with -100, 3 active targets)
    h1 = torch.randn(1, 4, embed_dim)
    t1 = torch.tensor([[-100, 2, 5, 8]], dtype=torch.long)
    active1 = (t1 != -100).sum().item()  # 3

    # Micro-batch 2: 3 tokens (e.g. 2 prompt tokens masked with -100, 1 active target)
    h2 = torch.randn(1, 3, embed_dim)
    t2 = torch.tensor([[-100, -100, 7]], dtype=torch.long)
    active2 = (t2 != -100).sum().item()  # 1

    total_active = active1 + active2  # 4 total active tokens in effective batch

    # Method 1: Reference effective batch
    # Concatenate inputs and targets (flattened across active positions)
    logits1_eff = head_eff(h1).view(-1, vocab_size)
    logits2_eff = head_eff(h2).view(-1, vocab_size)
    all_logits = torch.cat([logits1_eff, logits2_eff], dim=0)
    all_targets = torch.cat([t1.view(-1), t2.view(-1)], dim=0)

    loss_eff = nn.CrossEntropyLoss(ignore_index=-100)(all_logits, all_targets)
    loss_eff.backward()

    # Method 2: SFT token-weighted accumulation
    loss_sum_fn = nn.CrossEntropyLoss(ignore_index=-100, reduction="sum")
    head_accum.zero_grad()

    l1 = loss_sum_fn(head_accum(h1).view(-1, vocab_size), t1.view(-1))
    (l1 / total_active).backward()

    l2 = loss_sum_fn(head_accum(h2).view(-1, vocab_size), t2.view(-1))
    (l2 / total_active).backward()

    # Compare accumulated gradients with effective-batch gradients
    grad_diff = torch.max(torch.abs(head_eff.weight.grad - head_accum.weight.grad)).item()
    assert grad_diff < 1e-6, f"Gradient mismatch: max diff = {grad_diff}"
