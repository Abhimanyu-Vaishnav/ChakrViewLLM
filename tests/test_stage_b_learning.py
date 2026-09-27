"""
Step 6.3 Tests: Controlled Pre-Training Learning Validation on Stage B Shards.

Verifies:
1. Causal Target Shift Contract: y_t == x_{t+1} without future-token leakage.
2. Initial Baseline Loss Contract: Close to theoretical uniform entropy ln(4096) ~= 8.318.
3. Gradient & Parameter-Update Proof: Gradients finite/non-zero, parameters_before != parameters_after.
4. Checkpoint Save/Load Payload Contract: Model, optimizer, scheduler, RNG state preserved.
5. Deterministic Resumption Contract: Resumed training trajectory matches uninterrupted reference.
6. Bit-for-bit Deterministic Reproducibility: Identical seed produces identical loss sequence.
"""

import math
from pathlib import Path
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.training.seed import set_seed, get_rng_state, set_rng_state
from chakrview.training.dataset import StreamingTokenDataset
from chakrview.training.collator import CausalLanguageModelingCollator
from chakrview.training.loss import CausalLoss
from chakrview.training.checkpoint import CheckpointManager

ROOT_DIR = Path(__file__).resolve().parents[1]
STAGE_B_TOKENIZED = ROOT_DIR / "data" / "tokenized" / "stage_b"
TRAIN_SHARD_DIR = STAGE_B_TOKENIZED / "train"
VAL_SHARD_DIR = STAGE_B_TOKENIZED / "validation"


def test_stage_b_causal_target_shift_contract():
    """Verify next-token prediction targets are strictly shifted by 1 position."""
    dataset = StreamingTokenDataset(
        shard_dir=TRAIN_SHARD_DIR,
        sequence_length=64,
        loop=False,
        seed=42,
    )
    collator = CausalLanguageModelingCollator(max_context=512)
    loader = DataLoader(dataset, batch_size=4, collate_fn=collator)
    batch = next(iter(loader))

    inp = batch["input_ids"]
    tgt = batch["target_ids"]
    mask = batch["attention_mask"]

    assert inp.shape == (4, 64)
    assert tgt.shape == (4, 64)
    assert mask.shape == (4, 64)

    # Next-token prediction invariant: input[b, t+1] == target[b, t]
    assert torch.equal(inp[:, 1:], tgt[:, :-1])


def test_stage_b_initial_baseline_loss_theoretical_bounds():
    """Verify initial loss on randomly initialized weights matches theoretical ln(4096)."""
    set_seed(42)
    model = ChakrMicro(ModelConfig())
    loss_fn = CausalLoss(ignore_index=2)

    dataset = StreamingTokenDataset(
        shard_dir=TRAIN_SHARD_DIR,
        sequence_length=128,
        loop=False,
        seed=42,
    )
    collator = CausalLanguageModelingCollator(max_context=512)
    batch = next(iter(DataLoader(dataset, batch_size=2, collate_fn=collator)))

    model.eval()
    with torch.no_grad():
        logits = model(batch["input_ids"])
        loss = float(loss_fn(logits, batch["target_ids"]).item())

    theoretical_loss = math.log(4096)  # ~8.3178
    # Initial loss should be within 0.2 of theoretical uniform entropy
    assert abs(loss - theoretical_loss) < 0.2, (
        f"Initial loss {loss:.4f} deviates too far from theoretical ln(4096) = {theoretical_loss:.4f}"
    )


def test_stage_b_gradient_and_parameter_update_proof():
    """Verify that forward-backward computes valid gradients and optimizer updates weights."""
    set_seed(42)
    model = ChakrMicro(ModelConfig())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    loss_fn = CausalLoss(ignore_index=2)

    # Probe parameter: embedding weight
    emb_before = model.embedding.weight.clone().detach()
    w1_before = model.layers[0].ffn.gate_proj.weight.clone().detach()

    dataset = StreamingTokenDataset(
        shard_dir=TRAIN_SHARD_DIR,
        sequence_length=64,
        loop=False,
        seed=42,
    )
    batch = next(iter(DataLoader(dataset, batch_size=2, collate_fn=CausalLanguageModelingCollator(512))))

    model.train()
    optimizer.zero_grad()
    logits = model(batch["input_ids"])
    loss = loss_fn(logits, batch["target_ids"])
    loss.backward()

    # 1. Gradients must be finite and non-zero
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"Parameter {name} has None gradient"
            assert torch.isfinite(p.grad).all(), f"Parameter {name} has NaN/Inf gradient"

    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    assert float(grad_norm) > 0.0, "Total gradient norm was zero"

    # 2. Optimizer step must change parameters
    optimizer.step()

    emb_after = model.embedding.weight
    w1_after = model.layers[0].ffn.gate_proj.weight

    assert not torch.equal(emb_before, emb_after), "Embedding weights failed to update!"
    assert not torch.equal(w1_before, w1_after), "Feed-forward weights failed to update!"
    assert float(torch.norm(emb_after.detach() - emb_before)) > 0.0
    assert float(torch.norm(w1_after.detach() - w1_before)) > 0.0


def test_stage_b_checkpoint_save_load_contract(tmp_path: Path):
    """Verify CheckpointManager serializes and restores complete pre-training state."""
    set_seed(42)
    model = ChakrMicro(ModelConfig())
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100)
    mgr = CheckpointManager(checkpoint_dir=str(tmp_path), keep_last_n=2)

    saved_path = mgr.save(
        model=model,
        optimizer=opt,
        scheduler=sched,
        step=42,
        epoch=1,
        rng_state=get_rng_state(),
        train_metrics={"loss": 4.5},
    )

    assert saved_path.is_file()
    payload = CheckpointManager.load(saved_path)

    assert payload["step"] == 42
    assert payload["epoch"] == 1
    assert "model_state_dict" in payload
    assert "optimizer_state_dict" in payload
    assert "scheduler_state_dict" in payload
    assert "rng_state" in payload
    assert payload["train_metrics"]["loss"] == 4.5


def test_stage_b_deterministic_resumption(tmp_path: Path):
    """Verify resumed training trajectory exactly matches uninterrupted reference run."""
    seed = 777
    seq_len = 64
    batch_size = 2
    loss_fn = CausalLoss(ignore_index=2)
    collator = CausalLanguageModelingCollator(512)

    # 1. Reference run: 10 uninterrupted steps
    set_seed(seed)
    ref_model = ChakrMicro(ModelConfig())
    ref_opt = torch.optim.AdamW(ref_model.parameters(), lr=1e-3)
    ref_ds = StreamingTokenDataset(TRAIN_SHARD_DIR, sequence_length=seq_len, loop=True, seed=seed)
    ref_loader = iter(DataLoader(ref_ds, batch_size=batch_size, collate_fn=collator))

    ref_losses = []
    for _ in range(10):
        b = next(ref_loader)
        ref_opt.zero_grad()
        out = ref_model(b["input_ids"])
        l = loss_fn(out, b["target_ids"])
        l.backward()
        ref_opt.step()
        ref_losses.append(float(l.item()))

    # 2. Run A: 5 steps, save checkpoint
    set_seed(seed)
    run_a_model = ChakrMicro(ModelConfig())
    run_a_opt = torch.optim.AdamW(run_a_model.parameters(), lr=1e-3)
    run_a_ds = StreamingTokenDataset(TRAIN_SHARD_DIR, sequence_length=seq_len, loop=True, seed=seed)
    run_a_loader = iter(DataLoader(run_a_ds, batch_size=batch_size, collate_fn=collator))
    mgr = CheckpointManager(checkpoint_dir=str(tmp_path))

    for _ in range(5):
        b = next(run_a_loader)
        run_a_opt.zero_grad()
        out = run_a_model(b["input_ids"])
        l = loss_fn(out, b["target_ids"])
        l.backward()
        run_a_opt.step()

    ckpt_path = mgr.save(
        model=run_a_model,
        optimizer=run_a_opt,
        step=5,
        rng_state=get_rng_state(),
    )

    # 3. Run B: Restore checkpoint at step 5, train steps 6..10
    payload = CheckpointManager.load(ckpt_path)
    res_model = ChakrMicro(ModelConfig())
    res_model.load_state_dict(payload["model_state_dict"])
    res_opt = torch.optim.AdamW(res_model.parameters(), lr=1e-3)
    res_opt.load_state_dict(payload["optimizer_state_dict"])
    set_rng_state(payload["rng_state"])

    # Advance data loader by 5 steps
    run_b_ds = StreamingTokenDataset(TRAIN_SHARD_DIR, sequence_length=seq_len, loop=True, seed=seed)
    run_b_loader = iter(DataLoader(run_b_ds, batch_size=batch_size, collate_fn=collator))
    for _ in range(5):
        next(run_b_loader)

    res_losses = []
    for _ in range(5):
        b = next(run_b_loader)
        res_opt.zero_grad()
        out = res_model(b["input_ids"])
        l = loss_fn(out, b["target_ids"])
        l.backward()
        res_opt.step()
        res_losses.append(float(l.item()))

    # Compare steps 6..10
    ref_tail = ref_losses[5:]
    for r, b in zip(ref_tail, res_losses):
        assert abs(r - b) < 1e-5, f"Resumed trajectory drifted: ref={r} vs resumed={b}"


def test_stage_b_deterministic_reproducibility():
    """Verify two independent runs from seed 42 produce 100% bit-for-bit identical losses."""
    def run_session():
        set_seed(42)
        model = ChakrMicro(ModelConfig())
        opt = torch.optim.AdamW(model.parameters(), lr=5e-4)
        ds = StreamingTokenDataset(TRAIN_SHARD_DIR, sequence_length=64, loop=True, seed=42)
        loader = iter(DataLoader(ds, batch_size=2, collate_fn=CausalLanguageModelingCollator(512)))
        loss_fn = CausalLoss(ignore_index=2)
        losses = []
        for _ in range(5):
            b = next(loader)
            opt.zero_grad()
            out = model(b["input_ids"])
            l = loss_fn(out, b["target_ids"])
            l.backward()
            opt.step()
            losses.append(float(l.item()))
        return losses

    losses_1 = run_session()
    losses_2 = run_session()

    assert losses_1 == losses_2, f"Runs differed! Run 1: {losses_1} vs Run 2: {losses_2}"
