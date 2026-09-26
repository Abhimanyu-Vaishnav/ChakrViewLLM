"""
Tests for Optimizer Construction and LR Scheduler (Phase 8 & 18).
"""

import torch
from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.training.config import TrainingHyperparameters
from chakrview.training.optimizer import (
    build_optimizer,
    build_lr_scheduler,
    get_parameter_groups,
)


def test_optimizer_parameter_group_segregation_and_no_duplicates():
    """Verify 2D weights get decay, 1D norms get zero decay, and tied parameters are not duplicated."""
    cfg = ModelConfig()
    model = ChakrMicro(cfg)

    groups = get_parameter_groups(model, weight_decay=0.02)
    assert len(groups) == 2
    assert groups[0]["weight_decay"] == 0.02
    assert groups[1]["weight_decay"] == 0.0

    # Total parameters across both groups must match unique trainable parameters (3,443,136)
    total_in_groups = sum(p.numel() for g in groups for p in g["params"])
    assert total_in_groups == 3443136

    # Verify tied embedding/output head appears only once
    emb_weight = model.embedding.weight
    emb_count = sum(1 for g in groups for p in g["params"] if p is emb_weight)
    assert emb_count == 1


def test_cosine_lr_scheduler_warmup_and_decay():
    """Verify learning rate linearly warms up and cosine-decays."""
    cfg = ModelConfig()
    model = ChakrMicro(cfg)
    hypers = TrainingHyperparameters(
        learning_rate=1e-3,
        min_learning_rate=1e-4,
        warmup_steps=10,
        max_steps=100,
    )

    optimizer = build_optimizer(model, hypers)
    scheduler = build_lr_scheduler(optimizer, hypers)

    # Step 0: Warmup start
    lr_0 = optimizer.param_groups[0]["lr"]
    assert lr_0 == 0.0  # Warmup starts at 0.0

    # Advance 5 steps: mid-warmup
    for _ in range(5):
        optimizer.step()
        scheduler.step()
    lr_mid_warm = optimizer.param_groups[0]["lr"]
    assert 0.0 < lr_mid_warm < 1e-3

    # Advance to step 10: peak LR
    for _ in range(5):
        optimizer.step()
        scheduler.step()
    lr_peak = optimizer.param_groups[0]["lr"]
    assert abs(lr_peak - 1e-3) < 1e-5

    # Advance to step 100: decay down to min_learning_rate
    for _ in range(90):
        optimizer.step()
        scheduler.step()
    lr_final = optimizer.param_groups[0]["lr"]
    assert abs(lr_final - 1e-4) < 1e-5
