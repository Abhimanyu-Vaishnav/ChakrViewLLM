"""
Tests for Seed Utility and RNG State Restoration (Phase 3 & 18).
"""

import random
import numpy as np
import torch

from chakrview.training.seed import set_seed, get_rng_state, set_rng_state


def test_seed_determinism_across_libraries():
    """Verify set_seed ensures reproducible draws from random, numpy, and torch."""
    set_seed(12345)
    r1 = random.random()
    n1 = np.random.randn(3)
    t1 = torch.randn(3)

    set_seed(12345)
    r2 = random.random()
    n2 = np.random.randn(3)
    t2 = torch.randn(3)

    assert r1 == r2
    assert np.allclose(n1, n2)
    assert torch.allclose(t1, t2)


def test_rng_state_capture_and_restoration():
    """Verify capturing and restoring RNG state maintains identical trajectories."""
    set_seed(999)
    # Advance state
    _ = [random.random() for _ in range(10)]
    _ = np.random.randn(5)
    _ = torch.randn(5)

    captured_state = get_rng_state()

    # Draw next numbers in Run 1
    next_r1 = random.random()
    next_n1 = np.random.randn(3)
    next_t1 = torch.randn(3)

    # Scramble state
    set_seed(111)
    _ = random.random()

    # Restore captured state
    set_rng_state(captured_state)

    # Draw next numbers in Run 2
    next_r2 = random.random()
    next_n2 = np.random.randn(3)
    next_t2 = torch.randn(3)

    assert next_r1 == next_r2
    assert np.allclose(next_n1, next_n2)
    assert torch.allclose(next_t1, next_t2)
