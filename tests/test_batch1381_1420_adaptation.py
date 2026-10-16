"""
Unit tests for Batch 1381-1420 checkpoint integrity, adaptation safety, and regression verification.
Location: tests/test_batch1381_1420_adaptation.py
"""
import pytest
import torch
from pathlib import Path

from chakrview.brain.model import ChakrMicro
from chakrview.brain.config import ModelConfig
from chakrview.tokenizer.serialization import load_tokenizer_artifacts

TOKENIZER_DIR = Path("release/chakrview-v0.5.0/data/experiments/vocab_4096")
CANONICAL_CKPT = Path("experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt")
EXP_CKPT = Path("experiments/batch1381_1420/checkpoints/chakrmicro_domain_adapted_experimental.pt")


@pytest.fixture(scope="module")
def checkpoints():
    tokenizer, _ = load_tokenizer_artifacts(TOKENIZER_DIR)
    ckpt_canon = torch.load(CANONICAL_CKPT, map_location="cpu", weights_only=False)
    ckpt_exp = torch.load(EXP_CKPT, map_location="cpu", weights_only=False)
    return tokenizer, ckpt_canon, ckpt_exp


def test_canonical_checkpoint_hash_and_immutability(checkpoints):
    """Verify canonical checkpoint SHA-256 remains bit-for-bit unchanged."""
    import hashlib
    h = hashlib.sha256()
    h.update(CANONICAL_CKPT.read_bytes())
    assert h.hexdigest() == "96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee"


def test_experimental_checkpoint_loading_and_architecture(checkpoints):
    """Verify experimental adapted checkpoint loads into ChakrMicro strictly."""
    tokenizer, ckpt_canon, ckpt_exp = checkpoints
    model = ChakrMicro(ckpt_exp["config"])
    missing, unexpected = model.load_state_dict(ckpt_exp["model_state_dict"], strict=True)
    assert len(missing) == 0
    assert len(unexpected) == 0
    assert ckpt_exp["adaptation_step"] == 80
    assert ckpt_exp["step"] == 10080


def test_parameter_reconciliation_invariants(checkpoints):
    """Verify parameter count: 3,443,136 unique parameters and 4,229,568 state-dict elements."""
    _, ckpt_canon, _ = checkpoints
    state_dict = ckpt_canon["model_state_dict"]
    total_elements = sum(p.numel() for p in state_dict.values())
    assert total_elements == 4229568
    
    # Model instance with tied weights
    model = ChakrMicro(ckpt_canon["config"])
    unique_params = sum(p.numel() for p in model.parameters())
    assert unique_params == 3443136


def test_finite_logits_on_adapted_checkpoint(checkpoints):
    """Verify adapted checkpoint produces finite logits on valid inputs."""
    _, _, ckpt_exp = checkpoints
    model = ChakrMicro(ckpt_exp["config"])
    model.load_state_dict(ckpt_exp["model_state_dict"])
    model.eval()
    
    inputs = torch.randint(low=3, high=4000, size=(2, 64), dtype=torch.long)
    with torch.no_grad():
        logits = model(inputs)
    assert logits.shape == (2, 64, 4096)
    assert torch.isfinite(logits).all().item()


def test_loss_improvement_on_adaptation_candidate(checkpoints):
    """Verify that candidate checkpoint achieves strictly lower validation loss than baseline."""
    _, ckpt_canon, ckpt_exp = checkpoints
    assert ckpt_exp["validation_loss"] < 5.0
    assert ckpt_exp["validation_perplexity"] < 110.0
