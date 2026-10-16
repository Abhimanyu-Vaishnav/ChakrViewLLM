"""
Targeted verification tests for causal masking, loss correctness, and checkpoint fidelity.
Location: tests/test_batch1341_1380_lm_evaluation.py
"""
import math
import pytest
import torch
import torch.nn.functional as F
from pathlib import Path

from chakrview.brain.model import ChakrMicro
from chakrview.brain.config import ModelConfig
from chakrview.tokenizer.serialization import load_tokenizer_artifacts

TOKENIZER_DIR = Path("release/chakrview-v0.5.0/data/experiments/vocab_4096")
CANONICAL_CKPT = Path("experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt")


@pytest.fixture(scope="module")
def model_and_tokenizer():
    tokenizer, _ = load_tokenizer_artifacts(TOKENIZER_DIR)
    ckpt = torch.load(CANONICAL_CKPT, map_location="cpu", weights_only=False)
    model = ChakrMicro(ckpt["config"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, tokenizer, ckpt["config"]


def test_checkpoint_integrity_and_parameter_count(model_and_tokenizer):
    """Verify parameter count and structural invariants of canonical checkpoint."""
    model, _, config = model_and_tokenizer
    total_params = sum(p.numel() for p in model.parameters())
    assert config.vocab_size == 4096
    assert config.d_model == 192
    assert config.n_layers == 6
    # When tie_embeddings=True, embedding.weight and lm_head share the exact tensor,
    # yielding 3,443,136 unique parameters in model.parameters().
    assert total_params == 3443136


def test_strictly_causal_masking_no_future_leakage(model_and_tokenizer):
    """Verify that predictions at position t are invariant to tokens at positions > t."""
    model, _, _ = model_and_tokenizer
    
    # Construct two sequences with identical prefixes up to pos 3, but different tokens at pos 4
    seq1 = torch.tensor([[10, 20, 30, 40, 50]], dtype=torch.long)
    seq2 = torch.tensor([[10, 20, 30, 40, 99]], dtype=torch.long)
    
    with torch.no_grad():
        logits1 = model(seq1)
        logits2 = model(seq2)
        
    # Positions 0, 1, 2, 3 must be strictly identical across seq1 and seq2
    diff_prefix = (logits1[0, :4, :] - logits2[0, :4, :]).abs().max().item()
    assert diff_prefix == 0.0, f"Future token leakage detected! Difference: {diff_prefix}"
    
    # Position 4 can differ because it incorporates token at pos 4
    diff_pos4 = (logits1[0, 4, :] - logits2[0, 4, :]).abs().max().item()
    assert diff_pos4 > 0.0, "Position 4 unexpectedly identical despite token modification"


def test_causal_target_alignment_off_by_one(model_and_tokenizer):
    """Verify off-by-one target alignment: input x[:-1] predicts target x[1:]."""
    model, _, _ = model_and_tokenizer
    tokens = [100, 200, 300, 400]
    
    input_ids = torch.tensor([tokens[:-1]], dtype=torch.long) # [100, 200, 300]
    targets = torch.tensor(tokens[1:], dtype=torch.long)      # [200, 300, 400]
    
    with torch.no_grad():
        logits = model(input_ids) # [1, 3, 4096]
        
    assert logits.shape == (1, 3, 4096)
    # Logit at index 0 corresponds to predicting tokens[1] (200) given tokens[0] (100)
    # Logit at index 2 corresponds to predicting tokens[3] (400) given tokens[:3]
    loss = F.cross_entropy(logits[0], targets)
    assert torch.isfinite(loss)
    assert loss.item() > 0.0


def test_deterministic_eval_mode(model_and_tokenizer):
    """Verify that forward pass is deterministic in eval mode across multiple calls."""
    model, _, _ = model_and_tokenizer
    input_ids = torch.tensor([[50, 60, 70, 80]], dtype=torch.long)
    
    with torch.no_grad():
        out1 = model(input_ids)
        out2 = model(input_ids)
        
    assert torch.equal(out1, out2), "Model output is non-deterministic in eval mode"


def test_uniform_baseline_theoretical_loss():
    """Verify theoretical loss of uniform random baseline over 4096 vocabulary."""
    vocab_size = 4096
    expected_loss = math.log(vocab_size)
    assert abs(expected_loss - 8.317766) < 1e-4
    assert math.isclose(math.exp(expected_loss), 4096.0, rel_tol=1e-7)


def test_finite_logits_on_context_window_boundary(model_and_tokenizer):
    """Verify that model executes reliably up to maximum context window limit (512 tokens)."""
    model, _, config = model_and_tokenizer
    max_len = config.max_seq_len
    
    long_seq = torch.randint(low=3, high=4000, size=(1, max_len), dtype=torch.long)
    with torch.no_grad():
        logits = model(long_seq)
        
    assert logits.shape == (1, max_len, config.vocab_size)
    assert torch.isfinite(logits).all().item()
    
    # Beyond max_seq_len must raise ValueError
    overflow_seq = torch.randint(low=3, high=4000, size=(1, max_len + 1), dtype=torch.long)
    with pytest.raises(ValueError):
        model(overflow_seq)
