"""
UCIP-14C-R2: Three-Way Verification of Causal Target Alignment and Assistant-Only Loss.

Tests:
Method A: Standard Runtime Causal Shifting (HuggingFace / PyTorch convention: logits[:, :-1] vs targets[:, 1:]).
Method B: Pre-Shifted Next-Token Target Construction (ChakrView sft_engine convention: direct pairing).
Method C: Actual Model Forward & Gradient Propagation (Verifying zero gradient on prompt, active gradient on response & EOS).
Method D: Exact Equivalence Proof between Method A and Method B.
Method E: Multi-Turn Conversation Alignment under both conventions.
"""

import math
import sys
from pathlib import Path
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.tokenizer.serialization import load_tokenizer_artifacts
from chakrview.training.sft_engine import build_causal_sequence


@pytest.fixture(scope="module")
def tokenizer_artifacts():
    tok, _ = load_tokenizer_artifacts(ROOT / "data" / "experiments" / "vocab_4096")
    return tok


def test_01_method_a_standard_runtime_causal_shift(tokenizer_artifacts):
    """
    Method A: Standard Runtime Causal Shift (Unshifted sequence with runtime shift).
    Full sequence: [BOS] + prompt_tokens + response_tokens + [EOS] (length 9).
    Labels: [-100]*6 (BOS + prompt) + response_tokens + [EOS] (length 9).
    Shifted: shift_logits = logits[:, :-1] (length 8) vs shift_labels = labels[:, 1:] (length 8).
    """
    tok = tokenizer_artifacts
    p_str = "User: Hi"
    r_str = "Hello"
    p_toks = tok.encode(p_str)  # [88, 835, 324, 75, 108] (len 5)
    r_toks = tok.encode(r_str)  # [3860, 317] (len 2)
    
    # In standard unshifted representation:
    # index 0: BOS (0)
    # index 1..5: P0..P4 (88, 835, 324, 75, 108)
    # index 6..7: R0..R1 (3860, 317)
    # index 8: EOS (1)
    full_seq = [0] + p_toks + r_toks + [1]
    labels = ([-100] * (1 + len(p_toks))) + r_toks + [1]
    
    # Apply standard shift:
    # shift_seq (input to logits): full_seq[:-1] (length 8)
    # shift_labels: labels[1:] (length 8)
    shift_seq = full_seq[:-1]
    shift_labels = labels[1:]
    
    assert len(shift_seq) == 8
    assert len(shift_labels) == 8
    
    # Check shift_labels:
    # index 0..4 (corresponding to input BOS, P0, P1, P2, P3) -> -100
    assert shift_labels[:5] == [-100, -100, -100, -100, -100]
    
    # index 5 (corresponding to input P4 == 108) -> R0 == 3860
    assert shift_seq[5] == 108
    assert shift_labels[5] == 3860, "Logit position 5 (final prompt token) must predict R0 (3860)"
    
    # index 6 (corresponding to input R0 == 3860) -> R1 == 317
    assert shift_seq[6] == 3860
    assert shift_labels[6] == 317
    
    # index 7 (corresponding to input R1 == 317) -> EOS == 1
    assert shift_seq[7] == 317
    assert shift_labels[7] == 1, "Logit position 7 (final response token) must predict EOS (1)"


def test_02_method_b_preshifted_target_construction(tokenizer_artifacts):
    """
    Method B: Pre-Shifted Next-Token Target Construction (ChakrView sft_engine).
    build_causal_sequence constructs input_seq (len 8) and pre-shifted target_seq (len 8).
    Target at index i is already the token that should be predicted by input_seq[i]!
    """
    tok = tokenizer_artifacts
    p_str = "User: Hi"
    r_str = "Hello"
    p_toks = tok.encode(p_str)
    r_toks = tok.encode(r_str)
    
    inp, tgt = build_causal_sequence(p_toks, r_toks, bos_id=0, eos_id=1, ignore_index=-100, mask_prompt=True)
    
    assert len(inp) == 8
    assert len(tgt) == 8
    
    # Targets 0..4 are -100
    assert tgt[:5] == [-100, -100, -100, -100, -100]
    
    # Target 5 is R0 (3860), predicted by inp[5] == 108
    assert inp[5] == 108
    assert tgt[5] == 3860
    
    # Target 6 is R1 (317), predicted by inp[6] == 3860
    assert inp[6] == 3860
    assert tgt[6] == 317
    
    # Target 7 is EOS (1), predicted by inp[7] == 317
    assert inp[7] == 317
    assert tgt[7] == 1


def test_03_exact_mathematical_equivalence(tokenizer_artifacts):
    """
    Verify that Method A (runtime shift) and Method B (pre-shifted) produce
    100% bit-exact loss and 100% bit-exact logit gradients.
    """
    tok = tokenizer_artifacts
    p_toks = tok.encode("User: Hi")
    r_toks = tok.encode("Hello")
    
    # Method B tensors
    inp_b, tgt_b = build_causal_sequence(p_toks, r_toks, bos_id=0, eos_id=1, ignore_index=-100, mask_prompt=True)
    y_b = torch.tensor([tgt_b], dtype=torch.long)
    
    # Method A tensors
    full_seq_a = [0] + p_toks + r_toks + [1]
    labels_a = ([-100] * (1 + len(p_toks))) + r_toks + [1]
    y_a = torch.tensor([labels_a], dtype=torch.long)
    
    # Simulated model logits (shared random seed)
    torch.manual_seed(123)
    logits_b = torch.randn(1, 8, 4096, requires_grad=True)
    
    # Logits A has 9 positions; the first 8 are identical to logits_b
    torch.manual_seed(123)
    logits_a_base = torch.randn(1, 8, 4096)
    logits_a_extra = torch.randn(1, 1, 4096)
    logits_a = torch.cat([logits_a_base, logits_a_extra], dim=1).requires_grad_(True)
    
    # Loss B: Direct
    loss_b = F.cross_entropy(logits_b.view(-1, 4096), y_b.view(-1), ignore_index=-100)
    loss_b.backward()
    
    # Loss A: Runtime Shift
    shift_logits_a = logits_a[:, :-1, :]
    shift_labels_a = y_a[:, 1:]
    loss_a = F.cross_entropy(shift_logits_a.reshape(-1, 4096), shift_labels_a.reshape(-1), ignore_index=-100)
    loss_a.backward()
    
    # Assert exact equality
    assert torch.equal(shift_labels_a, y_b), "Shifted labels from Method A must be identical to target_seq from Method B"
    assert math.isclose(loss_a.item(), loss_b.item(), rel_tol=1e-7)
    assert torch.allclose(logits_b.grad, logits_a.grad[:, :-1, :], atol=1e-7)
    assert logits_a.grad[:, -1, :].abs().max().item() == 0.0, "Position 8 in Method A (extra EOS logit) receives 0 gradient"


def test_04_real_model_forward_gradient_isolation(tokenizer_artifacts):
    """
    Method C: Real ChakrMicro forward and backward pass.
    Confirms prompt positions produce 0.0 logit gradient, response & EOS produce active gradient,
    and padding produces 0.0 gradient.
    """
    tok = tokenizer_artifacts
    p_toks = tok.encode("User: Hi")
    r_toks = tok.encode("Hello")
    
    inp, tgt = build_causal_sequence(p_toks, r_toks, bos_id=0, eos_id=1, ignore_index=-100, mask_prompt=True)
    
    # Add 3 padding tokens
    pad_id = 2
    inp_padded = inp + [pad_id, pad_id, pad_id]
    tgt_padded = tgt + [-100, -100, -100]
    
    x = torch.tensor([inp_padded], dtype=torch.long)
    y = torch.tensor([tgt_padded], dtype=torch.long)
    
    model = ChakrMicro(ModelConfig(max_seq_len=512))
    model.train()
    model.zero_grad()
    
    logits = model(x)
    logits.retain_grad()
    
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    loss = loss_fn(logits.view(-1, 4096), y.view(-1))
    loss.backward()
    
    grad_norms = logits.grad.abs().sum(dim=-1)[0]
    
    # Prompt positions 0..4: zero gradient
    for i in range(len(p_toks)):
        assert grad_norms[i].item() == 0.0, f"Prompt pos {i} must have 0 gradient"
        
    # Active response positions 5..7: active gradient
    # Index 5: final prompt token predicting R0
    # Index 6: R0 predicting R1
    # Index 7: R1 predicting EOS
    for i in range(len(p_toks), len(p_toks) + len(r_toks) + 1):
        assert grad_norms[i].item() > 0.0, f"Response pos {i} must have positive gradient"
        
    # Padding positions 8..10: zero gradient
    for i in range(len(p_toks) + len(r_toks) + 1, len(inp_padded)):
        assert grad_norms[i].item() == 0.0, f"Padding pos {i} must have 0 gradient"


def test_05_multi_turn_causal_target_alignment(tokenizer_artifacts):
    """
    Method E: Multi-Turn Conversation Alignment under both conventions.
    """
    tok = tokenizer_artifacts
    prompt = "User: Name?\nAssistant: Alice\nUser: Repeat\nAssistant: "
    response = "Alice"
    p_toks = tok.encode(prompt)
    r_toks = tok.encode(response)
    
    # Method B (Pre-shifted)
    inp_b, tgt_b = build_causal_sequence(p_toks, r_toks, bos_id=0, eos_id=1, ignore_index=-100, mask_prompt=True)
    
    # Method A (Runtime shift)
    full_seq_a = [0] + p_toks + r_toks + [1]
    labels_a = ([-100] * (1 + len(p_toks))) + r_toks + [1]
    shift_labels_a = labels_a[1:]
    
    assert shift_labels_a == tgt_b
    assert tgt_b[:len(p_toks)] == [-100] * len(p_toks)
    assert tgt_b[len(p_toks)] == r_toks[0]
    assert tgt_b[-1] == 1
