"""
Causality test for ChakrMicro (Phase 7).

Proves that future tokens cannot influence earlier representations:
Construct Sequence A and Sequence B where a future token is changed at index k.
Verify that logits for positions 0 .. k-1 remain bit-exact or within float32 tolerance.
"""

import torch
from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro


def test_strict_mathematical_causality():
    """
    Construct two sequences A and B:
    - Sequence A: [t0, t1, t2, t3, t4, t5, t6, t7]
    - Sequence B: [t0, t1, t2, t3, ALTERED, ALTERED, ALTERED, ALTERED]
    
    Index of divergence: k = 4.
    Positions 0, 1, 2, 3 must have identical logits between A and B.
    Positions 4, 5, 6, 7 should differ because future/current inputs changed.
    """
    cfg = ModelConfig()
    model = ChakrMicro(cfg)
    model.eval()
    
    # Base sequence of length 8
    seq_A = torch.tensor([[100, 250, 500, 750, 1000, 1250, 1500, 1750]], dtype=torch.long)
    # Alter positions 4, 5, 6, 7
    seq_B = torch.tensor([[100, 250, 500, 750, 3000, 3250, 3500, 3750]], dtype=torch.long)
    
    with torch.no_grad():
        logits_A = model(seq_A)  # [1, 8, 4096]
        logits_B = model(seq_B)  # [1, 8, 4096]
        
    # Check positions 0, 1, 2, 3 (before index 4)
    # Must be identical within tight numerical tolerance (max difference < 1e-6)
    diff_prefix = torch.max(torch.abs(logits_A[:, :4, :] - logits_B[:, :4, :])).item()
    assert diff_prefix < 1e-6, f"Causality violation! Future tokens leaked into prefix: max diff = {diff_prefix}"
    
    # Check position 4 and beyond
    # Must differ
    diff_suffix = torch.max(torch.abs(logits_A[:, 4:, :] - logits_B[:, 4:, :])).item()
    assert diff_suffix > 1e-3, f"Expected altered tokens to change suffix representations, but got diff = {diff_suffix}"


def test_single_token_future_perturbation_causality():
    """
    Test across multiple sequence lengths that altering ONLY the last token
    leaves all preceding token logits completely unchanged.
    """
    cfg = ModelConfig()
    model = ChakrMicro(cfg)
    model.eval()
    
    for length in [4, 16, 64]:
        base = torch.randint(0, 4096, (1, length), dtype=torch.long)
        perturbed = base.clone()
        # Change only the last token
        perturbed[0, -1] = (perturbed[0, -1] + 1) % 4096
        
        with torch.no_grad():
            out_base = model(base)
            out_perturbed = model(perturbed)
            
        max_prefix_diff = torch.max(
            torch.abs(out_base[:, :-1, :] - out_perturbed[:, :-1, :])
        ).item()
        
        assert max_prefix_diff < 1e-6, (
            f"Future token leakage at length {length}! Preceding logits differed by {max_prefix_diff}"
        )


def test_four_token_explicit_causality_abcd():
    """
    Explicit test for sequence [A, B, C, D]:
    Changing token D to D' must NOT change the logits for positions A, B, C.
    Verifies isolation across attention mask, residual paths, normalization, and RoPE.
    """
    torch.manual_seed(42)
    cfg = ModelConfig()
    model = ChakrMicro(cfg)
    model.eval()
    
    # [A, B, C, D]
    seq1 = torch.tensor([[10, 20, 30, 40]], dtype=torch.long)
    # [A, B, C, D']
    seq2 = torch.tensor([[10, 20, 30, 999]], dtype=torch.long)
    
    with torch.no_grad():
        out1 = model(seq1)
        out2 = model(seq2)
        
    # Positions 0 (A), 1 (B), 2 (C) must be identical
    max_diff_abc = torch.max(torch.abs(out1[:, :3, :] - out2[:, :3, :])).item()
    assert max_diff_abc < 1e-6, f"Leakage detected! Logits for A, B, C changed by {max_diff_abc}"
    
    # Position 3 (D vs D') must differ
    diff_d = torch.max(torch.abs(out1[:, 3, :] - out2[:, 3, :])).item()
    assert diff_d > 1e-2, f"Expected position D to differ, but got diff {diff_d}"
