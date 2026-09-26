"""
Strict Causality Verification for ChakrMicro Neural Core (Phase 5).

Mathematically and empirically proves that:
1. The causal attention mask is strictly lower-triangular (zeros on and below diagonal, large negative values strictly above).
2. Token t cannot access any future token t + 1 .. T-1.
3. For sequence [A, B, C, D], changing only D guarantees that hidden states and logits for [A, B, C]
   remain identical within floating-point tolerance (tolerance <= 1e-6).
4. Causality holds at every individual transformer block layer, not merely at the final output head.
5. Batch independence holds without cross-batch or future-token leakage.
"""

import pytest
import torch
import torch.nn.functional as F

from chakrview.brain.config import ModelConfig
from chakrview.brain.masking import CausalMask
from chakrview.brain.model import ChakrMicro


def test_causal_mask_structure():
    """
    Verify the raw causal mask generator:
    - Diagonal and lower triangle must be exactly 0.0 (unmasked).
    - Strictly upper triangle must be a large negative value (-1e9) to zero out post-softmax.
    - Mask shape must match (1, 1, T, T).
    """
    T = 8
    causal_mask_module = CausalMask(max_seq_len=512)
    mask = causal_mask_module(T)
    assert mask.shape == (1, 1, T, T), f"Expected mask shape (1, 1, {T}, {T}), got {mask.shape}"
    
    m = mask[0, 0]
    for i in range(T):
        for j in range(T):
            if j <= i:
                assert m[i, j].item() == 0.0, f"Expected unmasked position ({i}, {j}) to be 0.0, got {m[i, j].item()}"
            else:
                assert m[i, j].item() <= -1e4, f"Expected masked future position ({i}, {j}) to be <= -10000, got {m[i, j].item()}"
                
    # Also verify softmax on row i: probabilities for j > i must be identically zero
    logits = torch.zeros(1, 1, T, T, dtype=torch.float32) + mask
    probs = F.softmax(logits, dim=-1)
    for i in range(T):
        for j in range(i + 1, T):
            assert probs[0, 0, i, j].item() == 0.0, f"Post-softmax probability for future token at ({i}, {j}) is non-zero: {probs[0, 0, i, j].item()}"


def test_strict_abcd_causality():
    """
    Standard [A, B, C, D] deterministic causality test:
    Given:
      Seq 1: [A, B, C, D]
      Seq 2: [A, B, C, D'] where D' != D
    Verify:
      - Logits for A, B, C are identical within float32 tolerance (<= 1e-6).
      - Logit for position D differs significantly (> 1e-3).
    """
    torch.manual_seed(42)
    cfg = ModelConfig()
    model = ChakrMicro(cfg)
    model.eval()

    # Tokens: A=45, B=120, C=890, D=2040, D'=3100
    seq1 = torch.tensor([[45, 120, 890, 2040]], dtype=torch.long)
    seq2 = torch.tensor([[45, 120, 890, 3100]], dtype=torch.long)

    with torch.no_grad():
        out1 = model(seq1)  # [1, 4, 4096]
        out2 = model(seq2)  # [1, 4, 4096]

    # Verify positions 0, 1, 2 (A, B, C)
    diff_prefix = torch.max(torch.abs(out1[:, :3, :] - out2[:, :3, :])).item()
    assert diff_prefix < 1e-6, (
        f"Strict causality violation! Changing token D altered logits for [A, B, C] by {diff_prefix:.8e}"
    )

    # Verify position 3 (D vs D')
    diff_d = torch.max(torch.abs(out1[:, 3, :] - out2[:, 3, :])).item()
    assert diff_d > 1e-2, (
        f"Expected output at position D to change when token was perturbed, but diff was {diff_d:.8e}"
    )


def test_layerwise_causality_isolation():
    """
    Verify that causality holds at every single TransformerBlock layer,
    proving that no residual connection, normalization layer, or attention head leaks future tokens.
    """
    torch.manual_seed(12345)
    cfg = ModelConfig()
    model = ChakrMicro(cfg)
    model.eval()

    seq1 = torch.tensor([[100, 200, 300, 400, 500]], dtype=torch.long)
    seq2 = torch.tensor([[100, 200, 300, 400, 999]], dtype=torch.long)

    with torch.no_grad():
        # Embedding
        h1 = model.embedding(seq1)
        h2 = model.embedding(seq2)
        # Positions 0..3 identical in embedding
        assert torch.max(torch.abs(h1[:, :4, :] - h2[:, :4, :])).item() == 0.0

        # Step through each block in model.layers
        for layer_idx, block in enumerate(model.layers):
            h1 = block(h1)
            h2 = block(h2)
            
            diff_prefix = torch.max(torch.abs(h1[:, :4, :] - h2[:, :4, :])).item()
            assert diff_prefix < 1e-6, (
                f"Causality leak detected at block {layer_idx}! Prefix max diff = {diff_prefix:.8e}"
            )
            
            diff_d = torch.max(torch.abs(h1[:, 4, :] - h2[:, 4, :])).item()
            assert diff_d > 1e-3, (
                f"Block {layer_idx} failed to register difference at perturbed token position: diff = {diff_d:.8e}"
            )


def test_systematic_single_token_future_perturbation():
    """
    Sweep through multiple sequence lengths and perturb only token t.
    Verify that all positions 0 .. t-1 remain unchanged.
    """
    torch.manual_seed(999)
    cfg = ModelConfig()
    model = ChakrMicro(cfg)
    model.eval()

    for seq_len in [4, 16, 64]:
        base_seq = torch.randint(0, cfg.vocab_size, (1, seq_len), dtype=torch.long)
        
        # Perturb token at the midpoint
        mid = seq_len // 2
        perturbed_seq = base_seq.clone()
        perturbed_seq[0, mid] = (perturbed_seq[0, mid] + 500) % cfg.vocab_size
        
        with torch.no_grad():
            logits_base = model(base_seq)
            logits_pert = model(perturbed_seq)
            
        # Logits at positions 0 .. mid-1 MUST be identical
        prefix_diff = torch.max(torch.abs(logits_base[:, :mid, :] - logits_pert[:, :mid, :])).item()
        assert prefix_diff < 1e-6, (
            f"Future token leakage at seq_len={seq_len}, mid={mid}! Max diff={prefix_diff:.8e}"
        )
        
        # Logits at position mid MUST differ
        at_mid_diff = torch.max(torch.abs(logits_base[:, mid, :] - logits_pert[:, mid, :])).item()
        assert at_mid_diff > 1e-3, (
            f"Expected difference at mid={mid}, got {at_mid_diff:.8e}"
        )


def test_batch_causality_and_sample_independence():
    """
    Verify that in batch size B=2:
    - Altering sequence 1 does NOT affect sequence 0.
    - Causality is maintained across both items simultaneously.
    """
    torch.manual_seed(777)
    cfg = ModelConfig()
    model = ChakrMicro(cfg)
    model.eval()

    batch_orig = torch.tensor([
        [10, 20, 30, 40, 50],
        [100, 200, 300, 400, 500]
    ], dtype=torch.long)

    # In batch_mod, keep sample 0 identical, but change sample 1's last token
    batch_mod = torch.tensor([
        [10, 20, 30, 40, 50],
        [100, 200, 300, 400, 999]
    ], dtype=torch.long)

    with torch.no_grad():
        out_orig = model(batch_orig)
        out_mod = model(batch_mod)

    # Sample 0 must be 100% bit-exact across all positions
    sample0_diff = torch.max(torch.abs(out_orig[0] - out_mod[0])).item()
    assert sample0_diff == 0.0, f"Cross-batch contamination detected! Sample 0 diff = {sample0_diff}"

    # Sample 1 prefix (positions 0..3) must be identical
    sample1_prefix_diff = torch.max(torch.abs(out_orig[1, :4, :] - out_mod[1, :4, :])).item()
    assert sample1_prefix_diff < 1e-6, f"Causality leak in batch sample 1: diff = {sample1_prefix_diff}"

    # Sample 1 position 4 must differ
    sample1_pos4_diff = torch.max(torch.abs(out_orig[1, 4, :] - out_mod[1, 4, :])).item()
    assert sample1_pos4_diff > 1e-3, f"Sample 1 position 4 expected diff, got {sample1_pos4_diff}"
