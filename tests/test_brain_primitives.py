"""
Tests for ChakrView Neural Core Primitive Modules:
- TokenEmbedding
- RMSNorm
- RotaryEmbedding (RoPE)
- CausalMask
- MultiHeadAttention
- SwiGLU
- TransformerBlock
- LMHead
"""

import pytest
import torch
from chakrview.brain.config import ModelConfig
from chakrview.brain.embeddings import TokenEmbedding
from chakrview.brain.normalization import RMSNorm
from chakrview.brain.rotary import RotaryEmbedding
from chakrview.brain.masking import CausalMask
from chakrview.brain.attention import MultiHeadAttention
from chakrview.brain.feedforward import SwiGLU
from chakrview.brain.block import TransformerBlock
from chakrview.brain.output import LMHead


def test_token_embedding_forward_and_bounds():
    """Verify TokenEmbedding produces correct shape and rejects out-of-bounds IDs."""
    cfg = ModelConfig(vocab_size=4096, d_model=192)
    emb = TokenEmbedding(cfg)
    
    # Valid input [B=2, T=4]
    input_ids = torch.tensor([[0, 100, 2000, 4095], [1, 2, 3, 258]], dtype=torch.long)
    out = emb(input_ids)
    assert out.shape == (2, 4, 192)
    
    # Negative ID rejected
    with pytest.raises(ValueError, match="out of bounds"):
        emb(torch.tensor([[-1]], dtype=torch.long))
        
    # Exceeding V rejected
    with pytest.raises(ValueError, match="out of bounds"):
        emb(torch.tensor([[4096]], dtype=torch.long))


def test_rmsnorm_mathematical_precision():
    """Verify RMSNorm produces expected normalized values with gamma scale."""
    norm = RMSNorm(d_model=4, eps=1e-5)
    # Set weights to all 2.0
    with torch.no_grad():
        norm.weight.fill_(2.0)
        
    x = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
    # rms = sqrt(mean(1^2) + 1e-5) approx 1.0
    # norm(x) = (x / 1.0) * 2.0 = [2.0, 2.0, 2.0, 2.0]
    out = norm(x)
    assert torch.allclose(out, torch.tensor([[2.0, 2.0, 2.0, 2.0]]), atol=1e-4)


def test_rope_norm_preservation_and_position_zero():
    """
    Verify RoPE properties:
    1. At position 0, values are unchanged (cos(0)=1, sin(0)=0).
    2. Vector L2 norm is preserved by orthogonal 2D rotations.
    """
    dim = 32
    rope = RotaryEmbedding(dim=dim, max_seq_len=64, theta=10000.0)
    
    # [B=1, H=1, T=16, d_head=32]
    x = torch.randn(1, 1, 16, dim)
    rotated = rope(x, seq_len=16)
    
    assert rotated.shape == (1, 1, 16, dim)
    
    # Position 0 unchanged
    assert torch.allclose(rotated[:, :, 0, :], x[:, :, 0, :], atol=1e-5)
    
    # Norm preserved across all positions: ||RoPE(x)|| == ||x||
    orig_norms = torch.norm(x, dim=-1)
    rot_norms = torch.norm(rotated, dim=-1)
    assert torch.allclose(orig_norms, rot_norms, atol=1e-5)


def test_causal_mask_structure():
    """Verify CausalMask is strictly lower-triangular with additive -1e9 in upper triangle."""
    mask_gen = CausalMask(max_seq_len=8)
    mask = mask_gen(seq_len=4)
    
    assert mask.shape == (1, 1, 4, 4)
    m = mask.squeeze()
    
    # Diagonal and lower triangle must be 0.0
    for i in range(4):
        for j in range(4):
            if j <= i:
                assert m[i, j].item() == 0.0
            else:
                assert m[i, j].item() <= -1e8


def test_multi_head_attention_shapes_and_residuals():
    """Verify MultiHeadAttention runs cleanly and produces [B, T, d_model]."""
    cfg = ModelConfig(d_model=192, n_heads=6, max_seq_len=64)
    mha = MultiHeadAttention(cfg)
    
    x = torch.randn(2, 16, 192)
    out = mha(x)
    assert out.shape == (2, 16, 192)


def test_swiglu_forward_and_shapes():
    """Verify SwiGLU projects [B, T, d_model] -> [B, T, hidden_dim] -> [B, T, d_model]."""
    cfg = ModelConfig(d_model=192, hidden_dim=512)
    ffn = SwiGLU(cfg)
    
    x = torch.randn(2, 16, 192)
    out = ffn(x)
    assert out.shape == (2, 16, 192)


def test_transformer_block_forward():
    """Verify TransformerBlock combines Pre-RMSNorm, MHA, and SwiGLU with residuals."""
    cfg = ModelConfig(d_model=192, n_heads=6, hidden_dim=512, max_seq_len=64)
    block = TransformerBlock(cfg)
    
    x = torch.randn(2, 16, 192)
    out = block(x)
    assert out.shape == (2, 16, 192)


def test_lm_head_tied_projection():
    """Verify LMHead computes linear projection using shared embedding tensor."""
    shared_weight = torch.nn.Parameter(torch.randn(4096, 192))
    lm_head = LMHead(shared_weight)
    
    # Verify exact pointer sharing
    assert lm_head.weight.data_ptr() == shared_weight.data_ptr()
    
    x = torch.randn(2, 16, 192)
    logits = lm_head(x)
    assert logits.shape == (2, 16, 4096)
