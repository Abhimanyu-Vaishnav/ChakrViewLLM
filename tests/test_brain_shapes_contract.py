"""
Strict Shape Contract Tests for ChakrMicro v0.1 (Phase 4).

Verifies exact tensor dimensions for every major transition across diverse
batch sizes (B in {1, 2, 4}) and sequence lengths (T in {1, 16, 128, 512}):
- input_ids: [B, T]
- embedding: [B, T, 192]
- attention Q/K/V: [B, H, T, head_dim]
- attention output: [B, T, 192]
- feed-forward output: [B, T, 192]
- final hidden state: [B, T, 192]
- logits: [B, T, 4096]
- head_dim * num_heads == d_model invariant
"""

import pytest
import torch
from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro


def test_head_dim_num_heads_product_invariant():
    """Verify that head_dim * num_heads strictly equals d_model."""
    cfg = ModelConfig()
    assert cfg.head_dim * cfg.n_heads == cfg.d_model
    assert cfg.head_dim == 32
    assert cfg.n_heads == 6
    assert cfg.d_model == 192


@pytest.mark.parametrize("batch_size", [1, 2, 4])
@pytest.mark.parametrize("seq_len", [1, 16, 128, 512])
def test_all_major_tensor_transitions_shapes(batch_size: int, seq_len: int):
    """
    Verify shapes at every individual layer and intermediate projection:
    Input [B, T] -> Embedding [B, T, 192] -> Q/K/V [B, 6, T, 32] ->
    Attn Out [B, T, 192] -> FFN Out [B, T, 192] -> Final Hidden [B, T, 192] ->
    Logits [B, T, 4096].
    """
    cfg = ModelConfig()
    model = ChakrMicro(cfg)
    model.eval()
    
    B, T = batch_size, seq_len
    input_ids = torch.randint(0, cfg.vocab_size, (B, T), dtype=torch.long)
    assert input_ids.shape == (B, T)
    
    with torch.no_grad():
        # 1. Embedding transition
        emb = model.embedding(input_ids)
        assert emb.shape == (B, T, 192)
        
        # 2. Block 0 Attention Q, K, V transitions
        block0 = model.layers[0]
        norm_x = block0.norm_1(emb)
        assert norm_x.shape == (B, T, 192)
        
        q = block0.attn.q_proj(norm_x).view(B, T, cfg.n_heads, cfg.head_dim).transpose(1, 2)
        k = block0.attn.k_proj(norm_x).view(B, T, cfg.n_heads, cfg.head_dim).transpose(1, 2)
        v = block0.attn.v_proj(norm_x).view(B, T, cfg.n_heads, cfg.head_dim).transpose(1, 2)
        assert q.shape == (B, 6, T, 32)
        assert k.shape == (B, 6, T, 32)
        assert v.shape == (B, 6, T, 32)
        
        # RoPE shape
        q_rot = block0.attn.rotary(q, T)
        k_rot = block0.attn.rotary(k, T)
        assert q_rot.shape == (B, 6, T, 32)
        assert k_rot.shape == (B, 6, T, 32)
        
        # Attention score and context shapes
        attn_out = block0.attn(norm_x)
        assert attn_out.shape == (B, T, 192)
        
        # Attention residual
        x_post_attn = emb + attn_out
        assert x_post_attn.shape == (B, T, 192)
        
        # 3. Feed-forward (SwiGLU) transition
        norm_ffn = block0.norm_2(x_post_attn)
        ffn_out = block0.ffn(norm_ffn)
        assert ffn_out.shape == (B, T, 192)
        
        # Full block output
        block_out = block0(emb)
        assert block_out.shape == (B, T, 192)
        
        # 4. End-to-end forward pass
        x = emb
        for layer in model.layers:
            x = layer(x)
        assert x.shape == (B, T, 192)
        
        # 5. Final RMSNorm (final hidden state)
        final_hidden = model.final_norm(x)
        assert final_hidden.shape == (B, T, 192)
        
        # 6. Final Logits
        logits = model.lm_head(final_hidden)
        assert logits.shape == (B, T, 4096)


@pytest.mark.parametrize(
    "B, T",
    [
        (1, 1),
        (1, 16),
        (2, 32),
        (2, 128),
        (1, 512),
    ],
)
def test_priority_5_shape_combinations_and_transitions(B: int, T: int):
    """
    Priority 5 required combinations:
    - B=1, T=1
    - B=1, T=16
    - B=2, T=32
    - B=2, T=128
    - B=1, T=512
    Verifies:
      input_ids: [B, T]
      embedding output: [B, T, 192]
      attention output: [B, T, 192]
      final hidden state: [B, T, 192]
      logits: [B, T, 4096]
    """
    cfg = ModelConfig()
    model = ChakrMicro(cfg)
    model.eval()

    input_ids = torch.randint(0, cfg.vocab_size, (B, T), dtype=torch.long)
    assert input_ids.shape == (B, T)

    with torch.no_grad():
        emb = model.embedding(input_ids)
        assert emb.shape == (B, T, 192)

        # Attention output from first block
        attn_out = model.layers[0].attn(model.layers[0].norm_1(emb))
        assert attn_out.shape == (B, T, 192)

        # Final hidden state
        x = emb
        for layer in model.layers:
            x = layer(x)
        final_hidden = model.final_norm(x)
        assert final_hidden.shape == (B, T, 192)

        # Logits
        logits = model.lm_head(final_hidden)
        assert logits.shape == (B, T, 4096)

        # End-to-end forward call
        logits_direct = model(input_ids)
        assert logits_direct.shape == (B, T, 4096)
        assert torch.equal(logits, logits_direct)


def test_rejection_of_sequences_exceeding_max_context():
    """Verify that sequences T > 512 are rejected with ValueError and not silently truncated."""
    cfg = ModelConfig()
    model = ChakrMicro(cfg)
    model.eval()

    exceeded_input = torch.randint(0, cfg.vocab_size, (1, 513), dtype=torch.long)
    with pytest.raises(ValueError, match="exceeds maximum context window"):
        _ = model(exceeded_input)

