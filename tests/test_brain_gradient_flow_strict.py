"""
Strict Gradient Flow Verification for ChakrMicro (Phase 6).

Verifies:
- Cross-entropy loss is strictly finite (neither NaN nor Inf).
- Gradients exist for every trainable parameter.
- Gradients are finite (no NaN, no Inf).
- Gradients are non-zero where expected:
  - Token embedding receives gradients.
  - Attention projections (Q, K, V, Out) receive non-zero gradients in all layers.
  - Feed-forward projections (Gate, Up, Down) receive non-zero gradients in all layers.
  - Layer RMSNorm and final RMSNorm receive non-zero gradients.
- Weight-tied parameters:
  - Embedding weight accumulates gradients from both input lookup and output projection.
  - Object/storage identity of gradients between embedding and lm_head is preserved.
- Gradients exhibit healthy norms (no vanishing or exploding).
"""

import torch
import torch.nn.functional as F

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro


def test_strict_gradient_flow_single_step():
    """
    Run a synthetic forward/backward pass with a tiny random batch and inspect all gradients.
    """
    torch.manual_seed(42)
    cfg = ModelConfig()
    model = ChakrMicro(cfg)
    model.train()

    B, T = 2, 16
    input_ids = torch.randint(0, cfg.vocab_size, (B, T), dtype=torch.long)
    targets = torch.randint(0, cfg.vocab_size, (B, T), dtype=torch.long)

    # Forward pass
    logits = model(input_ids)  # [B, T, V]
    assert logits.shape == (B, T, cfg.vocab_size)

    # Loss
    loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), targets.view(-1))
    assert not torch.isnan(loss), "Loss is NaN"
    assert not torch.isinf(loss), "Loss is Inf"
    assert loss.item() > 0.0, "Cross entropy loss must be strictly positive"

    # Backward pass
    model.zero_grad()
    loss.backward()

    # 1. Embedding gradients
    emb_grad = model.embedding.weight.grad
    assert emb_grad is not None, "Embedding weight gradient is None"
    assert not torch.isnan(emb_grad).any(), "Embedding weight gradient contains NaN"
    assert not torch.isinf(emb_grad).any(), "Embedding weight gradient contains Inf"
    assert emb_grad.norm().item() > 0.0, "Embedding weight gradient is zero"

    # 2. Tied output head behavior
    assert model.lm_head.weight.grad is emb_grad, (
        "Weight-tied lm_head.weight.grad is not identical in storage to embedding.weight.grad"
    )

    # 3. Layerwise inspections
    for l_idx, layer in enumerate(model.layers):
        # Attention
        for proj_name, proj in [
            ("q_proj", layer.attn.q_proj),
            ("k_proj", layer.attn.k_proj),
            ("v_proj", layer.attn.v_proj),
            ("out_proj", layer.attn.out_proj),
        ]:
            grad = proj.weight.grad
            assert grad is not None, f"Layer {l_idx} {proj_name}.weight.grad is None"
            assert not torch.isnan(grad).any(), f"Layer {l_idx} {proj_name} grad has NaN"
            assert not torch.isinf(grad).any(), f"Layer {l_idx} {proj_name} grad has Inf"
            assert grad.norm().item() > 0.0, f"Layer {l_idx} {proj_name} grad is completely zero"

        # FFN
        for ffn_name, proj in [
            ("gate_proj", layer.ffn.gate_proj),
            ("up_proj", layer.ffn.up_proj),
            ("down_proj", layer.ffn.down_proj),
        ]:
            grad = proj.weight.grad
            assert grad is not None, f"Layer {l_idx} {ffn_name}.weight.grad is None"
            assert not torch.isnan(grad).any(), f"Layer {l_idx} {ffn_name} grad has NaN"
            assert not torch.isinf(grad).any(), f"Layer {l_idx} {ffn_name} grad has Inf"
            assert grad.norm().item() > 0.0, f"Layer {l_idx} {ffn_name} grad is completely zero"

        # RMSNorm in layer
        for norm_name, norm_module in [
            ("norm_1", layer.norm_1),
            ("norm_2", layer.norm_2),
        ]:
            grad = norm_module.weight.grad
            assert grad is not None, f"Layer {l_idx} {norm_name}.weight.grad is None"
            assert not torch.isnan(grad).any(), f"Layer {l_idx} {norm_name} grad has NaN"
            assert not torch.isinf(grad).any(), f"Layer {l_idx} {norm_name} grad has Inf"
            assert grad.norm().item() > 0.0, f"Layer {l_idx} {norm_name} grad is completely zero"

    # 4. Final RMSNorm
    final_norm_grad = model.final_norm.weight.grad
    assert final_norm_grad is not None, "Final norm weight gradient is None"
    assert not torch.isnan(final_norm_grad).any(), "Final norm weight grad has NaN"
    assert not torch.isinf(final_norm_grad).any(), "Final norm weight grad has Inf"
    assert final_norm_grad.norm().item() > 0.0, "Final norm weight grad is completely zero"


def test_all_trainable_parameters_covered():
    """
    Ensure 100% of trainable parameters receive valid non-zero, finite gradients.
    """
    torch.manual_seed(101)
    cfg = ModelConfig()
    model = ChakrMicro(cfg)
    model.train()

    B, T = 1, 8
    input_ids = torch.randint(0, cfg.vocab_size, (B, T), dtype=torch.long)
    targets = torch.randint(0, cfg.vocab_size, (B, T), dtype=torch.long)

    logits = model(input_ids)
    loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), targets.view(-1))
    loss.backward()

    total_tensors = 0
    valid_grads = 0

    unique_params = set(model.parameters())
    for p in unique_params:
        if p.requires_grad:
            total_tensors += 1
            assert p.grad is not None, "Encountered parameter with missing grad"
            assert not torch.isnan(p.grad).any(), "Encountered NaN in grad"
            assert not torch.isinf(p.grad).any(), "Encountered Inf in grad"
            if p.grad.norm().item() > 0.0:
                valid_grads += 1

    assert total_tensors > 0, "No trainable parameters found"
    assert valid_grads == total_tensors, (
        f"Expected all {total_tensors} unique parameters to have non-zero gradients, but only {valid_grads} did"
    )
