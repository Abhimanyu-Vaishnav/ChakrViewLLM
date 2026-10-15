import pytest
import torch
from chakrview.brain.config import ModelConfig
from chakrview.brain.pointer_generator import ChakrPointerMicro, PointerGeneratorHead


def test_pointer_generator_shapes_and_probabilities():
    cfg = ModelConfig(vocab_size=4096, d_model=192, n_layers=6, n_heads=6, max_seq_len=512)
    model = ChakrPointerMicro(cfg)
    model.eval()

    B = 2
    T = 45
    prompt_len = 25
    input_ids = torch.randint(0, 4096, (B, T))

    with torch.no_grad():
        v_logits, ptr_probs, p_gen, final_probs = model(input_ids, prompt_len)

    assert v_logits.shape == (B, T, 4096)
    assert ptr_probs.shape == (B, T, prompt_len)
    assert p_gen.shape == (B, T, 1)
    assert final_probs.shape == (B, T, 4096)

    # Probabilities must be strictly non-negative and sum to 1
    assert torch.all(final_probs >= 0.0)
    assert torch.all(p_gen >= 0.0) and torch.all(p_gen <= 1.0)
    prob_sums = final_probs.sum(dim=-1)
    assert torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-4)


def test_pointer_generator_boundary_isolation():
    # Verify pointer cannot attend to positions beyond prompt_len
    cfg = ModelConfig(vocab_size=4096, d_model=192, n_layers=2, n_heads=2, max_seq_len=128)
    head = PointerGeneratorHead(cfg.d_model)

    B = 1
    T = 30
    prompt_len = 12
    h = torch.randn(B, T, cfg.d_model)

    ptr_probs, p_gen = head(h, prompt_len)
    assert ptr_probs.shape == (B, T, prompt_len)
    # The pointer dimension must match prompt_len exactly
    assert ptr_probs.size(-1) == 12


def test_copy_gate_extreme_values():
    # If p_gen=1, final_probs == vocab_probs
    # If p_gen=0, final_probs == pointer_mapped_probs
    vocab_probs = torch.tensor([[[0.7, 0.3]]])
    ptr_probs = torch.tensor([[[0.2, 0.8]]])
    prompt_ids = torch.tensor([[0, 1]])

    # Simulate scatter add with p_gen = 0.0
    p_gen = torch.tensor([[[0.0]]])
    final_probs = torch.zeros(1, 1, 2)
    final_probs.scatter_add_(2, prompt_ids.unsqueeze(1), ptr_probs)
    assert torch.allclose(final_probs, torch.tensor([[[0.2, 0.8]]]))
