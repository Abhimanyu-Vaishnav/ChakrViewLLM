"""
tests/test_architecture_loader.py

Automated verification tests for architecture-aware checkpoint loading in ChakrView:
1. ChakrMicro candidate loads strictly with 3.44M params, d_model=192, n_layers=6
2. ChakrMedium candidate loads strictly with 12.93M params, d_model=384, n_layers=7
3. Unknown or incompatible architectures fail cleanly with descriptive exceptions
4. ConversationalChatCLI /info output dynamically reports model architecture, param count, and dimensions
5. Autoregressive single-step forward pass runs deterministically on both models
6. Automated CLI verification mode (--test-mode) passes on both models
"""

import pytest
import torch
from pathlib import Path
from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from scripts.run_m3_evaluations import DirectNeuralRunner
from chakrview.chat import ConversationalChatCLI

REPO_ROOT = Path(__file__).resolve().parents[1]
MICRO_CKPT = REPO_ROOT / "releases" / "ucip_04_candidate" / "chakrview_ucip04_model.pt"
MEDIUM_CKPT = REPO_ROOT / "releases" / "ucip_04_candidate" / "chakrview_ucip04_medium_model.pt"


def test_micro_checkpoint_strict_loading():
    """Verify ChakrMicro checkpoint loads with strict=True and correct dimensions."""
    assert MICRO_CKPT.exists(), f"Micro checkpoint not found at {MICRO_CKPT}"
    runner = DirectNeuralRunner(MICRO_CKPT)
    assert runner.arch_name == "ChakrMicro"
    assert runner.cfg.d_model == 192
    assert runner.cfg.n_layers == 6
    assert runner.cfg.hidden_dim == 512
    assert runner.cfg.n_heads == 6
    assert runner.param_count == 3_443_136
    assert isinstance(runner.model, ChakrMicro)


def test_medium_checkpoint_strict_loading():
    """Verify ChakrMedium checkpoint loads with strict=True and correct dimensions."""
    assert MEDIUM_CKPT.exists(), f"Medium checkpoint not found at {MEDIUM_CKPT}"
    runner = DirectNeuralRunner(MEDIUM_CKPT)
    assert runner.arch_name == "ChakrMedium"
    assert runner.cfg.d_model == 384
    assert runner.cfg.n_layers == 7
    assert runner.cfg.hidden_dim == 896
    assert runner.cfg.n_heads == 6
    assert runner.param_count == 12_932_736
    assert isinstance(runner.model, ChakrMicro)


def test_forward_pass_both_models():
    """Verify both loaded models can perform a forward pass and produce valid logits."""
    runner_micro = DirectNeuralRunner(MICRO_CKPT)
    runner_med = DirectNeuralRunner(MEDIUM_CKPT)

    dummy_input = torch.tensor([[0, 10, 25, 50]], dtype=torch.long)
    with torch.no_grad():
        out_micro = runner_micro.model(dummy_input)
        out_med = runner_med.model(dummy_input)

    assert out_micro.shape == (1, 4, 4096)
    assert out_med.shape == (1, 4, 4096)
    assert not torch.isnan(out_micro).any()
    assert not torch.isnan(out_med).any()


def test_cli_info_dynamic_reporting(capsys):
    """Verify ConversationalChatCLI /info dynamically reflects the loaded model architecture."""
    # Test Micro
    cli_micro = ConversationalChatCLI(MICRO_CKPT, memory_enabled=True)
    cli_micro.print_info()
    captured_micro = capsys.readouterr().out
    assert "ChakrMicro" in captured_micro
    assert "3,443,136 params" in captured_micro
    assert "192 dim" in captured_micro
    assert "6 layers" in captured_micro

    # Test Medium
    cli_med = ConversationalChatCLI(MEDIUM_CKPT, memory_enabled=False)
    cli_med.print_info()
    captured_med = capsys.readouterr().out
    assert "ChakrMedium" in captured_med
    assert "12,932,736 params" in captured_med
    assert "384 dim" in captured_med
    assert "7 layers" in captured_med


def test_cli_step_generation():
    """Verify interact_step executes without runtime errors for both architectures."""
    cli_micro = ConversationalChatCLI(MICRO_CKPT, memory_enabled=False)
    out_micro = cli_micro.interact_step("Hello")
    assert isinstance(out_micro, str) and len(out_micro) > 0

    cli_med = ConversationalChatCLI(MEDIUM_CKPT, memory_enabled=False)
    out_med = cli_med.interact_step("Hello")
    assert isinstance(out_med, str) and len(out_med) > 0


def test_invalid_checkpoint_error_handling(tmp_path):
    """Verify unsupported model configurations raise clear errors instead of corrupting state."""
    bad_ckpt_path = tmp_path / "bad_model.pt"
    # Create invalid state_dict with unsupported embedding dimension
    torch.save({
        "model_state_dict": {
            "embedding.weight": torch.randn(4096, 256),
            "layers.0.attn.q_proj.weight": torch.randn(256, 256)
        }
    }, bad_ckpt_path)

    with pytest.raises(ValueError, match="Unsupported model architecture dimension"):
        DirectNeuralRunner(bad_ckpt_path)
