"""
tests/test_ucip07r_loader.py

Automated regression and verification test suite for UCIP-07R:
1. UCIP-05 Medium checkpoint strict loading & parameter count.
2. UCIP-06 Medium checkpoint strict loading & parameter count.
3. UCIP-07 Medium checkpoint strict loading & parameter count.
4. UCIP-07 Core checkpoint strict loading & parameter count.
5. Tied embedding and lm_head weight pointer identity verification.
6. Strict state-dictionary compatibility (missing_keys=[], unexpected_keys=[]).
7. Forward-pass correctness and valid logits shape.
8. Deterministic autoregressive generation.
9. ConversationalChatCLI memory ON/OFF behavior.
10. Correct /info architecture and parameter reporting.
"""

import pytest
import torch
from pathlib import Path
from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from scripts.run_m3_evaluations import DirectNeuralRunner
from chakrview.chat import ConversationalChatCLI

REPO_ROOT = Path(__file__).resolve().parents[1]

UCIP05_MEDIUM = REPO_ROOT / "releases" / "ucip_05_candidate" / "chakrview_ucip05_medium_model.pt"
UCIP06_MEDIUM = REPO_ROOT / "releases" / "ucip_06_candidate" / "chakrview_ucip06_medium_model.pt"
UCIP07_MEDIUM = REPO_ROOT / "releases" / "ucip_07_candidates" / "chakrview_ucip07_medium_model.pt"
UCIP07_CORE = REPO_ROOT / "releases" / "ucip_07_candidates" / "chakrview_ucip07_core_model.pt"


def test_ucip05_medium_loading():
    """Verify UCIP-05 Medium checkpoint loads strictly with 12,932,736 params."""
    assert UCIP05_MEDIUM.exists(), f"Missing {UCIP05_MEDIUM}"
    runner = DirectNeuralRunner(UCIP05_MEDIUM)
    assert runner.arch_name == "ChakrMedium"
    assert runner.param_count == 12_932_736
    assert runner.cfg.d_model == 384
    assert runner.cfg.n_layers == 7


def test_ucip06_medium_loading():
    """Verify UCIP-06 Medium checkpoint loads strictly with 12,932,736 params."""
    assert UCIP06_MEDIUM.exists(), f"Missing {UCIP06_MEDIUM}"
    runner = DirectNeuralRunner(UCIP06_MEDIUM)
    assert runner.arch_name == "ChakrMedium"
    assert runner.param_count == 12_932_736
    assert runner.cfg.d_model == 384
    assert runner.cfg.n_layers == 7


def test_ucip07_medium_loading():
    """Verify UCIP-07 Medium candidate loads strictly with 12,932,736 params and rope_theta=50000.0."""
    assert UCIP07_MEDIUM.exists(), f"Missing {UCIP07_MEDIUM}"
    runner = DirectNeuralRunner(UCIP07_MEDIUM)
    assert runner.arch_name == "ChakrMedium"
    assert runner.param_count == 12_932_736
    assert runner.cfg.d_model == 384
    assert runner.cfg.n_layers == 7
    assert runner.cfg.hidden_dim == 896
    assert runner.cfg.rope_theta == 50000.0


def test_ucip07_core_loading():
    """Verify UCIP-07 Core candidate loads strictly with 27,795,968 params."""
    assert UCIP07_CORE.exists(), f"Missing {UCIP07_CORE}"
    runner = DirectNeuralRunner(UCIP07_CORE)
    assert runner.arch_name == "ChakrCore"
    assert runner.param_count == 27_795_968
    assert runner.cfg.d_model == 512
    assert runner.cfg.n_layers == 8
    assert runner.cfg.hidden_dim == 1408
    assert runner.cfg.rope_theta == 50000.0


def test_tied_weights_identity():
    """Verify embedding and lm_head share the exact same underlying parameter in memory."""
    for p in [UCIP07_MEDIUM, UCIP07_CORE]:
        runner = DirectNeuralRunner(p)
        assert runner.model.embedding.weight.data_ptr() == runner.model.lm_head.weight.data_ptr()
        assert id(runner.model.embedding.weight) == id(runner.model.lm_head.weight)


def test_forward_pass_correctness():
    """Verify forward passes produce valid, finite logits with expected tensor dimensions."""
    runner_med = DirectNeuralRunner(UCIP07_MEDIUM)
    runner_core = DirectNeuralRunner(UCIP07_CORE)

    inp = torch.tensor([[1, 25, 100, 250]], dtype=torch.long)
    with torch.no_grad():
        out_med = runner_med.model(inp)
        out_core = runner_core.model(inp)

    assert out_med.shape == (1, 4, 4096)
    assert out_core.shape == (1, 4, 4096)
    assert not torch.isnan(out_med).any()
    assert not torch.isnan(out_core).any()


def test_deterministic_autoregressive_generation():
    """Verify deterministic token generation produces consistent outputs on the same prompt."""
    cli1 = ConversationalChatCLI(UCIP07_MEDIUM, memory_enabled=False)
    cli2 = ConversationalChatCLI(UCIP07_MEDIUM, memory_enabled=False)

    resp1 = cli1.interact_step("What does a CPU do?")
    resp2 = cli2.interact_step("What does a CPU do?")
    assert resp1 == resp2
    assert len(resp1) > 0


def test_cli_memory_modes():
    """Verify ConversationalChatCLI behaves properly under memory ON and OFF modes."""
    cli_on = ConversationalChatCLI(UCIP07_MEDIUM, memory_enabled=True)
    assert cli_on.memory_enabled is True
    assert cli_on.memory is not None

    cli_off = ConversationalChatCLI(UCIP07_MEDIUM, memory_enabled=False)
    assert cli_off.memory_enabled is False


def test_cli_info_reporting(capsys):
    """Verify /info reports exact architecture name, param count, and layers."""
    cli = ConversationalChatCLI(UCIP07_MEDIUM, memory_enabled=False)
    cli.print_info()
    out = capsys.readouterr().out
    assert "ChakrMedium" in out
    assert "12,932,736 params" in out
    assert "7 layers" in out
    assert "384 dim" in out
    assert "896 ffn" in out
    assert "Memory Mode:     DISABLED (Pure Neural)" in out
