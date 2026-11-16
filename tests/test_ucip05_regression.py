import pytest
from pathlib import Path
from scripts.run_m3_evaluations import DirectNeuralRunner
from chakrview.chat import ConversationalChatCLI

REPO_ROOT = Path(__file__).resolve().parents[1]

def test_ucip05_medium_strict_loading():
    p = REPO_ROOT / "releases/ucip_05_candidate/chakrview_ucip05_medium_model.pt"
    assert p.exists()
    runner = DirectNeuralRunner(p)
    assert runner.arch_name == "ChakrMedium"
    assert sum(p.numel() for p in runner.model.parameters()) == 12932736

def test_ucip05_micro_strict_loading():
    p = REPO_ROOT / "releases/ucip_05_candidate/chakrview_ucip05_micro_model.pt"
    assert p.exists()
    runner = DirectNeuralRunner(p)
    assert runner.arch_name == "ChakrMicro"
    assert sum(p.numel() for p in runner.model.parameters()) == 3443136

def test_ucip05_cli_generation_and_banner():
    p = REPO_ROOT / "releases/ucip_05_candidate/chakrview_ucip05_medium_model.pt"
    cli = ConversationalChatCLI(checkpoint_path=p, memory_enabled=False)
    resp = cli.interact_step("Say exactly: The sky is blue.")
    assert "The sky is blue." in resp
