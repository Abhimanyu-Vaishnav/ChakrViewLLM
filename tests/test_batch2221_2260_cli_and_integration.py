"""
Comprehensive test suite for Batch 2221-2260:
1. End-to-end assistant CLI integration (status, ask, arithmetic tool routing, device selection, consent).
2. Fresh reproduction evaluation artifact validation.
3. Hardware benchmark thresholds.
"""

import json
import os
import subprocess
import sys
from pathlib import Path
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.batch2221_2260.integrated_assistant_adapter import IntegratedAssistantAdapter


def test_independent_reproduction_results():
    """Verify fresh reproduction evaluation artifact matches documented metrics."""
    eval_path = REPO_ROOT / "experiments" / "batch2221_2260" / "REPRODUCED_EVALUATION.json"
    assert eval_path.exists(), "Missing REPRODUCED_EVALUATION.json"
    
    with open(eval_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    assert "models" in data
    assert "canonical_baseline" in data["models"]
    assert "cpt_seed_42_best" in data["models"]
    
    base_tw = data["models"]["canonical_baseline"]["token_weighted_ce"]
    s42_tw = data["models"]["cpt_seed_42_best"]["token_weighted_ce"]
    
    assert base_tw == 4.4789
    assert s42_tw == 4.2861
    
    paired = data["paired_comparisons_vs_baseline"]["cpt_seed_42_best"]
    assert paired["token_weighted_ce_delta"] == -0.1928
    assert paired["document_win_rate_pct"] == 76.55


def test_hardware_benchmark_artifact():
    """Verify inference hardware benchmarks contain both CPU and CUDA measurements."""
    hw_path = REPO_ROOT / "experiments" / "batch2221_2260" / "HARDWARE_BENCHMARKS.json"
    assert hw_path.exists()
    
    with open(hw_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    assert "devices" in data
    assert "cpu" in data["devices"]
    assert data["devices"]["cpu"]["prefill_throughput_tok_per_sec"]["median"] > 1000.0
    
    if torch.cuda.is_available():
        assert "cuda" in data["devices"]
        assert data["devices"]["cuda"]["batched_forward_pass_ctx256"]["batch_32"]["throughput_tok_per_sec"] > 50000.0


def test_integrated_assistant_adapter_arithmetic_routing():
    """Verify IntegratedAssistantAdapter intercepts arithmetic and calculates accurately."""
    adapter = IntegratedAssistantAdapter()
    
    # Direct arithmetic
    res = adapter.ask("What is 500 - 234?")
    assert res.decision == "CALCULATED"
    assert res.confidence == 1.0
    assert "266" in res.answer
    assert res.tool_trace is not None
    assert res.tool_trace["tool_name"] == "safe_calculator"


def test_integrated_assistant_adapter_consent_gate(tmp_path):
    """Verify that interaction is not logged to feedback store without consent."""
    fb_file = tmp_path / "test_feedback.json"
    mem_file = tmp_path / "test_memory.json"
    
    # 1. No consent
    adapter_no_consent = IntegratedAssistantAdapter(
        memory_path=mem_file,
        feedback_path=fb_file,
        enable_consent_default=False
    )
    res1 = adapter_no_consent.ask("Calculate 12 * 12")
    assert res1.feedback_id is None
    
    # 2. Consent granted
    adapter_with_consent = IntegratedAssistantAdapter(
        memory_path=mem_file,
        feedback_path=fb_file,
        enable_consent_default=True
    )
    res2 = adapter_with_consent.ask("Calculate 12 * 12")
    assert res2.feedback_id is not None
    assert fb_file.exists()
    
    with open(fb_file, "r", encoding="utf-8") as f:
        records = json.load(f)
    assert res2.feedback_id in records
    assert records[res2.feedback_id]["user_consent"] is True


def test_cli_subcommands_execution():
    """Test the assistant CLI directly via python -m subprocess."""
    # 1. status command
    res_status = subprocess.run(
        [sys.executable, "-m", "chakrview.runtime.assistant_cli", "status"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True
    )
    assert res_status.returncode == 0
    assert "CHAKRVIEW HARDWARE RUNTIME & SYSTEM STATUS" in res_status.stdout
    assert "CPU Logical Cores:" in res_status.stdout
    
    # 2. ask command with arithmetic
    res_ask = subprocess.run(
        [sys.executable, "-m", "chakrview.runtime.assistant_cli", "ask", "What is 25 * 4?"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True
    )
    assert res_ask.returncode == 0
    assert "Decision: CALCULATED" in res_ask.stdout
    assert "100" in res_ask.stdout
    
    # 3. ask command with explicit device
    res_dev = subprocess.run(
        [sys.executable, "-m", "chakrview.runtime.assistant_cli", "--device", "cpu", "ask", "What is 100 - 37?"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True
    )
    assert res_dev.returncode == 0
    assert "Device: CPU" in res_dev.stdout
    assert "63" in res_dev.stdout
