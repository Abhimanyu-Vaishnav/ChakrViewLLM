"""
Tests for ChakrView Hardware-Adaptive Execution Engine.

Verifies:
1. Hardware discovery (safe probing, no crash if GPU absent)
2. CPU-only fallback
3. GPU unavailable handling
4. Low-memory profile allocation
5. Sufficient-memory single-pass vs oversized workload chunking
6. Checkpointed multi-chunk execution & recovery on chunk failure
7. Resource budget enforcement (RESOURCE AVAILABILITY != PERMISSION != REQUIREMENT)
8. User workload cooperation & throttle
9. Canonical baseline parameter count (3,443,136) and weight hash bit-exactness (Delta W = 0)
"""

from __future__ import annotations

import pytest
import torch

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.brain.canonical_hash import (
    verify_canonical_baseline,
    CANONICAL_BASELINE_SHA256,
    CANONICAL_PARAM_COUNT,
)
from chakrview.runtime.hardware_adaptive_engine import (
    HardwareDiscovery,
    HardwareEnvironment,
    HardwareAdaptiveGovernor,
    HardwareExecutionProfile,
    ExecutionBudgetMode,
    ChunkedAdaptiveExecutor,
    HardwareDiagnostics,
    ResourceAllocationBudget,
)


def test_hardware_discovery_safe_probe():
    """Verify hardware discovery executes safely without crashing."""
    env = HardwareDiscovery.probe()
    assert env.physical_cores >= 1
    assert env.logical_threads >= 1
    assert env.total_ram_bytes > 0
    assert env.available_ram_bytes > 0
    assert isinstance(env.os_info, str)
    assert isinstance(env.gpu_available, bool)


def test_cpu_only_fallback_when_gpu_unavailable():
    """Verify that when GPU is unavailable, system falls back to CPU profile."""
    # Synthetic environment with no GPU
    env = HardwareEnvironment(
        cpu_model="Test CPU",
        physical_cores=4,
        logical_threads=8,
        total_ram_bytes=16 * (1024 ** 3),
        available_ram_bytes=8 * (1024 ** 3),
        gpu_available=False,
        gpu_name=None,
        gpu_vram_bytes=0,
        is_integrated_gpu=False,
        cuda_available=False,
        mps_available=False,
        os_info="TestOS",
    )
    budget = HardwareAdaptiveGovernor.select_profile(env)
    assert budget.profile in (HardwareExecutionProfile.BALANCED, HardwareExecutionProfile.CPU_HIGH_MEMORY)
    assert budget.selected_device == torch.device("cpu")
    assert budget.allows_gpu_execution is False
    assert budget.fallback_policy in ("STANDARD_CPU", "CPU_IN_MEMORY")


def test_gpu_available_but_unusable_or_igpu_conservative():
    """Verify that integrated GPUs or shared-memory devices are handled conservatively."""
    env_igpu = HardwareEnvironment(
        cpu_model="Test APU",
        physical_cores=8,
        logical_threads=16,
        total_ram_bytes=16 * (1024 ** 3),
        available_ram_bytes=6 * (1024 ** 3),
        gpu_available=True,
        gpu_name="Intel Iris Xe Graphics",
        gpu_vram_bytes=2 * (1024 ** 3),
        is_integrated_gpu=True,
        cuda_available=True,
        mps_available=False,
        os_info="TestOS",
    )
    budget = HardwareAdaptiveGovernor.select_profile(env_igpu)
    assert budget.profile == HardwareExecutionProfile.IGPU_SHARED_MEMORY
    assert budget.selected_device == torch.device("cpu")
    assert budget.allows_gpu_execution is False


def test_low_memory_profile_allocation():
    """Verify low-resource profile when RAM is severely constrained."""
    env_low = HardwareEnvironment(
        cpu_model="Low-End CPU",
        physical_cores=2,
        logical_threads=2,
        total_ram_bytes=2 * (1024 ** 3),
        available_ram_bytes=512 * (1024 ** 2),  # 512 MB
        gpu_available=False,
        gpu_name=None,
        gpu_vram_bytes=0,
        is_integrated_gpu=False,
        cuda_available=False,
        mps_available=False,
        os_info="TestOS",
    )
    budget = HardwareAdaptiveGovernor.select_profile(env_low)
    assert budget.profile == HardwareExecutionProfile.LOW_RESOURCE
    assert budget.allocated_threads == 1
    assert budget.default_chunk_chars <= 250


def test_resource_budget_does_not_reserve_all_host_memory():
    """Verify RESOURCE AVAILABILITY != PERMISSION: ChakrView leaves safety margins."""
    env = HardwareDiscovery.probe()
    budget = HardwareAdaptiveGovernor.select_profile(env)
    # Budget must not claim all available RAM (at most 50%)
    assert budget.max_ram_budget_bytes < env.available_ram_bytes
    assert budget.max_ram_budget_bytes <= int(env.available_ram_bytes * 0.55)


def test_user_workload_cooperation_yield():
    """Verify that when human user interactive workload is active, ChakrView yields."""
    env = HardwareDiscovery.probe()
    budget_user_active = HardwareAdaptiveGovernor.select_profile(env, user_workload_active=True)
    assert budget_user_active.yield_priority_to_user is True
    assert budget_user_active.allocated_threads == 1
    assert budget_user_active.allows_gpu_execution is False
    assert "yielding" in budget_user_active.summary.lower()


def test_task_sizing_single_pass_vs_chunking():
    """Verify small workloads execute in single pass while large ones are chunked."""
    env = HardwareDiscovery.probe()
    budget = HardwareAdaptiveGovernor.select_profile(env)

    # 1. Short text -> Single pass
    short_text = "Ram lives in Delhi."
    mode_short, chunks_short = ChunkedAdaptiveExecutor.estimate_workload_mode(len(short_text), budget)
    assert mode_short == ExecutionBudgetMode.SINGLE_PASS
    assert chunks_short == 1

    # 2. Oversized text -> Chunked streaming
    long_text = ("This is an extended relational knowledge claim about diverse entities. " * 30)
    mode_long, chunks_long = ChunkedAdaptiveExecutor.estimate_workload_mode(len(long_text), budget)
    assert mode_long == ExecutionBudgetMode.CHUNKED_STREAMING
    assert chunks_long > 1


def test_chunked_execution_with_checkpointing_and_recovery():
    """Verify multi-chunk execution with failure recovery (CHUNK FAILURE != TASK FAILURE)."""
    env = HardwareDiscovery.probe()
    budget = HardwareAdaptiveGovernor.select_profile(env)

    test_document = (
        "Alpha connects to Beta. "
        "Beta connects to Gamma. "
        "Gamma connects to Delta. "
        "Delta connects to Epsilon. "
        "Epsilon connects to Zeta."
    )

    processed_chunks = []
    def mock_processor(chunk_text: str):
        processed_chunks.append(chunk_text)
        return f"PROCESSED({len(chunk_text)} chars)"

    # Simulate fault on chunk 1
    record = ChunkedAdaptiveExecutor.execute_with_checkpointing(
        task_id="test_task_recover",
        text=test_document,
        chunk_processor_fn=mock_processor,
        budget=budget,
        simulate_failure_chunk=0,
    )

    assert record.is_complete is True
    assert record.completed_chunks == record.total_chunks
    assert record.replanned_count == 1
    assert len(record.committed_results) == record.total_chunks


def test_canonical_baseline_immutability():
    """Verify that hardware adaptation leaves the canonical neural baseline bit-exact."""
    torch.manual_seed(42)
    model = ChakrMicro(ModelConfig())
    verif = verify_canonical_baseline(model)
    assert verif["total_params"] == CANONICAL_PARAM_COUNT
    assert verif["digest"] == CANONICAL_BASELINE_SHA256
    assert verif["delta_w"] == 0
    assert verif["is_bit_exact"] is True


def test_hardware_diagnostics_formatting():
    """Verify hardware diagnostics formatting does not expose private reasoning traces."""
    diag = HardwareDiagnostics.get_summary()
    assert "hardware_profile" in diag
    assert "cpu_model" in diag
    assert "selected_device" in diag
    assert "ram_budget_mb" in diag
    assert "gpu_support_status" in diag
    # Ensure no internal secret traces
    for k, v in diag.items():
        assert "chain_of_thought" not in str(v).lower()
        assert "hidden_state" not in str(v).lower()


def test_hardware_execution_cuda_or_cpu_fallback():
    """Verify tensor execution on selected hardware device or graceful fallback."""
    env = HardwareDiscovery.probe()
    budget = HardwareAdaptiveGovernor.select_profile(env)

    device = budget.selected_device
    if budget.allows_gpu_execution and torch.cuda.is_available():
        # Execute tensor operation on CUDA
        t1 = torch.tensor([1.0, 2.0], device=device)
        t2 = torch.tensor([3.0, 4.0], device=device)
        t3 = t1 + t2
        torch.cuda.synchronize()
        res = t3.cpu().tolist()
        assert res == [4.0, 6.0]
        # Verify bounded VRAM budget (not reserving all VRAM)
        assert torch.cuda.memory_allocated() < budget.max_vram_budget_bytes
    else:
        # CPU execution
        t1 = torch.tensor([1.0, 2.0], device=torch.device("cpu"))
        t2 = torch.tensor([3.0, 4.0], device=torch.device("cpu"))
        t3 = t1 + t2
        assert t3.tolist() == [4.0, 6.0]
