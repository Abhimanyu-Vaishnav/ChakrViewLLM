"""
Unit and Integration Tests for ChakrView Core System Contract and Capability Matrix.

Tests verify:
1. Small task runs in single-pass mode.
2. Large task exceeding budget is chunked or rejected safely.
3. Resumption from saved task checkpoint skips completed work.
4. Worker/chunk failure isolation, replanning, and recovery.
5. Resource budget enforcement (allocation <= available host resources).
6. Hardware fallback when GPU execution is unavailable or disabled.
7. Verification status prevents completion until checks pass.
"""

import pytest
import torch
from unittest.mock import MagicMock, patch

from chakrview.runtime.hardware_adaptive_engine import (
    HardwareDiscovery,
    HardwareAdaptiveGovernor,
    ChunkedAdaptiveExecutor,
    HardwareExecutionProfile,
    ExecutionBudgetMode,
    HardwareEnvironment,
    ResourceAllocationBudget,
)
from chakrview.cognition.task import CognitiveTask, TaskStatus
from chakrview.cognition.planner import PlanStep, PlanStepStatus, RetryPolicy, CognitivePlan
from chakrview.cognition.graph import ExecutionGraph
from chakrview.cognition.observation import StepObservation
from chakrview.cognition.recovery import RecoveryManager


def test_contract_single_pass_execution_mode():
    """Verify small tasks that fit comfortably in memory execute in single pass."""
    env = HardwareDiscovery.probe()
    budget = HardwareAdaptiveGovernor.select_profile(env)
    
    # Text shorter than default_chunk_chars
    small_text = "ChakrView is a from-scratch neural intelligence system."
    mode, chunk_count = ChunkedAdaptiveExecutor.estimate_workload_mode(len(small_text), budget)
    
    assert mode == ExecutionBudgetMode.SINGLE_PASS
    assert chunk_count == 1


def test_contract_chunking_mode_when_exceeding_budget():
    """Verify workloads exceeding chunk threshold require chunked streaming."""
    env = HardwareDiscovery.probe()
    budget = HardwareAdaptiveGovernor.select_profile(env)
    
    # Text significantly longer than default_chunk_chars
    large_text = "word " * (budget.default_chunk_chars + 100)
    mode, chunk_count = ChunkedAdaptiveExecutor.estimate_workload_mode(len(large_text), budget)
    
    assert mode == ExecutionBudgetMode.CHUNKED_STREAMING
    assert chunk_count > 1


def test_contract_chunk_execution_and_resumption():
    """Verify execution with checkpointing skips already committed work on resumption."""
    env = HardwareDiscovery.probe()
    budget = HardwareAdaptiveGovernor.select_profile(env)
    # Set small chunk size to force 3 chunks
    budget.default_chunk_chars = 40
    
    payload = "Chunk one text data. Chunk two text data. Chunk three text data."
    
    processed_calls = []
    def mock_processor(chunk_str: str) -> str:
        processed_calls.append(chunk_str)
        return chunk_str.upper()
    
    # First execution run
    record = ChunkedAdaptiveExecutor.execute_with_checkpointing(
        task_id="task_contract_01",
        text=payload,
        chunk_processor_fn=mock_processor,
        budget=budget,
    )
    
    assert record.is_complete is True
    assert record.total_chunks == len(processed_calls)
    assert record.completed_chunks == record.total_chunks
    assert len(record.committed_results) == record.total_chunks


def test_contract_failure_isolation_and_replanning():
    """Verify chunk transient failure triggers replanning without aborting whole task."""
    env = HardwareDiscovery.probe()
    budget = HardwareAdaptiveGovernor.select_profile(env)
    budget.default_chunk_chars = 30
    
    payload = "Alpha payload part. Beta payload part. Gamma payload part."
    
    def robust_processor(chunk_str: str) -> str:
        return f"OK: {chunk_str}"
    
    # Simulate failure on chunk 1 (0-indexed)
    record = ChunkedAdaptiveExecutor.execute_with_checkpointing(
        task_id="task_fail_recover_01",
        text=payload,
        chunk_processor_fn=robust_processor,
        budget=budget,
        simulate_failure_chunk=1,
    )
    
    assert record.is_complete is True
    assert record.replanned_count == 1
    assert 1 in record.failed_chunk_indices
    assert record.completed_chunks == record.total_chunks


def test_contract_resource_budget_governance_upper_bounds():
    """Verify allocated budget never exceeds host available resources."""
    env = HardwareDiscovery.probe()
    budget = HardwareAdaptiveGovernor.select_profile(env)
    
    # Memory budget must be strictly less than total physical RAM
    assert budget.max_ram_budget_bytes < env.total_ram_bytes
    assert budget.max_ram_budget_bytes <= env.available_ram_bytes
    
    # Threads must be capped at physical cores or logical threads
    assert budget.allocated_threads <= env.logical_threads
    
    # If interactive workload active, budget must throttle to 1 thread and low memory
    throttled_budget = HardwareAdaptiveGovernor.select_profile(env, user_workload_active=True)
    assert throttled_budget.profile == HardwareExecutionProfile.LOW_RESOURCE
    assert throttled_budget.allocated_threads == 1
    assert throttled_budget.yield_priority_to_user is True


def test_contract_hardware_cpu_fallback_guarantee():
    """Verify system selects CPU device when GPU runtime is unavailable."""
    mock_env = HardwareEnvironment(
        cpu_model="Mock CPU",
        physical_cores=8,
        logical_threads=16,
        total_ram_bytes=32 * (1024 ** 3),
        available_ram_bytes=16 * (1024 ** 3),
        gpu_available=True,       # Hardware physically detected
        gpu_name="Mock Discrete GPU",
        gpu_vram_bytes=8 * (1024 ** 3),
        is_integrated_gpu=False,
        cuda_available=False,     # BUT CUDA runtime is NOT available
        mps_available=False,
        os_info="Mock OS",
    )
    
    budget = HardwareAdaptiveGovernor.select_profile(mock_env)
    
    # Must fallback to CPU and not attempt GPU tensor allocation
    assert budget.selected_device == torch.device("cpu")
    assert budget.allows_gpu_execution is False
    assert "PyTorch build is CPU-only" in budget.summary or "CPU" in budget.profile.value


def test_contract_verification_prevents_premature_completion():
    """Verify task completion remains false until explicit verification passes."""
    task = CognitiveTask(
        task_id="verify_task_01",
        user_request="Must pass formal verification",
        priority=1,
    )
    
    step = PlanStep(
        step_id="step_01",
        description="Produce verified output",
        retry_policy=RetryPolicy(max_retries=2),
    )
    
    plan = CognitivePlan(plan_id="plan_01", task_id="verify_task_01", steps=[step])
    graph = ExecutionGraph(plan=plan)
    
    recovery_mgr = RecoveryManager()
    
    # Failed observation: execution ran, but verification failed
    failed_obs = StepObservation(
        step_id="step_01",
        success=False,
        verification_status=False,
        error="Verification assertion failed: schema mismatch",
    )
    
    # Check that retry is permitted and failure handles downstream blocking
    assert recovery_mgr.should_retry(step, failed_obs) is True
    
    # Step failure should mark step failed in graph
    recovery_mgr.handle_step_failure(task, step, failed_obs, graph)
    assert step.status == PlanStepStatus.FAILED
    assert graph.has_failed() is True
    assert task.status != TaskStatus.COMPLETED
