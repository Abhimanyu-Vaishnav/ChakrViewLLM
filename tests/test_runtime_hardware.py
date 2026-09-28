"""
Unit tests for Step 9 Hardware Adaptation Framework (chakrview.runtime.hardware).
"""

import pytest
from chakrview.runtime.hardware import (
    ComputeDevice,
    PrecisionType,
    HardwareProfile,
    HardwareCapabilityDetector,
    ModelExecutionPlan,
    RuntimePlanner,
)


def test_hardware_detector():
    profile = HardwareCapabilityDetector.detect()
    assert profile.device in (ComputeDevice.CPU, ComputeDevice.CUDA, ComputeDevice.MPS)
    assert profile.physical_cores >= 1
    assert profile.logical_threads >= 1
    assert profile.total_ram_bytes > 0
    assert profile.available_ram_bytes > 0
    assert PrecisionType.FP32 in profile.supported_precisions
    
    d = profile.to_dict()
    assert "cpu_model" in d
    assert "supported_precisions" in d


def test_runtime_planner_standard_system():
    # Simulate a modern multi-core workstation with 16 GB available RAM
    profile = HardwareProfile(
        device=ComputeDevice.CPU,
        cpu_model="Simulated 16-Core CPU",
        physical_cores=16,
        logical_threads=32,
        total_ram_bytes=32 * (1024 ** 3),
        available_ram_bytes=16 * (1024 ** 3),
        supported_precisions=[PrecisionType.FP32, PrecisionType.BF16],
    )
    plan = RuntimePlanner.plan(profile, requested_context_len=512)
    assert plan.device == "cpu"
    assert plan.max_context_len == 512
    assert plan.precision == PrecisionType.FP32
    assert plan.thread_count <= 8  # capped to avoid thrashing
    assert plan.kv_cache_strategy == "full"
    assert plan.memory_limit_mb == 1024


def test_runtime_planner_constrained_low_ram():
    # Simulate an older/low-end PC with only 512 MB available RAM
    profile = HardwareProfile(
        device=ComputeDevice.CPU,
        cpu_model="Simulated Dual-Core Legacy PC",
        physical_cores=2,
        logical_threads=2,
        total_ram_bytes=2 * (1024 ** 3),
        available_ram_bytes=512 * (1024 ** 2),  # 0.5 GB available
        supported_precisions=[PrecisionType.FP32],
    )
    plan = RuntimePlanner.plan(profile, requested_context_len=512)
    # Under 1.0 GB RAM: context capped to 128, memory limited to 384 MB
    assert plan.max_context_len == 128
    assert plan.memory_limit_mb == 384
    assert plan.kv_cache_strategy == "sliding_window"
    assert plan.thread_count <= 2
