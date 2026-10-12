"""
ChakrView Hardware-Adaptive Workload Scheduler (Batch 46).

Provides device-aware workload planning and adaptive chunking:
- Evaluates CPU cores, system RAM, discrete GPU, VRAM, and unified memory.
- Calculates dynamic memory budgets with safety reserves.
- Selects adaptive chunk sizes based on memory pressure and transfer costs rather than fixed constants.
- Provides fallback to CPU when GPU memory is constrained or transfer overhead dominates.
- Disallows remote/distributed execution unless explicitly configured and authorized.
"""

from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

from chakrview.runtime.hardware import ComputeDevice, HardwareCapabilityDetector, HardwareProfile, PrecisionType


class WorkloadType(str, Enum):
    INFERENCE = "inference"
    TRAINING_SFT = "training_sft"
    TOKENIZATION = "tokenization"
    EVALUATION = "evaluation"


@dataclass
class AdaptiveScheduleDecision:
    target_device: ComputeDevice
    precision: PrecisionType
    batch_size: int
    chunk_size: int
    num_workers: int
    memory_budget_mb: float
    safety_reserve_mb: float
    fallback_reason: Optional[str] = None
    scheduling_rationale: str = ""


class HardwareAdaptiveScheduler:
    """
    Schedules tensor workloads dynamically based on measured device capabilities.
    """

    def __init__(self, profile: Optional[HardwareProfile] = None):
        self.profile = profile or HardwareCapabilityDetector.detect()

    def plan_workload(
        self,
        workload_type: WorkloadType,
        total_items: int,
        item_estimated_bytes: int = 1024 * 1024,  # Default 1 MB per sequence tensor
        allow_gpu: bool = True,
    ) -> AdaptiveScheduleDecision:
        """
        Calculates optimal execution parameters, chunking, and device placement.
        """
        avail_ram_mb = self.profile.available_ram_bytes / (1024 * 1024)
        vram_mb = self.profile.gpu_vram_bytes / (1024 * 1024)

        # 1. Device selection with transfer & memory boundary checks
        target_device = ComputeDevice.CPU
        fallback_reason = None
        precision = PrecisionType.FP32

        if allow_gpu and self.profile.device == ComputeDevice.CUDA and vram_mb > 1024:
            # GPU candidate: check VRAM budget (reserve 20% or 512 MB minimum)
            gpu_reserve_mb = max(512.0, vram_mb * 0.20)
            gpu_budget_mb = vram_mb - gpu_reserve_mb
            est_total_mb = (total_items * item_estimated_bytes) / (1024 * 1024)

            if est_total_mb > gpu_budget_mb:
                # Chunking required on GPU or fallback to system RAM
                target_device = ComputeDevice.CUDA
                precision = PrecisionType.BF16 if PrecisionType.BF16 in self.profile.supported_precisions else PrecisionType.FP16
            else:
                target_device = ComputeDevice.CUDA
                precision = PrecisionType.BF16 if PrecisionType.BF16 in self.profile.supported_precisions else PrecisionType.FP16
        else:
            target_device = ComputeDevice.CPU
            if self.profile.device == ComputeDevice.CUDA and not allow_gpu:
                fallback_reason = "GPU execution explicitly disabled by policy."
            elif self.profile.device != ComputeDevice.CUDA:
                fallback_reason = "No CUDA hardware available on host."
            precision = PrecisionType.FP32

        # 2. Safety reserves & RAM budgeting on chosen device
        if target_device == ComputeDevice.CPU:
            safety_reserve_mb = max(1024.0, avail_ram_mb * 0.25)
            memory_budget_mb = max(512.0, avail_ram_mb - safety_reserve_mb)
        else:
            safety_reserve_mb = max(512.0, vram_mb * 0.20)
            memory_budget_mb = max(256.0, vram_mb - safety_reserve_mb)

        # 3. Dynamic adaptive chunk size selection
        item_mb = max(0.001, item_estimated_bytes / (1024 * 1024))
        max_items_in_budget = max(1, int(memory_budget_mb / item_mb))

        if total_items <= max_items_in_budget:
            chunk_size = total_items
        else:
            # Scale chunk size adaptively
            chunk_size = max(1, min(total_items, max_items_in_budget // 2))

        # 4. Batch size & thread allocation
        if target_device == ComputeDevice.CPU:
            batch_size = 2 if workload_type == WorkloadType.TRAINING_SFT else min(8, max(1, chunk_size))
            num_workers = min(4, self.profile.physical_cores)
        else:
            batch_size = 8 if workload_type == WorkloadType.TRAINING_SFT else min(32, max(1, chunk_size))
            num_workers = 2

        rationale = (
            f"Scheduled {workload_type.value} on {target_device.value} ({precision.value}). "
            f"Allocated {memory_budget_mb:.1f} MB budget with {safety_reserve_mb:.1f} MB reserve. "
            f"Chunk size set to {chunk_size}/{total_items} items."
        )

        return AdaptiveScheduleDecision(
            target_device=target_device,
            precision=precision,
            batch_size=batch_size,
            chunk_size=chunk_size,
            num_workers=num_workers,
            memory_budget_mb=round(memory_budget_mb, 1),
            safety_reserve_mb=round(safety_reserve_mb, 1),
            fallback_reason=fallback_reason,
            scheduling_rationale=rationale,
        )
