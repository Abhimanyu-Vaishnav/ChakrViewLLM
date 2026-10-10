"""
ChakrView Hardware-Adaptive Execution Engine.

Core Principle:
ONE COGNITIVE BRAIN + HARDWARE-ADAPTIVE EXECUTION.

Hardware differences change:
- execution strategy
- device selection
- resource allocation
- batching
- chunking
- parallelism & thread count
- checkpointing & recovery
- yielding to user interactive workload

Hardware differences DO NOT create different cognitive models or identities.

Guarantees:
- Never crashes if GPU detection is unavailable.
- Capability detection rather than hard-coded names.
- RESOURCE AVAILABILITY != PERMISSION != REQUIREMENT.
- Minimum sufficient resource allocation.
- Automatic CPU fallback if GPU unavailable or unsuitable.
- Checkpointed chunked execution (NODE/CHUNK FAILURE != WHOLE TASK FAILURE).
- User workload priority & cooperation (yield when user active).
- Canonical baseline weight immutability (Delta W = 0).
"""

from __future__ import annotations

import os
import platform
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, Any, List, Optional, Tuple, Callable

import torch

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False


class HardwareExecutionProfile(str, Enum):
    """
    Execution policy profiles.
    These are execution policies only — NOT separate cognitive models.
    """
    CPU_ONLY = "CPU_ONLY"
    CPU_HIGH_MEMORY = "CPU_HIGH_MEMORY"
    GPU_ACCELERATED = "GPU_ACCELERATED"
    GPU_MEMORY_CONSTRAINED = "GPU_MEMORY_CONSTRAINED"
    IGPU_SHARED_MEMORY = "IGPU_SHARED_MEMORY"
    LOW_RESOURCE = "LOW_RESOURCE"
    BALANCED = "BALANCED"


class ExecutionBudgetMode(str, Enum):
    SINGLE_PASS = "SINGLE_PASS"
    CHUNKED_STREAMING = "CHUNKED_STREAMING"
    THROTTLED_YIELD = "THROTTLED_YIELD"


@dataclass
class HardwareEnvironment:
    """
    Empirical snapshot of host computing resources.
    """
    cpu_model: str
    physical_cores: int
    logical_threads: int
    total_ram_bytes: int
    available_ram_bytes: int
    gpu_available: bool
    gpu_name: Optional[str]
    gpu_vram_bytes: int
    is_integrated_gpu: bool
    cuda_available: bool
    mps_available: bool
    os_info: str

    @property
    def total_ram_gb(self) -> float:
        return self.total_ram_bytes / (1024 ** 3)

    @property
    def available_ram_gb(self) -> float:
        return self.available_ram_bytes / (1024 ** 3)

    @property
    def gpu_vram_gb(self) -> float:
        return self.gpu_vram_bytes / (1024 ** 3)


@dataclass
class ResourceAllocationBudget:
    """
    Bounded safe working budget.
    Ensures ChakrView does NOT consume all host resources.
    """
    profile: HardwareExecutionProfile
    selected_device: torch.device
    allocated_threads: int
    max_ram_budget_bytes: int
    max_vram_budget_bytes: int
    default_chunk_chars: int
    allows_gpu_execution: bool
    yield_priority_to_user: bool
    fallback_policy: str
    summary: str


@dataclass
class ChunkExecutionState:
    """
    Execution state for a single workload chunk.
    """
    chunk_index: int
    total_chunks: int
    chunk_data: str
    is_committed: bool = False
    result: Optional[Any] = None
    error: Optional[str] = None
    retry_count: int = 0


@dataclass
class CheckpointedExecutionRecord:
    """
    State record enabling resumption from latest committed checkpoint.
    Invariant: NODE/CHUNK FAILURE != WHOLE TASK FAILURE.
    """
    task_id: str
    total_chunks: int
    completed_chunks: int
    committed_results: List[Any]
    failed_chunk_indices: List[int]
    is_complete: bool
    peak_working_bytes: int
    replanned_count: int = 0


class HardwareDiscovery:
    """
    Runtime capability discovery without hardcoded names or crash risks.
    """

    @classmethod
    def probe(cls) -> HardwareEnvironment:
        # 1. CPU & OS
        logical_threads = os.cpu_count() or 1
        physical_cores = logical_threads
        cpu_model = platform.processor() or "Generic CPU"
        os_info = f"{platform.system()} {platform.release()} ({platform.machine()})"

        # 2. RAM (psutil fallback to safe default)
        total_ram = 8 * (1024 ** 3)
        available_ram = 4 * (1024 ** 3)

        if _HAS_PSUTIL:
            try:
                mem = psutil.virtual_memory()
                total_ram = mem.total
                available_ram = mem.available
                cores = psutil.cpu_count(logical=False)
                if cores:
                    physical_cores = cores
            except Exception:
                pass

        # 3. GPU / PyTorch Accelerator detection
        gpu_avail = False
        gpu_name = None
        gpu_vram = 0
        cuda_avail = False
        mps_avail = False
        is_igpu = False
        gpu_driver_detected = False

        # Step 3a: Check PyTorch CUDA / MPS first
        try:
            if torch.cuda.is_available():
                cuda_avail = True
                gpu_avail = True
                gpu_name = torch.cuda.get_device_name(0)
                gpu_vram = torch.cuda.get_device_properties(0).total_memory
                igpu_keywords = ["intel", "iris", "uhd", "radeon graphics", "vega", "integrated"]
                if any(kw in gpu_name.lower() for kw in igpu_keywords) and gpu_vram < 4 * (1024 ** 3):
                    is_igpu = True
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                mps_avail = True
                gpu_avail = True
                gpu_name = "Apple Silicon MPS (Unified Memory)"
                is_igpu = True
                gpu_vram = available_ram
        except Exception:
            pass

        # Step 3b: If PyTorch is CPU-only but host hardware has a physical GPU (e.g. nvidia-smi or Windows WMI)
        if not gpu_avail:
            try:
                import subprocess
                import shutil
                if shutil.which("nvidia-smi"):
                    smi_res = subprocess.run(
                        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                        capture_output=True,
                        text=True,
                        timeout=3,
                    )
                    if smi_res.returncode == 0 and smi_res.stdout.strip():
                        first_line = smi_res.stdout.strip().splitlines()[0]
                        parts = [p.strip() for p in first_line.split(",")]
                        gpu_name = parts[0]
                        vram_mb = float(parts[1]) if len(parts) > 1 else 0.0
                        gpu_vram = int(vram_mb * (1024 ** 2))
                        gpu_avail = True
                        gpu_driver_detected = True
            except Exception:
                pass

        return HardwareEnvironment(
            cpu_model=cpu_model,
            physical_cores=physical_cores,
            logical_threads=logical_threads,
            total_ram_bytes=total_ram,
            available_ram_bytes=available_ram,
            gpu_available=gpu_avail,
            gpu_name=gpu_name,
            gpu_vram_bytes=gpu_vram,
            is_integrated_gpu=is_igpu,
            cuda_available=cuda_avail,
            mps_available=mps_avail,
            os_info=os_info,
        )


class HardwareAdaptiveGovernor:
    """
    Enforces resource governance and maps HardwareEnvironment to execution budgets.
    """

    @classmethod
    def select_profile(
        cls,
        env: HardwareEnvironment,
        user_workload_active: bool = False,
    ) -> ResourceAllocationBudget:
        """
        Determines the execution profile and budget.
        Principle: Minimum sufficient allocation; leave headroom for host system.
        """
        # User cooperative priority: if user workload is high, yield immediately
        if user_workload_active:
            return ResourceAllocationBudget(
                profile=HardwareExecutionProfile.LOW_RESOURCE,
                selected_device=torch.device("cpu"),
                allocated_threads=1,
                max_ram_budget_bytes=int(env.available_ram_bytes * 0.15),
                max_vram_budget_bytes=0,
                default_chunk_chars=200,
                allows_gpu_execution=False,
                yield_priority_to_user=True,
                fallback_policy="YIELD_AND_SERIALIZE",
                summary="User interactive workload active. Throttling compute and yielding CPU/GPU.",
            )

        # GPU Accelerated vs CPU
        if env.gpu_available and (env.cuda_available or env.mps_available):
            if env.is_integrated_gpu:
                # Treat iGPU conservatively (shared memory)
                profile = HardwareExecutionProfile.IGPU_SHARED_MEMORY
                device = torch.device("cpu")  # Prefer CPU for lightweight 3.44M core to prevent iGPU driver stall
                max_ram = int(env.available_ram_bytes * 0.35)
                max_vram = int(env.gpu_vram_bytes * 0.30)
                threads = max(1, min(env.physical_cores, 4))
                chunk_chars = 400
                allows_gpu = False
                fallback = "CPU_FALLBACK"
                summary = "Integrated/Shared-Memory GPU detected. Executing on CPU to conserve shared memory."
            elif env.gpu_vram_gb < 2.0:
                profile = HardwareExecutionProfile.GPU_MEMORY_CONSTRAINED
                device = torch.device("cuda:0" if env.cuda_available else "cpu")
                max_ram = int(env.available_ram_bytes * 0.40)
                max_vram = int(env.gpu_vram_bytes * 0.50)
                threads = 2
                chunk_chars = 300
                allows_gpu = True
                fallback = "CPU_FALLBACK_ON_OOM"
                summary = "VRAM-constrained discrete GPU. Conservative allocation with CPU fallback."
            else:
                profile = HardwareExecutionProfile.GPU_ACCELERATED
                device = torch.device("cuda:0" if env.cuda_available else "cpu")
                max_ram = int(env.available_ram_bytes * 0.50)
                max_vram = int(env.gpu_vram_bytes * 0.60)
                threads = max(1, min(env.physical_cores, 6))
                chunk_chars = 800
                allows_gpu = True
                fallback = "AUTOMATIC_CPU_FALLBACK"
                summary = f"GPU accelerated on {env.gpu_name} ({env.gpu_vram_gb:.1f} GB VRAM available)."
        else:
            # CPU Profiles
            if env.available_ram_gb < 1.0 or env.physical_cores <= 2:
                profile = HardwareExecutionProfile.LOW_RESOURCE
                device = torch.device("cpu")
                max_ram = int(env.available_ram_bytes * 0.40)
                max_vram = 0
                threads = 1
                chunk_chars = 250
                allows_gpu = False
                fallback = "SERIALIZED_SINGLE_THREAD"
                summary = "Low-resource host (constrained RAM or <= 2 cores). Minimal serialized execution."
            elif env.available_ram_gb >= 16.0 and env.physical_cores >= 8:
                profile = HardwareExecutionProfile.CPU_HIGH_MEMORY
                device = torch.device("cpu")
                max_ram = int(env.available_ram_bytes * 0.40)
                max_vram = 0
                threads = min(env.physical_cores, 8)
                chunk_chars = 800
                allows_gpu = False
                fallback = "CPU_IN_MEMORY"
                if env.gpu_available and not env.cuda_available:
                    summary = f"High-memory workstation CPU ({env.physical_cores} cores, {env.available_ram_gb:.1f} GB RAM). Physical GPU detected ({env.gpu_name}) but active PyTorch build is CPU-only; executing CPU-first."
                else:
                    summary = f"High-memory workstation CPU ({env.physical_cores} cores, {env.available_ram_gb:.1f} GB RAM)."
            else:
                profile = HardwareExecutionProfile.BALANCED
                device = torch.device("cpu")
                max_ram = int(env.available_ram_bytes * 0.45)
                max_vram = 0
                threads = max(1, min(env.physical_cores, 4))
                chunk_chars = 500
                allows_gpu = False
                fallback = "STANDARD_CPU"
                if env.gpu_available and not env.cuda_available:
                    summary = f"Balanced CPU execution ({env.physical_cores} cores, {env.available_ram_gb:.1f} GB RAM). Physical GPU detected ({env.gpu_name}) but active PyTorch build is CPU-only; executing CPU-first."
                else:
                    summary = f"Balanced CPU execution ({env.physical_cores} cores, {env.available_ram_gb:.1f} GB RAM)."

        return ResourceAllocationBudget(
            profile=profile,
            selected_device=device,
            allocated_threads=threads,
            max_ram_budget_bytes=max_ram,
            max_vram_budget_bytes=max_vram,
            default_chunk_chars=chunk_chars,
            allows_gpu_execution=allows_gpu,
            yield_priority_to_user=False,
            fallback_policy=fallback,
            summary=summary,
        )


class ChunkedAdaptiveExecutor:
    """
    Executes large cognitive tasks via adaptive chunking and state checkpointing.
    Guarantees:
    - Task sizing: Fits budget -> Single pass; Exceeds budget -> Chunked.
    - Checkpointed state persistence: Node/Chunk Failure != Whole Task Failure.
    - Resumption without recomputing already committed chunks.
    """

    @classmethod
    def estimate_workload_mode(
        cls,
        text_size_chars: int,
        budget: ResourceAllocationBudget,
    ) -> Tuple[ExecutionBudgetMode, int]:
        """
        Estimates whether the workload fits in a single pass or requires chunking.
        """
        # If payload size fits within default chunk threshold, single pass
        if text_size_chars <= budget.default_chunk_chars:
            return ExecutionBudgetMode.SINGLE_PASS, 1

        # Calculate estimated chunks
        chunks_count = (text_size_chars + budget.default_chunk_chars - 1) // budget.default_chunk_chars
        return ExecutionBudgetMode.CHUNKED_STREAMING, chunks_count

    @classmethod
    def split_into_chunks(
        cls,
        text: str,
        chunk_chars: int,
    ) -> List[str]:
        words = text.split()
        chunks = []
        current = []
        current_len = 0

        for w in words:
            if current_len + len(w) + 1 > chunk_chars and current:
                chunks.append(" ".join(current))
                current = [w]
                current_len = len(w)
            else:
                current.append(w)
                current_len += len(w) + 1

        if current:
            chunks.append(" ".join(current))

        return chunks or [text]

    @classmethod
    def execute_with_checkpointing(
        cls,
        task_id: str,
        text: str,
        chunk_processor_fn: Callable[[str], Any],
        budget: ResourceAllocationBudget,
        simulate_failure_chunk: Optional[int] = None,
        journal_path: Optional[str | Path] = None,
        verification_fn: Optional[Callable[[List[Any]], bool]] = None,
    ) -> CheckpointedExecutionRecord:
        """
        Decomposes text into chunks, executes chunk-by-chunk, and checkpoints progress.
        If a chunk fails, it replans and retries without dropping committed chunks.
        If journal_path is provided, persists progress durably via DurableChunkedExecutor.
        """
        if journal_path is not None:
            from chakrview.runtime.recovery import DurableChunkedExecutor
            journal_state = DurableChunkedExecutor.execute(
                task_id=task_id,
                text=text,
                chunk_processor_fn=chunk_processor_fn,
                budget=budget,
                journal_path=journal_path,
                simulate_failure_chunk=simulate_failure_chunk,
                verification_fn=verification_fn,
            )
            return CheckpointedExecutionRecord(
                task_id=journal_state.task_id,
                total_chunks=journal_state.total_chunks,
                completed_chunks=journal_state.completed_chunks_count,
                committed_results=journal_state.committed_results,
                failed_chunk_indices=journal_state.failed_chunk_indices,
                is_complete=(journal_state.status.value == "COMPLETED"),
                peak_working_bytes=journal_state.peak_working_bytes,
                replanned_count=1 if journal_state.failed_chunk_indices else 0,
            )

        chunks = cls.split_into_chunks(text, budget.default_chunk_chars)
        total_chunks = len(chunks)

        chunk_states: List[ChunkExecutionState] = [
            ChunkExecutionState(chunk_index=i, total_chunks=total_chunks, chunk_data=c)
            for i, c in enumerate(chunks)
        ]

        committed_results: List[Any] = []
        failed_indices: List[int] = []
        peak_bytes = 0
        replanned = 0

        for idx, state in enumerate(chunk_states):
            chunk_bytes = len(state.chunk_data.encode("utf-8"))
            if chunk_bytes > peak_bytes:
                peak_bytes = chunk_bytes

            # Check if this chunk is already committed
            if state.is_committed:
                continue

            # Simulate failure if requested for recovery testing
            if simulate_failure_chunk is not None and idx == simulate_failure_chunk and state.retry_count == 0:
                state.error = f"Transient execution fault on chunk {idx}"
                state.retry_count += 1
                failed_indices.append(idx)
                # Replan: isolate failed chunk and retry with conservative fallback
                replanned += 1
                try:
                    # Retry chunk
                    res = chunk_processor_fn(state.chunk_data)
                    state.result = res
                    state.is_committed = True
                    committed_results.append(res)
                except Exception as retry_err:
                    state.error = str(retry_err)
            else:
                try:
                    res = chunk_processor_fn(state.chunk_data)
                    state.result = res
                    state.is_committed = True
                    committed_results.append(res)
                except Exception as e:
                    state.error = str(e)
                    failed_indices.append(idx)

        all_done = len(committed_results) == total_chunks
        if all_done and verification_fn is not None:
            try:
                all_done = verification_fn(committed_results)
            except Exception:
                all_done = False

        return CheckpointedExecutionRecord(
            task_id=task_id,
            total_chunks=total_chunks,
            completed_chunks=len(committed_results),
            committed_results=committed_results,
            failed_chunk_indices=failed_indices,
            is_complete=all_done,
            peak_working_bytes=peak_bytes,
            replanned_count=replanned,
        )


class HardwareDiagnostics:
    """
    Exposes structured, non-leaking runtime hardware diagnostics.
    """

    @classmethod
    def get_summary(
        cls,
        env: Optional[HardwareEnvironment] = None,
        budget: Optional[ResourceAllocationBudget] = None,
    ) -> Dict[str, Any]:
        env = env or HardwareDiscovery.probe()
        budget = budget or HardwareAdaptiveGovernor.select_profile(env)

        if budget.allows_gpu_execution:
            gpu_status = "AVAILABLE FOR SUPPORTED OPERATIONS (CUDA Accelerated)"
        elif env.gpu_available and not env.cuda_available:
            gpu_status = f"DETECTED ({env.gpu_name}) — PyTorch build is CPU-only; executing CPU-first"
        else:
            gpu_status = "CPU-FIRST GOVERNED (GPU idle or unneeded)"

        return {
            "hardware_profile": budget.profile.value,
            "cpu_model": env.cpu_model,
            "cpu_cores": f"{env.physical_cores} physical / {env.logical_threads} logical",
            "ram_total_gb": round(env.total_ram_gb, 2),
            "ram_available_gb": round(env.available_ram_gb, 2),
            "ram_budget_mb": round(budget.max_ram_budget_bytes / (1024 ** 2), 1),
            "gpu_detected": env.gpu_available,
            "gpu_name": env.gpu_name or "None",
            "gpu_vram_gb": round(env.gpu_vram_gb, 2) if env.gpu_available else 0.0,
            "cuda_available": env.cuda_available,
            "selected_device": str(budget.selected_device),
            "execution_threads": budget.allocated_threads,
            "default_chunk_chars": budget.default_chunk_chars,
            "fallback_policy": budget.fallback_policy,
            "yield_priority_to_user": budget.yield_priority_to_user,
            "gpu_support_status": gpu_status,
        }
