"""
Hardware Adaptation Framework for ChakrView (Step 9).

Detects system compute and memory capabilities, and creates deterministic execution plans:
- Low-end and older CPU machines (constrained context, small batch, single-thread)
- Modern multi-core workstations (full context, multi-thread, FP32/BF16)
- Accelerated hardware (CUDA/MPS/ROCm)
"""

from dataclasses import dataclass, field, asdict
from enum import Enum
import os
import platform
from typing import Dict, List, Optional, Any
import torch

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False


class ComputeDevice(str, Enum):
    """Available compute device types."""
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"
    ROCM = "rocm"


class PrecisionType(str, Enum):
    """Supported floating-point and integer quantization precisions."""
    FP32 = "fp32"
    BF16 = "bf16"
    FP16 = "fp16"
    INT8 = "int8"
    INT4 = "int4"


@dataclass
class HardwareProfile:
    """
    Empirical snapshot of available hardware resources.
    
    Attributes:
        device: Primary compute device.
        cpu_model: String description of host processor.
        physical_cores: Count of physical execution cores.
        logical_threads: Total logical execution threads.
        total_ram_bytes: Physical system RAM in bytes.
        available_ram_bytes: Free or available RAM in bytes.
        gpu_name: Optional GPU device name.
        gpu_vram_bytes: GPU memory in bytes (0 if CPU).
        supported_precisions: Precisions physically supported and safe on this host.
        os_platform: Operating system name and release.
    """
    device: ComputeDevice
    cpu_model: str
    physical_cores: int
    logical_threads: int
    total_ram_bytes: int
    available_ram_bytes: int
    gpu_name: Optional[str] = None
    gpu_vram_bytes: int = 0
    supported_precisions: List[PrecisionType] = field(
        default_factory=lambda: [PrecisionType.FP32]
    )
    os_platform: str = field(default_factory=lambda: f"{platform.system()} {platform.release()}")

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["device"] = self.device.value
        data["supported_precisions"] = [p.value for p in self.supported_precisions]
        return data

    @property
    def total_ram_gb(self) -> float:
        return self.total_ram_bytes / (1024 ** 3)

    @property
    def available_ram_gb(self) -> float:
        return self.available_ram_bytes / (1024 ** 3)


class HardwareCapabilityDetector:
    """
    Deterministic system capability probe.
    """

    @staticmethod
    def detect() -> HardwareProfile:
        """Probe local hardware and construct a HardwareProfile."""
        # 1. Device detection
        device = ComputeDevice.CPU
        gpu_name = None
        gpu_vram = 0

        if torch.cuda.is_available():
            device = ComputeDevice.CUDA
            gpu_name = torch.cuda.get_device_name(0)
            gpu_vram = torch.cuda.get_device_properties(0).total_memory
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = ComputeDevice.MPS
            gpu_name = "Apple Silicon Neural/MPS"

        # 2. CPU and Core counts
        logical_threads = os.cpu_count() or 1
        physical_cores = logical_threads
        cpu_model = platform.processor() or "Unknown CPU"

        # 3. RAM detection
        total_ram = 8 * (1024 ** 3)  # Safe default 8 GB
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

        # 4. Precision support
        supported = [PrecisionType.FP32]
        if device == ComputeDevice.CUDA:
            supported.extend([PrecisionType.FP16, PrecisionType.BF16, PrecisionType.INT8])
        elif hasattr(torch, "bfloat16"):
            # PyTorch on CPU supports BF16 emulation/acceleration on modern CPUs
            supported.append(PrecisionType.BF16)

        return HardwareProfile(
            device=device,
            cpu_model=cpu_model,
            physical_cores=physical_cores,
            logical_threads=logical_threads,
            total_ram_bytes=total_ram,
            available_ram_bytes=available_ram,
            gpu_name=gpu_name,
            gpu_vram_bytes=gpu_vram,
            supported_precisions=supported,
        )


@dataclass
class ModelExecutionPlan:
    """
    Concrete runtime parameters tailored to hardware capacity.
    
    Attributes:
        device: Target execution device ("cpu", "cuda", etc.).
        precision: Arithmetic precision type.
        max_context_len: Bounded context length for inference (T <= 512).
        batch_size: Batch size for execution.
        thread_count: Number of threads allocated.
        kv_cache_strategy: KV caching mode ("full", "sliding_window", "disabled").
        memory_limit_mb: Upper RSS memory budget in megabytes.
    """
    device: str
    precision: PrecisionType
    max_context_len: int
    batch_size: int
    thread_count: int
    kv_cache_strategy: str = "full"
    memory_limit_mb: int = 1024

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["precision"] = self.precision.value
        return data


class RuntimePlanner:
    """
    Maps a HardwareProfile and application constraints to a safe ModelExecutionPlan.
    """

    @staticmethod
    def plan(
        profile: HardwareProfile,
        requested_context_len: int = 512,
        batch_size: int = 1,
        conservative_memory: bool = False,
    ) -> ModelExecutionPlan:
        """
        Produce a deterministic execution plan.
        
        Ensures low-RAM environments gracefully scale down context length and thread count.
        """
        # Hard cap context to ChakrMicro v0.1 architectural maximum (512)
        target_context = min(requested_context_len, 512)

        # 1. Thread allocation
        if profile.device == ComputeDevice.CPU:
            # Allocate min(physical_cores, 8) to prevent thrashing
            thread_count = max(1, min(profile.physical_cores, 8))
        else:
            thread_count = 2

        # 2. Memory-based constraint adaptation
        avail_gb = profile.available_ram_gb

        if avail_gb < 1.0 or conservative_memory:
            # Severely constrained system: restrict context to 128
            target_context = min(target_context, 128)
            memory_limit_mb = 384
            kv_cache_strategy = "sliding_window"
            thread_count = max(1, min(thread_count, 2))
        elif avail_gb < 2.0:
            # Modest RAM (older PC): restrict context to 256
            target_context = min(target_context, 256)
            memory_limit_mb = 512
            kv_cache_strategy = "full"
        else:
            # Healthy RAM: full 512 context permitted
            memory_limit_mb = 1024
            kv_cache_strategy = "full"

        # 3. Precision selection
        precision = PrecisionType.FP32
        if PrecisionType.BF16 in profile.supported_precisions and profile.device == ComputeDevice.CUDA:
            precision = PrecisionType.BF16

        return ModelExecutionPlan(
            device=profile.device.value,
            precision=precision,
            max_context_len=target_context,
            batch_size=batch_size,
            thread_count=thread_count,
            kv_cache_strategy=kv_cache_strategy,
            memory_limit_mb=memory_limit_mb,
        )
