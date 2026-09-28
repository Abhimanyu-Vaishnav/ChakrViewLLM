"""
Hardware & Resource Probing for ChakrView Adaptation (Step 23).

Inspects runtime resources:
- CPU architecture and core counts
- Total and available RAM
- PyTorch thread configuration
- Device type and GPU presence
- Basic memory pressure

Safe Fallback Principle:
If any metric cannot be safely detected, mark it as 'UNKNOWN' rather than inventing a value.
"""

from dataclasses import dataclass, field, asdict
import os
import platform
import time
from typing import Dict, Optional, Any, Union
import torch

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False


UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class CPUInfo:
    """CPU hardware attributes."""
    architecture: str
    processor: str
    logical_cores: int
    physical_cores: Union[int, str]
    torch_threads: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "architecture": self.architecture,
            "processor": self.processor,
            "logical_cores": self.logical_cores,
            "physical_cores": self.physical_cores,
            "torch_threads": self.torch_threads,
        }


@dataclass(frozen=True)
class MemoryInfo:
    """System memory attributes."""
    total_bytes: Union[int, str]
    available_bytes: Union[int, str]
    memory_pressure: str  # LOW, NORMAL, HIGH, CRITICAL, or UNKNOWN

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_bytes": self.total_bytes,
            "available_bytes": self.available_bytes,
            "memory_pressure": self.memory_pressure,
        }


@dataclass(frozen=True)
class DeviceInfo:
    """Compute device attributes."""
    device_type: str
    gpu_available: bool
    gpu_name: Optional[str] = None
    gpu_memory_bytes: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "device_type": self.device_type,
            "gpu_available": self.gpu_available,
            "gpu_name": self.gpu_name,
            "gpu_memory_bytes": self.gpu_memory_bytes,
        }


@dataclass(frozen=True)
class HardwareProfileSnapshot:
    """Empirical snapshot of host environment resources."""
    cpu: CPUInfo
    memory: MemoryInfo
    device: DeviceInfo
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cpu": self.cpu.to_dict(),
            "memory": self.memory.to_dict(),
            "device": self.device.to_dict(),
            "timestamp": self.timestamp,
        }


class HardwareProfiler:
    """
    Deterministic system capability probe with fail-safe fallback to UNKNOWN.
    """

    @staticmethod
    def profile() -> HardwareProfileSnapshot:
        """Probe local hardware and construct a HardwareProfileSnapshot."""
        # 1. CPU Detection
        arch = platform.machine() or UNKNOWN
        processor = platform.processor() or UNKNOWN
        logical_cores = os.cpu_count() or 1
        torch_threads = torch.get_num_threads()

        physical_cores: Union[int, str] = UNKNOWN
        if _HAS_PSUTIL:
            try:
                cores = psutil.cpu_count(logical=False)
                if cores is not None and cores > 0:
                    physical_cores = cores
            except Exception:
                physical_cores = UNKNOWN

        cpu_info = CPUInfo(
            architecture=arch,
            processor=processor,
            logical_cores=logical_cores,
            physical_cores=physical_cores,
            torch_threads=torch_threads,
        )

        # 2. Memory Detection
        total_ram: Union[int, str] = UNKNOWN
        avail_ram: Union[int, str] = UNKNOWN
        mem_pressure = UNKNOWN

        if _HAS_PSUTIL:
            try:
                mem = psutil.virtual_memory()
                total_ram = int(mem.total)
                avail_ram = int(mem.available)
                usage_pct = mem.percent
                if usage_pct < 60:
                    mem_pressure = "LOW"
                elif usage_pct < 80:
                    mem_pressure = "NORMAL"
                elif usage_pct < 95:
                    mem_pressure = "HIGH"
                else:
                    mem_pressure = "CRITICAL"
            except Exception:
                total_ram = UNKNOWN
                avail_ram = UNKNOWN
                mem_pressure = UNKNOWN

        mem_info = MemoryInfo(
            total_bytes=total_ram,
            available_bytes=avail_ram,
            memory_pressure=mem_pressure,
        )

        # 3. Device Detection
        gpu_available = torch.cuda.is_available()
        gpu_name = None
        gpu_mem = None
        device_type = "cpu"

        if gpu_available:
            try:
                device_type = "cuda"
                gpu_name = torch.cuda.get_device_name(0)
                gpu_mem = torch.cuda.get_device_properties(0).total_memory
            except Exception:
                device_type = "cpu"
                gpu_available = False

        dev_info = DeviceInfo(
            device_type=device_type,
            gpu_available=gpu_available,
            gpu_name=gpu_name,
            gpu_memory_bytes=gpu_mem,
        )

        return HardwareProfileSnapshot(
            cpu=cpu_info,
            memory=mem_info,
            device=dev_info,
            timestamp=time.time(),
        )
