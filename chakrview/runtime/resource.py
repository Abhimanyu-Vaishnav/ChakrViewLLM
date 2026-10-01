from dataclasses import dataclass, field
from enum import Enum, auto
import os
import platform

class RuntimeStrategy(Enum):
    LOW_RESOURCE = auto()
    STANDARD = auto()
    ACCELERATED = auto()
    DISTRIBUTED_READY = auto()

@dataclass
class HardwareCapability:
    cpu_cores: int
    cpu_architecture: str
    ram_gb: float
    gpu_available: bool
    gpu_vendor: str
    vram_gb: float
    storage_capacity_gb: float
    network_available: bool

@dataclass
class ResourcePolicy:
    strategy: RuntimeStrategy
    max_context_size: int
    batch_size: int
    allow_parallel_execution: bool
    use_accelerator: bool

class ResourceDetector:
    @staticmethod
    def detect() -> HardwareCapability:
        # Graceful hardware detection without hard dependencies on large ML libraries if not needed
        try:
            cpu_cores = os.cpu_count() or 1
        except Exception:
            cpu_cores = 1
            
        cpu_arch = platform.machine()
        
        # Mock RAM and Storage for demonstration, in a real system use psutil
        ram_gb = 8.0 
        storage_gb = 100.0
        
        gpu_available = False
        gpu_vendor = "none"
        vram_gb = 0.0
        
        # Safely try to detect torch
        try:
            import torch
            if torch.cuda.is_available():
                gpu_available = True
                gpu_vendor = "nvidia"
                vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        except ImportError:
            pass

        return HardwareCapability(
            cpu_cores=cpu_cores,
            cpu_architecture=cpu_arch,
            ram_gb=ram_gb,
            gpu_available=gpu_available,
            gpu_vendor=gpu_vendor,
            vram_gb=vram_gb,
            storage_capacity_gb=storage_gb,
            network_available=True
        )
        
    @staticmethod
    def determine_strategy(capability: HardwareCapability) -> ResourcePolicy:
        if capability.gpu_available and capability.vram_gb >= 8.0:
            return ResourcePolicy(
                strategy=RuntimeStrategy.ACCELERATED,
                max_context_size=8192,
                batch_size=16,
                allow_parallel_execution=True,
                use_accelerator=True
            )
        elif capability.cpu_cores >= 4 and capability.ram_gb >= 8.0:
            return ResourcePolicy(
                strategy=RuntimeStrategy.STANDARD,
                max_context_size=2048,
                batch_size=4,
                allow_parallel_execution=True,
                use_accelerator=False
            )
        else:
            return ResourcePolicy(
                strategy=RuntimeStrategy.LOW_RESOURCE,
                max_context_size=512,
                batch_size=1,
                allow_parallel_execution=False,
                use_accelerator=False
            )
