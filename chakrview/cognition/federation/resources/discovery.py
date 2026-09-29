"""
Local Hardware Resource and Capability Discovery Provider (Step 37).

Discovers node hardware characteristics, memory capacity, platform environment,
and registered computational capabilities without proprietary driver dependencies
or sensitive privacy leakage.

Supported Platforms: Linux, Windows, macOS, Raspberry Pi, Android, Generic POSIX.
"""

import ctypes
import hashlib
import multiprocessing
import os
import platform
import shutil
import sys
import time
from typing import Dict, List, Optional, Any, Tuple

from chakrview.capability.registry import CapabilityRegistry
from chakrview.cognition.federation.resources.models import (
    CPUResource,
    MemoryResource,
    AcceleratorResource,
    AcceleratorType,
    StorageResource,
    StorageClass,
    PlatformResource,
    NodeResourceProfile,
    ExecutionType,
    AdvertisedCapability,
)


class LocalResourceDetector:
    """
    Platform-neutral detector for local node computational resources.
    Discovers CPU, memory, accelerators, storage, and platform attributes.
    """

    def __init__(
        self,
        node_epoch: int = 1,
        coarse_privacy_default: bool = True,
        override_accelerator: Optional[AcceleratorResource] = None,
    ) -> None:
        self.node_epoch = node_epoch
        self.coarse_privacy_default = coarse_privacy_default
        self.override_accelerator = override_accelerator

    # ========================================================================
    # 1. CPU Discovery
    # ========================================================================

    def detect_cpu(self) -> CPUResource:
        """Detect CPU architecture, core counts, and basic compute features."""
        try:
            logical_cores = multiprocessing.cpu_count()
        except Exception:
            logical_cores = os.cpu_count() or 1

        arch = platform.machine().lower() or "generic"

        # Safely identify basic instruction set capabilities without C-extensions
        capabilities: List[str] = []
        if "x86_64" in arch or "amd64" in arch:
            capabilities.append("x86_64")
        elif "arm" in arch or "aarch64" in arch:
            capabilities.append("ARM_NEON")
        elif "riscv" in arch:
            capabilities.append("RISCV")

        if sys.maxsize > 2**32:
            capabilities.append("64BIT")
        else:
            capabilities.append("32BIT")

        # In CPU-first baseline, logical cores are available
        available_cores = float(logical_cores)

        return CPUResource(
            architecture=arch,
            logical_cores=logical_cores,
            physical_cores=max(1, logical_cores // 2) if logical_cores > 1 else 1,
            instruction_capabilities=capabilities,
            total_capacity_mhz=None,
            available_cores=available_cores,
            utilization_percent=None,  # Volatile metric omitted or populated dynamically
        )

    # ========================================================================
    # 2. Memory Discovery
    # ========================================================================

    def detect_memory(self) -> MemoryResource:
        """Detect total and available system memory across operating systems."""
        total_mb = 4096
        avail_mb = 2048

        os_name = platform.system()
        if os_name == "Windows":
            try:
                class MEMORYSTATUSEX(ctypes.Structure):
                    _fields_ = [
                        ("dwLength", ctypes.c_ulong),
                        ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                    ]
                stat = MEMORYSTATUSEX()
                stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
                if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                    total_mb = int(stat.ullTotalPhys / (1024 * 1024))
                    avail_mb = int(stat.ullAvailPhys / (1024 * 1024))
            except Exception:
                pass

        elif os_name == "Linux":
            try:
                with open("/proc/meminfo", "r", encoding="utf-8") as f:
                    meminfo = {}
                    for line in f:
                        parts = line.split(":")
                        if len(parts) == 2:
                            meminfo[parts[0].strip()] = parts[1].strip()
                    if "MemTotal" in meminfo:
                        total_kb = int(meminfo["MemTotal"].split()[0])
                        total_mb = total_kb // 1024
                    if "MemAvailable" in meminfo:
                        avail_kb = int(meminfo["MemAvailable"].split()[0])
                        avail_mb = avail_kb // 1024
            except Exception:
                pass

        return MemoryResource(
            total_memory_mb=max(total_mb, 512),
            available_memory_mb=max(avail_mb, 256),
            reservable_memory_mb=int(total_mb * 0.75),
            utilization_percent=round(100.0 * (1.0 - (avail_mb / total_mb)), 1) if total_mb > 0 else None,
        )

    # ========================================================================
    # 3. Accelerator Discovery (Generic, Vendor-Neutral)
    # ========================================================================

    def detect_accelerator(self) -> AcceleratorResource:
        """Detect generic hardware accelerators if present without proprietary drivers."""
        if self.override_accelerator is not None:
            return self.override_accelerator

        # Check PyTorch backend availability if imported
        try:
            import torch
            if torch.cuda.is_available():
                count = torch.cuda.device_count()
                return AcceleratorResource(
                    accelerator_type=AcceleratorType.GPU,
                    device_count=count,
                    model_name="Generic CUDA Accelerator",
                    total_memory_mb=None,
                    available_memory_mb=None,
                    compute_capabilities=["CUDA", "FP32", "FP16"],
                    is_available=True,
                )
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return AcceleratorResource(
                    accelerator_type=AcceleratorType.GPU,
                    device_count=1,
                    model_name="Apple Metal MPS",
                    compute_capabilities=["METAL", "FP32", "FP16"],
                    is_available=True,
                )
        except Exception:
            pass

        return AcceleratorResource(
            accelerator_type=AcceleratorType.NONE,
            device_count=0,
            model_name=None,
            is_available=False,
        )

    # ========================================================================
    # 4. Storage Discovery
    # ========================================================================

    def detect_storage(self) -> StorageResource:
        """Detect available storage capacity without exposing file paths."""
        try:
            usage = shutil.disk_usage(os.getcwd())
            avail_mb = int(usage.free / (1024 * 1024))
            total_mb = int(usage.total / (1024 * 1024))
        except Exception:
            avail_mb = 10240
            total_mb = 51200

        return StorageResource(
            storage_class=StorageClass.FAST_SSD,
            available_storage_mb=avail_mb,
            total_storage_mb=total_mb,
            capability_flags=["READ", "WRITE", "EPHEMERAL"],
        )

    # ========================================================================
    # 5. Platform Discovery
    # ========================================================================

    def detect_platform(self) -> PlatformResource:
        """Detect high-level platform and runtime environment."""
        return PlatformResource(
            os_family=platform.system() or "GenericPOSIX",
            os_release=platform.release() or "unknown",
            python_version=sys.version.split()[0],
            chakrview_version="37.0",
            node_epoch=self.node_epoch,
        )

    # ========================================================================
    # 6. Composite Profile Generation
    # ========================================================================

    def detect_profile(self, coarse: Optional[bool] = None) -> NodeResourceProfile:
        """
        Generate complete composite NodeResourceProfile.
        Applies privacy sanitization if coarse is True.
        """
        cpu = self.detect_cpu()
        mem = self.detect_memory()
        accel = self.detect_accelerator()
        storage = self.detect_storage()
        plat = self.detect_platform()

        hasher = hashlib.sha256()
        hasher.update(f"{cpu.architecture}:{cpu.logical_cores}:{mem.total_memory_mb}:{plat.os_family}".encode("utf-8"))
        profile_id = f"prof_{hasher.hexdigest()[:16]}"

        profile = NodeResourceProfile(
            profile_id=profile_id,
            cpu=cpu,
            memory=mem,
            platform=plat,
            accelerator=accel if accel.is_available else None,
            storage=storage,
            timestamp=time.time(),
            is_coarse=False,
        )

        use_coarse = self.coarse_privacy_default if coarse is None else coarse
        if use_coarse:
            return profile.sanitize_for_privacy()
        return profile

    # ========================================================================
    # 7. Computational Capability Discovery
    # ========================================================================

    def detect_capabilities(
        self,
        registry: Optional[CapabilityRegistry] = None,
    ) -> List[AdvertisedCapability]:
        """
        Detect computational capabilities available on this node.
        If a CapabilityRegistry is passed, maps registered descriptors into AdvertisedCapability manifests.
        Always includes core foundational capabilities (Tokenization, Preprocessing, Inference).
        """
        advertised: List[AdvertisedCapability] = []

        # Standard core capabilities supported by ChakrView runtime
        advertised.append(
            AdvertisedCapability(
                capability_id="core.tokenization",
                name="BytePiece Tokenizer",
                version="1.0.0",
                execution_type=ExecutionType.TOKENIZATION,
                supported_inputs=["text/plain"],
                supported_outputs=["application/x-token-ids"],
                concurrency_limit=4,
                requires_accelerator=False,
                is_exposed=True,
                description="Hermetic BytePiece tokenization and decoding.",
            )
        )
        advertised.append(
            AdvertisedCapability(
                capability_id="core.inference",
                name="ChakrMicro Neural Core",
                version="1.0.0",
                execution_type=ExecutionType.INFERENCE,
                supported_inputs=["application/x-token-ids"],
                supported_outputs=["application/x-logits"],
                concurrency_limit=2,
                requires_accelerator=False,
                is_exposed=True,
                description="ChakrMicro 3.44M parameter autoregressive inference.",
            )
        )
        advertised.append(
            AdvertisedCapability(
                capability_id="core.crypto",
                name="Ed25519 Cryptographic Suite",
                version="1.0.0",
                execution_type=ExecutionType.CRYPTOGRAPHIC_OPS,
                supported_inputs=["application/octet-stream"],
                supported_outputs=["application/x-signature"],
                concurrency_limit=8,
                requires_accelerator=False,
                is_exposed=True,
                description="Hermetic Ed25519 signature generation and verification.",
            )
        )

        # Inspect local CapabilityRegistry if provided
        if registry:
            for desc in registry.list_capabilities():
                exec_type = ExecutionType.GENERIC_COMPUTE
                desc_name_lower = desc.name.lower()
                if "calc" in desc_name_lower:
                    exec_type = ExecutionType.GENERIC_COMPUTE
                elif "text" in desc_name_lower or "transform" in desc_name_lower:
                    exec_type = ExecutionType.PREPROCESSING
                elif "sensor" in desc_name_lower:
                    exec_type = ExecutionType.DOCUMENT_PROCESSING

                advertised.append(
                    AdvertisedCapability(
                        capability_id=desc.capability_id,
                        name=desc.name,
                        version=desc.version,
                        execution_type=exec_type,
                        supported_inputs=["application/json"],
                        supported_outputs=["application/json"],
                        concurrency_limit=1,
                        requires_accelerator=False,
                        is_exposed=True,
                        description=desc.description,
                    )
                )

        return advertised
