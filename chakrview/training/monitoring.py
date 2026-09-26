"""
Lightweight CPU and Memory Resource Monitoring for ChakrView (Phase 16).

Monitors:
- Process Resident Set Size (RSS RAM) in MB
- System RAM Utilization
- CPU utilization percentage
"""

import os
from typing import Dict, Any, Optional

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False


class ResourceMonitor:
    """
    Lightweight resource monitor with zero-overhead fallback.
    """
    def __init__(self) -> None:
        self.process = psutil.Process(os.getpid()) if _HAS_PSUTIL else None

    def get_snapshot(self) -> Dict[str, Any]:
        """
        Capture current CPU and RAM metrics without blocking.
        """
        if not _HAS_PSUTIL or self.process is None:
            return {
                "process_ram_mb": 0.0,
                "system_ram_percent": 0.0,
                "cpu_percent": 0.0,
            }

        try:
            mem_info = self.process.memory_info()
            process_ram_mb = mem_info.rss / (1024 * 1024)
            system_ram = psutil.virtual_memory()
            cpu_pct = self.process.cpu_percent(interval=None)

            return {
                "process_ram_mb": round(process_ram_mb, 2),
                "system_ram_percent": round(system_ram.percent, 1),
                "cpu_percent": round(cpu_pct, 1),
            }
        except Exception:
            return {
                "process_ram_mb": 0.0,
                "system_ram_percent": 0.0,
                "cpu_percent": 0.0,
            }
