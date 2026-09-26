"""
Root runner for ChakrMicro CPU Benchmark.
Delegates to scripts.benchmark_brain_cpu.run_benchmark().
"""

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from scripts.benchmark_brain_cpu import run_benchmark

if __name__ == "__main__":
    run_benchmark()
