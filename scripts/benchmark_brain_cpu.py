"""
CPU-First Neural Core Forward Pass Benchmark for ChakrMicro v0.1.

Measures:
- Parameter count
- Model initialization time
- Forward pass latency (median, p95, min, max) across T in {1, 16, 64, 128, 256, 512}
- Throughput (tokens/second)
- Peak process memory
"""

import time
import json
import statistics
import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


def get_ram_mb() -> float:
    if HAS_PSUTIL:
        return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)
    return 0.0


def benchmark_cpu(warmup_runs: int = 5, measured_runs: int = 20) -> dict:
    base_ram_mb = get_ram_mb()
    
    # 1. Initialization Benchmark
    t0 = time.perf_counter()
    cfg = ModelConfig()
    model = ChakrMicro(cfg)
    model.eval()
    init_time_ms = (time.perf_counter() - t0) * 1000.0
    
    param_counts = model.count_parameters()
    post_init_ram_mb = get_ram_mb()
    
    results = {
        "device": "cpu",
        "cpu_threads": torch.get_num_threads(),
        "initialization_time_ms": round(init_time_ms, 2),
        "base_ram_mb": round(base_ram_mb, 2),
        "post_init_ram_mb": round(post_init_ram_mb, 2),
        "parameters": param_counts,
        "sequence_benchmarks": {}
    }
    
    # 2. Sequence Lengths: 1, 16, 64, 128, 256, 512
    test_lengths = [1, 16, 64, 128, 256, 512]
    
    for seq_len in test_lengths:
        input_ids = torch.randint(0, cfg.vocab_size, (1, seq_len), dtype=torch.long)
        
        # Warmup
        with torch.no_grad():
            for _ in range(warmup_runs):
                _ = model(input_ids)
                
        # Measured runs
        latencies = []
        with torch.no_grad():
            for _ in range(measured_runs):
                t_start = time.perf_counter()
                _ = model(input_ids)
                latencies.append((time.perf_counter() - t_start) * 1000.0)
                
        latencies.sort()
        median_ms = statistics.median(latencies)
        p95_ms = latencies[int(len(latencies) * 0.95)]
        min_ms = min(latencies)
        max_ms = max(latencies)
        tokens_per_sec = (seq_len / (median_ms / 1000.0))
        
        current_ram_mb = get_ram_mb()
        
        results["sequence_benchmarks"][str(seq_len)] = {
            "seq_len": seq_len,
            "median_ms": round(median_ms, 3),
            "p95_ms": round(p95_ms, 3),
            "min_ms": round(min_ms, 3),
            "max_ms": round(max_ms, 3),
            "tokens_per_sec": round(tokens_per_sec, 1),
            "ram_mb": round(current_ram_mb, 2)
        }
        print(
            f"T={seq_len:3d}: median={median_ms:.3f}ms | p95={p95_ms:.3f}ms | "
            f"throughput={tokens_per_sec:.1f} tok/s | RAM={current_ram_mb:.1f}MB"
        )
        
    return results


if __name__ == "__main__":
    torch.set_num_threads(4)
    print("Running ChakrMicro CPU Benchmark...")
    res = benchmark_cpu()
    os.makedirs("docs", exist_ok=True)
    with open("docs/step_04_cpu_benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print("Benchmark complete. Results saved to docs/step_04_cpu_benchmark_results.json")
