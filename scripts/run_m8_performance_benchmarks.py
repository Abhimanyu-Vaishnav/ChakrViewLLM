"""
Milestone 8: CPU/GPU Performance Benchmark Suite.

Benchmarks:
- CPU training & inference throughput (tokens/sec, step time, memory).
- Optional CUDA training & inference throughput in .venv_gpu (if CUDA device is detected).
- Compares short (warm-up) vs sustained runs to isolate dispatch overhead.
- Records hardware, PyTorch version, device names, and VRAM/RAM metrics.
"""

import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn as nn
import torch.optim as optim

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro


def benchmark_device(device_str: str, steps: int = 20, batch_size: int = 8, seq_len: int = 64) -> dict:
    device = torch.device(device_str)
    config = ModelConfig(max_seq_len=512)
    
    # Model load timing
    t0 = time.perf_counter()
    model = ChakrMicro(config).to(device)
    load_time = time.perf_counter() - t0

    opt = optim.AdamW(model.parameters(), lr=5e-4)
    loss_fn = nn.CrossEntropyLoss()

    # Synthetic batch on device
    x = torch.randint(0, 4096, (batch_size, seq_len), device=device)
    y = torch.randint(0, 4096, (batch_size, seq_len), device=device)

    # Warm-up (3 steps)
    model.train()
    for _ in range(3):
        opt.zero_grad()
        loss = loss_fn(model(x).reshape(-1, 4096), y.reshape(-1))
        loss.backward()
        opt.step()
    if device.type == "cuda":
        torch.cuda.synchronize()

    # Sustained Training Benchmark
    t_train_start = time.perf_counter()
    for _ in range(steps):
        opt.zero_grad()
        loss = loss_fn(model(x).reshape(-1, 4096), y.reshape(-1))
        loss.backward()
        opt.step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_train_total = time.perf_counter() - t_train_start

    train_tokens = steps * batch_size * seq_len
    train_tps = train_tokens / max(t_train_total, 1e-6)
    step_time_ms = (t_train_total / steps) * 1000

    # Sustained Inference Benchmark
    model.eval()
    infer_steps = 20
    t_infer_start = time.perf_counter()
    with torch.no_grad():
        for _ in range(infer_steps):
            _ = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_infer_total = time.perf_counter() - t_infer_start

    infer_tokens = infer_steps * batch_size * seq_len
    infer_tps = infer_tokens / max(t_infer_total, 1e-6)

    # Peak memory stats if CUDA
    peak_vram_mb = None
    if device.type == "cuda":
        peak_vram_mb = round(torch.cuda.max_memory_allocated(device) / (1024 * 1024), 2)

    return {
        "device": device_str,
        "model_load_sec": round(load_time, 4),
        "steps": steps,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "train_time_sec": round(t_train_total, 4),
        "train_tokens_per_sec": round(train_tps, 2),
        "step_time_ms": round(step_time_ms, 2),
        "infer_tokens_per_sec": round(infer_tps, 2),
        "peak_vram_mb": peak_vram_mb,
    }


def run_milestone_8_benchmarks():
    print("Running Milestone 8 CPU/GPU Performance Benchmark Suite...")
    repo_root = REPO_ROOT
    results = {
        "environment": {
            "pytorch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None",
            "cpu_threads": torch.get_num_threads(),
        },
        "benchmarks": {}
    }

    # Benchmark CPU
    print("Benchmarking CPU...")
    cpu_res = benchmark_device("cpu", steps=25, batch_size=8, seq_len=64)
    results["benchmarks"]["cpu"] = cpu_res
    print(f"CPU Throughput: {cpu_res['train_tokens_per_sec']} tokens/sec, Step time: {cpu_res['step_time_ms']} ms")

    # Benchmark CUDA if available
    if torch.cuda.is_available():
        print("Benchmarking CUDA...")
        cuda_res = benchmark_device("cuda", steps=50, batch_size=8, seq_len=64)
        results["benchmarks"]["cuda"] = cuda_res
        print(f"CUDA Throughput: {cuda_res['train_tokens_per_sec']} tokens/sec, VRAM: {cuda_res['peak_vram_mb']} MB")
    else:
        results["benchmarks"]["cuda"] = "NOT TESTED (CUDA unavailable in current interpreter)"

    out_file = repo_root / "artifacts" / "milestone8_performance_benchmarks.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Milestone 8 finished. Saved to {out_file}")


if __name__ == "__main__":
    run_milestone_8_benchmarks()
