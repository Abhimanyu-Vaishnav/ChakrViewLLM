"""
Inference-appropriate benchmarking script for ChakrView.
Uses torch.inference_mode(), warmups, repeated iterations, median/variability tracking,
CUDA synchronization, and measures both prefill, TTFT, single-token generation,
and batched forward throughput.
"""

from __future__ import annotations

import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import psutil
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chakrview.brain.model import ChakrMicro
from chakrview.tokenizer.serialization import load_tokenizer_artifacts

CANONICAL_CHECKPOINT_PATH = REPO_ROOT / "experiments" / "batch16_21_combined" / "main_training_10000" / "training_checkpoint_step_10000.pt"
TOKENIZER_DIR = REPO_ROOT / "release" / "chakrview-v0.5.0" / "data" / "experiments" / "vocab_4096"
OUTPUT_DIR = REPO_ROOT / "experiments" / "batch2221_2260"

PROMPTS = [
    "The fundamental principle of information theory states that",
    "In coordinate geometry, the equation of a straight line can be expressed as",
    "def calculate_factorial(n: int) -> int:\n    \"\"\"Compute factorial recursively.\"\"\"",
    "ChakrView neural core architecture operates with",
    "Mathematics in ancient India contributed significantly to"
]


def benchmark_device(device_name: str, repetitions: int = 5) -> Dict[str, Any]:
    print(f"\nBenchmarking Device: {device_name.upper()}")
    device = torch.device(device_name)
    is_cuda = device_name.startswith("cuda")
    
    # 1. Model Initialization Time (measured across 3 cold loads)
    load_times = []
    for _ in range(3):
        t0 = time.perf_counter()
        ckpt = torch.load(CANONICAL_CHECKPOINT_PATH, map_location=device, weights_only=False)
        m = ChakrMicro(ckpt["config"])
        m.load_state_dict(ckpt["model_state_dict"], strict=True)
        m.eval()
        m.to(device)
        if is_cuda:
            torch.cuda.synchronize()
        load_times.append(time.perf_counter() - t0)
    model = m
    
    # 2. Tokenizer
    tokenizer, _ = load_tokenizer_artifacts(TOKENIZER_DIR)
    
    # 3. Warm-up
    warm_inp = torch.randint(0, 4096, (1, 64), dtype=torch.long, device=device)
    for _ in range(5):
        with torch.inference_mode():
            _ = model(warm_inp)
    if is_cuda:
        torch.cuda.synchronize()

    # 4. Prompt Prefill & Time-to-First-Token (TTFT)
    ttft_records = []
    prefill_throughput_records = []
    
    for _ in range(repetitions):
        for p in PROMPTS:
            toks = tokenizer.encode(p)
            inp = torch.tensor([toks], dtype=torch.long, device=device)
            if is_cuda:
                torch.cuda.synchronize()
            t_start = time.perf_counter()
            with torch.inference_mode():
                out = model(inp)
                first_tok = torch.argmax(out[:, -1, :], dim=-1)
            if is_cuda:
                torch.cuda.synchronize()
            dur = time.perf_counter() - t_start
            ttft_records.append(dur * 1000.0)
            prefill_throughput_records.append(len(toks) / dur)

    # 5. Autoregressive Generation Throughput (batch=1, 32 tokens)
    tokens_to_gen = 32
    gen_throughput_records = []
    gen_latency_records = []
    
    for _ in range(repetitions):
        for p in PROMPTS:
            toks = tokenizer.encode(p)
            curr = torch.tensor([toks], dtype=torch.long, device=device)
            if is_cuda:
                torch.cuda.synchronize()
            t_gen_start = time.perf_counter()
            for _ in range(tokens_to_gen):
                with torch.inference_mode():
                    logits = model(curr)
                    nxt = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                    curr = torch.cat([curr, nxt], dim=-1)
            if is_cuda:
                torch.cuda.synchronize()
            gen_dur = time.perf_counter() - t_gen_start
            gen_latency_records.append(gen_dur * 1000.0)
            gen_throughput_records.append(tokens_to_gen / gen_dur)

    # 6. Batched Forward Throughput (Batch 1, 4, 16, 32 at Context 256)
    batched_results = {}
    for b_size in [1, 4, 16, 32]:
        x = torch.randint(0, 4096, (b_size, 256), dtype=torch.long, device=device)
        # warm
        for _ in range(3):
            with torch.inference_mode():
                _ = model(x)
        if is_cuda:
            torch.cuda.synchronize()
            
        b_latencies = []
        for _ in range(10):
            t_b_start = time.perf_counter()
            with torch.inference_mode():
                _ = model(x)
            if is_cuda:
                torch.cuda.synchronize()
            b_latencies.append((time.perf_counter() - t_b_start) * 1000.0)
            
        med_lat = float(np.median(b_latencies))
        throughput_tok_per_sec = (b_size * 256) / (med_lat / 1000.0)
        batched_results[f"batch_{b_size}"] = {
            "median_latency_ms": round(med_lat, 2),
            "throughput_tok_per_sec": round(throughput_tok_per_sec, 2)
        }

    # 7. Memory Tracking
    vram_peak_mb = 0.0
    if is_cuda:
        vram_peak_mb = float(torch.cuda.max_memory_allocated(device) / (1024 * 1024))
    ram_used_mb = float(psutil.Process().memory_info().rss / (1024 * 1024))

    return {
        "device": device_name,
        "model_load_latency_ms": {
            "median": round(float(np.median(load_times)) * 1000.0, 2),
            "min": round(float(np.min(load_times)) * 1000.0, 2),
            "max": round(float(np.max(load_times)) * 1000.0, 2)
        },
        "time_to_first_token_ms": {
            "median": round(float(np.median(ttft_records)), 2),
            "std": round(float(np.std(ttft_records)), 2),
            "min": round(float(np.min(ttft_records)), 2),
            "max": round(float(np.max(ttft_records)), 2)
        },
        "prefill_throughput_tok_per_sec": {
            "median": round(float(np.median(prefill_throughput_records)), 2),
            "mean": round(float(np.mean(prefill_throughput_records)), 2)
        },
        "autoregressive_gen_b1_tok_per_sec": {
            "median": round(float(np.median(gen_throughput_records)), 2),
            "std": round(float(np.std(gen_throughput_records)), 2)
        },
        "autoregressive_32tok_latency_ms": {
            "median": round(float(np.median(gen_latency_records)), 2),
            "std": round(float(np.std(gen_latency_records)), 2)
        },
        "batched_forward_pass_ctx256": batched_results,
        "peak_vram_allocated_mb": round(vram_peak_mb, 2),
        "peak_host_rss_mb": round(ram_used_mb, 2)
    }


def run_benchmarks():
    print("=" * 70)
    print("CHAKRVIEW INFERENCE-APPROPRIATE HARDWARE BENCHMARKS — BATCH 2221-2260")
    print("=" * 70)
    
    benchmarks = {
        "metadata": {
            "cpu_model": platform.processor(),
            "cpu_logical_cores": os.cpu_count(),
            "ram_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "gpu_model": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None",
            "cuda_version": torch.version.cuda,
            "pytorch_version": torch.__version__,
            "inference_mode": True
        },
        "devices": {}
    }
    
    # 1. CPU Benchmark
    benchmarks["devices"]["cpu"] = benchmark_device("cpu", repetitions=3)
    
    # 2. CUDA GPU Benchmark
    if torch.cuda.is_available():
        benchmarks["devices"]["cuda"] = benchmark_device("cuda", repetitions=3)
    else:
        benchmarks["devices"]["cuda"] = "NOT RUN (CUDA unavailable)"
        
    out_file = OUTPUT_DIR / "HARDWARE_BENCHMARKS.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(benchmarks, f, indent=2)
        
    print(f"\nInference benchmarks written to: {out_file}")
    return benchmarks


if __name__ == "__main__":
    run_benchmarks()
