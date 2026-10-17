"""
Benchmarking script comparing CPU and GPU execution modes for ChakrMicro under identical conditions.
"""

import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Dict, Any, List

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chakrview.brain.model import ChakrMicro
from chakrview.tokenizer.serialization import load_tokenizer_artifacts

CANONICAL_CHECKPOINT_PATH = REPO_ROOT / "experiments" / "batch16_21_combined" / "main_training_10000" / "training_checkpoint_step_10000.pt"
TOKENIZER_DIR = REPO_ROOT / "release" / "chakrview-v0.5.0" / "data" / "experiments" / "vocab_4096"
OUTPUT_DIR = REPO_ROOT / "experiments" / "batch2181_2220"

PROMPTS = [
    "The fundamental principle of information theory states that",
    "In coordinate geometry, the equation of a straight line can be expressed as",
    "def calculate_factorial(n: int) -> int:\n    \"\"\"Compute factorial recursively.\"\"\"",
    "ChakrView neural core architecture operates with",
    "Mathematics in ancient India contributed significantly to"
]


def benchmark_mode(device_name: str, runs: int = 5) -> Dict[str, Any]:
    print(f"\nBenchmarking Execution Mode: {device_name.upper()}")
    device = torch.device(device_name)
    
    # 1. Measure Model Load Time
    t0 = time.perf_counter()
    canonical_ckpt = torch.load(CANONICAL_CHECKPOINT_PATH, map_location=device, weights_only=False)
    config = canonical_ckpt["config"]
    model = ChakrMicro(config)
    model.load_state_dict(canonical_ckpt["model_state_dict"], strict=True)
    model.eval()
    model.to(device)
    load_time_sec = time.perf_counter() - t0
    
    # 2. Tokenizer
    tokenizer, _ = load_tokenizer_artifacts(TOKENIZER_DIR)
    
    # 3. Warmup
    warmup_toks = torch.tensor([[10, 20, 30, 40, 50]], dtype=torch.long, device=device)
    for _ in range(3):
        with torch.no_grad():
            _ = model(warmup_toks)
    if device_name.startswith("cuda"):
        torch.cuda.synchronize()

    # 4. Prefill Benchmark (Prompt Processing)
    prefill_latencies = []
    prefill_tok_counts = []
    
    for p in PROMPTS:
        toks = tokenizer.encode(p)
        inp = torch.tensor([toks], dtype=torch.long, device=device)
        if device_name.startswith("cuda"):
            torch.cuda.synchronize()
        t_start = time.perf_counter()
        with torch.no_grad():
            _ = model(inp)
        if device_name.startswith("cuda"):
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t_start
        prefill_latencies.append(elapsed)
        prefill_tok_counts.append(len(toks))
        
    avg_prefill_throughput = sum(prefill_tok_counts) / sum(prefill_latencies)
    avg_prefill_latency_ms = (sum(prefill_latencies) / len(prefill_latencies)) * 1000.0

    # 5. Autoregressive Generation Benchmark (32 tokens per prompt)
    gen_latencies = []
    tokens_generated = 32
    time_to_first_token_list = []
    
    for p in PROMPTS:
        toks = tokenizer.encode(p)
        curr = torch.tensor([toks], dtype=torch.long, device=device)
        
        # Measure TTFT
        if device_name.startswith("cuda"):
            torch.cuda.synchronize()
        t_ttft_start = time.perf_counter()
        with torch.no_grad():
            out = model(curr)
        if device_name.startswith("cuda"):
            torch.cuda.synchronize()
        ttft = time.perf_counter() - t_ttft_start
        time_to_first_token_list.append(ttft)
        
        # Generation loop
        t_gen_start = time.perf_counter()
        for _ in range(tokens_generated):
            with torch.no_grad():
                logits = model(curr)
                next_tok = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                curr = torch.cat([curr, next_tok], dim=-1)
        if device_name.startswith("cuda"):
            torch.cuda.synchronize()
        gen_time = time.perf_counter() - t_gen_start
        gen_latencies.append(gen_time)

    avg_gen_throughput = (len(PROMPTS) * tokens_generated) / sum(gen_latencies)
    avg_gen_latency_ms = (sum(gen_latencies) / len(gen_latencies)) * 1000.0
    avg_ttft_ms = (sum(time_to_first_token_list) / len(time_to_first_token_list)) * 1000.0

    # Memory Tracking
    vram_peak_mb = 0.0
    if device_name.startswith("cuda"):
        vram_peak_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)

    return {
        "device": device_name,
        "model_load_time_ms": round(load_time_sec * 1000.0, 2),
        "prefill_throughput_tok_per_sec": round(avg_prefill_throughput, 2),
        "prefill_latency_avg_ms": round(avg_prefill_latency_ms, 2),
        "generation_throughput_tok_per_sec": round(avg_gen_throughput, 2),
        "generation_latency_32tok_avg_ms": round(avg_gen_latency_ms, 2),
        "time_to_first_token_avg_ms": round(avg_ttft_ms, 2),
        "peak_vram_mb": round(vram_peak_mb, 2)
    }


def run_hardware_benchmarks():
    results = {}
    
    # 1. CPU Mode
    results["cpu"] = benchmark_mode("cpu")
    
    # 2. CUDA GPU Mode
    if torch.cuda.is_available():
        results["cuda"] = benchmark_mode("cuda")
    else:
        results["cuda"] = "NOT RUN (CUDA unavailable)"
        
    out_path = OUTPUT_DIR / "HARDWARE_CAPABILITY_MATRIX.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        
    print(f"\nHardware benchmarks written to: {out_path}")
    return results


if __name__ == "__main__":
    run_hardware_benchmarks()
