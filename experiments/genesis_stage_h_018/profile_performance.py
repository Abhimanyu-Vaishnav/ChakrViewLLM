"""
Benchmark Latency, Throughput and Memory Profiler for NR-018
Measures:
1. Pure Neural Stage-D Latency (ms/sample)
2. Neural Router Latency (ms/sample)
3. Deterministic Executor Latency (ms/sample)
4. Memory (VRAM / RAM) footprints
5. Parameter counts and model size on disk
"""

import time
import os
import torch
import json
from chakrview.brain.genesis.chakr_genesis_17m import ChakrGenesis17M, Genesis17MConfig
from chakrview.runtime.restricted_executor import RestrictedExecutionEngine

def profile_performance():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Parameter counts & disk sizes
    staged_path = "experiments/genesis_stage_d_011/checkpoints/genesis_17m_staged_best.pt"
    router_path = "experiments/genesis_stage_h_018/neural_router_17m.pt"
    staged_sz_mb = os.path.getsize(staged_path) / (1024 * 1024)
    router_sz_mb = os.path.getsize(router_path) / (1024 * 1024)

    st = torch.load(staged_path, map_location=device, weights_only=False)
    raw_cfg = st['config']
    cfg = raw_cfg if isinstance(raw_cfg, Genesis17MConfig) else Genesis17MConfig(**raw_cfg)
    model = ChakrGenesis17M(cfg).to(device)
    model.load_state_dict(st['model_state_dict'])
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())

    # Measure memory
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        vram_allocated_mb = torch.cuda.memory_allocated() / (1024 * 1024)
        vram_reserved_mb = torch.cuda.memory_reserved() / (1024 * 1024)
    else:
        vram_allocated_mb = 0
        vram_reserved_mb = 0

    # 2. Benchmark Executor Latency
    executor = RestrictedExecutionEngine()
    test_req = {"operation": "math_mul", "arguments": {"a": 12345, "b": 6789}}
    t0 = time.perf_counter()
    N_EXEC = 5000
    for _ in range(N_EXEC):
        executor.execute(test_req)
    t1 = time.perf_counter()
    exec_latency_ms = ((t1 - t0) / N_EXEC) * 1000.0

    # 3. Benchmark Neural Forward Latency (tokens = 32)
    dummy_input = torch.randint(0, cfg.vocab_size, (1, 32), device=device)
    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = model(dummy_input)
    if device == "cuda": torch.cuda.synchronize()

    t0 = time.perf_counter()
    N_NEURAL = 200
    for _ in range(N_NEURAL):
        with torch.no_grad():
            _ = model(dummy_input)
    if device == "cuda": torch.cuda.synchronize()
    t1 = time.perf_counter()
    neural_forward_latency_ms = ((t1 - t0) / N_NEURAL) * 1000.0

    profile_data = {
        "device": device,
        "total_parameters": total_params,
        "checkpoint_disk_mb": {
            "stage_d_best": round(staged_sz_mb, 2),
            "neural_router": round(router_sz_mb, 2)
        },
        "vram_allocated_mb": round(vram_allocated_mb, 2),
        "vram_reserved_mb": round(vram_reserved_mb, 2),
        "executor_latency_ms_per_op": round(exec_latency_ms, 4),
        "neural_forward_latency_ms_32tokens": round(neural_forward_latency_ms, 2),
        "executor_speedup_vs_neural_forward": round(neural_forward_latency_ms / max(exec_latency_ms, 1e-6), 1)
    }

    with open("data/stage_h_018/performance_profile.json", "w", encoding="utf-8") as f:
        json.dump(profile_data, f, indent=2)

    print("=== PERFORMANCE & COST PROFILE ===")
    print(json.dumps(profile_data, indent=2))

if __name__ == "__main__":
    profile_performance()
