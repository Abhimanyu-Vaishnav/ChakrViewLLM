"""
release/chakrview-v0.6.0/benchmark_v060.py

Deterministic release benchmark measuring:
- Model load time (cold start)
- First-token latency
- Short-generation latency and tokens per second
- Peak process memory (via psutil / tracemalloc)
- Output determinism across repeated runs
- Smoke-test success rate

Strictly adheres to project safety rules: inference and evaluation only;
zero parameter modifications or training.
"""

import sys
import os
import json
import time
import platform
import hashlib
from pathlib import Path
import tracemalloc

# Ensure project root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from chakrview.runtime.neural_runtime_adapter import (
    NeuralRuntimeAdapter,
    CANONICAL_CHECKPOINTS_ALLOWLIST,
    EXPECTED_TOKENIZER_FILES_HASHES,
    DEFAULT_CHECKPOINT_KEY,
    compute_file_sha256,
)

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False


def run_benchmark():
    print("=" * 70)
    print("CHAKRVIEW v0.6.0 DETERMINISTIC INFERENCE BENCHMARK")
    print("=" * 70)
    
    tracemalloc.start()
    
    # Environment info
    env_info = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "processor": platform.processor(),
        "cpu_count_logical": os.cpu_count(),
        "date": "2026-10-19"
    }
    if _HAS_PSUTIL:
        mem = psutil.virtual_memory()
        env_info["system_ram_gb"] = round(mem.total / (1024 ** 3), 2)
        env_info["available_ram_gb"] = round(mem.available / (1024 ** 3), 2)

    # 1. Verification of Canonical Artifacts
    ckpt_path, expected_ckpt_hash, label = CANONICAL_CHECKPOINTS_ALLOWLIST[DEFAULT_CHECKPOINT_KEY]
    actual_ckpt_hash = compute_file_sha256(ckpt_path)
    assert actual_ckpt_hash == expected_ckpt_hash, f"Checkpoint hash mismatch: {actual_ckpt_hash}"
    
    tokenizer_dir = REPO_ROOT / "data" / "experiments" / "vocab_4096"
    for fname, exp_hash in EXPECTED_TOKENIZER_FILES_HASHES.items():
        fpath = tokenizer_dir / fname
        act_h = compute_file_sha256(fpath)
        assert act_h == exp_hash, f"Tokenizer file {fname} mismatch: {act_h}"
        
    print("[PASS] Canonical artifacts SHA-256 hashes verified bit-identical.")
    
    # 2. Cold-Start Model Load Time Measurement
    t0 = time.perf_counter()
    adapter = NeuralRuntimeAdapter(checkpoint_path=ckpt_path)
    adapter.initialize()
    t_load = (time.perf_counter() - t0) * 1000.0
    print(f"[METRIC] Cold Start Load Time: {t_load:.2f} ms")
    
    # 3. Deterministic Smoke Test Inputs
    benchmark_prompts = [
        "ChakrView is an indigenous",
        "The neural architecture of ChakrMicro",
        "Language tokenization forms the critical",
        "Research in modular artificial intelligence",
        "Autoregressive generation proceeds token by token"
    ]
    
    # 4. Latency and Tokens/Sec Measurement
    run_measurements = []
    smoke_success_count = 0
    
    # Warmup pass
    _ = adapter.generate("Warmup prompt for inference", max_new_tokens=8)
    
    for idx, prompt in enumerate(benchmark_prompts):
        # Measure first-token latency (single token generation)
        t_first_start = time.perf_counter()
        first_out = adapter.generate(prompt, max_new_tokens=1)
        t_first_latency = (time.perf_counter() - t_first_start) * 1000.0
        
        # Measure 32-token generation
        t_gen_start = time.perf_counter()
        gen_out = adapter.generate(prompt, max_new_tokens=32)
        t_gen_duration = (time.perf_counter() - t_gen_start)
        
        gen_tokens = gen_out["output_token_count"]
        toks_per_sec = gen_tokens / t_gen_duration if t_gen_duration > 0 else 0.0
        
        # Check non-empty and smoke success
        is_success = bool(gen_out["generated_text"] and len(gen_out["generated_tokens"]) == 32)
        if is_success:
            smoke_success_count += 1
            
        run_measurements.append({
            "prompt": prompt,
            "first_token_latency_ms": round(t_first_latency, 2),
            "generation_duration_sec": round(t_gen_duration, 4),
            "tokens_generated": gen_tokens,
            "tokens_per_second": round(toks_per_sec, 2),
            "generated_text_preview": gen_out["generated_text"][:60],
            "success": is_success
        })
        print(f"  Sample [{idx+1}/{len(benchmark_prompts)}] TTFT: {t_first_latency:.2f} ms | Speed: {toks_per_sec:.2f} tok/s | Success: {is_success}")

    # 5. Output Reproducibility Test (Greedy determinism across 3 repeated trials)
    test_prompt = "ChakrView research baseline"
    outputs = []
    for _ in range(3):
        out = adapter.generate(test_prompt, max_new_tokens=24, decoding="greedy")
        outputs.append(out["generated_text"])
        
    is_deterministic = (outputs[0] == outputs[1] == outputs[2])
    print(f"[CHECK] Greedy Output Determinism across 3 trials: {'PASS' if is_deterministic else 'FAIL'}")
    
    # Memory usage
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    process_rss_mb = None
    if _HAS_PSUTIL:
        proc = psutil.Process(os.getpid())
        process_rss_mb = round(proc.memory_info().rss / (1024 * 1024), 2)
        
    # Aggregate metrics
    avg_first_token_ms = round(sum(m["first_token_latency_ms"] for m in run_measurements) / len(run_measurements), 2)
    avg_toks_per_sec = round(sum(m["tokens_per_second"] for m in run_measurements) / len(run_measurements), 2)
    success_rate = f"{smoke_success_count}/{len(benchmark_prompts)} ({round(smoke_success_count/len(benchmark_prompts)*100, 1)}%)"
    
    results = {
        "metadata": {
            "version": "v0.6.0-rc1",
            "benchmark_name": "CHAKRVIEW_v060_CPU_INFERENCE_BENCHMARK",
            "environment": env_info,
            "canonical_artifacts": {
                "checkpoint": {
                    "path": str(ckpt_path),
                    "sha256": actual_ckpt_hash,
                    "verified": True
                },
                "tokenizer": {
                    "dir": str(tokenizer_dir),
                    "files_verified": True
                }
            }
        },
        "metrics": {
            "cold_start_load_ms": round(t_load, 2),
            "avg_first_token_latency_ms": avg_first_token_ms,
            "avg_tokens_per_second": avg_toks_per_sec,
            "peak_traced_memory_mb": round(peak_mem / (1024 * 1024), 2),
            "process_rss_memory_mb": process_rss_mb,
            "smoke_test_success_rate": success_rate,
            "greedy_determinism_verified": is_deterministic
        },
        "per_prompt_runs": run_measurements
    }
    
    out_path = REPO_ROOT / "release" / "chakrview-v0.6.0" / "BENCHMARK_RESULTS.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        
    print(f"\n[OK] Benchmark completed successfully. Saved to {out_path}")
    print("=" * 70)
    print(f"Summary: Cold Load: {t_load:.1f} ms | TTFT: {avg_first_token_ms:.1f} ms | Throughput: {avg_toks_per_sec:.1f} tok/s | Success: {success_rate}")
    print("=" * 70)
    return results


if __name__ == "__main__":
    run_benchmark()
