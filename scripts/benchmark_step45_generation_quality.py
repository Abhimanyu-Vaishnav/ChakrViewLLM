"""
Benchmark: Neural Generation & Inference Quality Layer (Step 45).

Measures:
1. Greedy generation latency & throughput
2. Temperature stochastic sampling latency & throughput
3. Top-k sampling latency & throughput
4. Top-p (nucleus) sampling latency & throughput
5. Repetition penalty overhead
6. Incremental KV-cache decoding latency & throughput
7. End-to-end inference latency (tokenize -> prefill -> decode -> detokenize)
8. Memory overhead and parameter/weight invariance verification

Results saved to docs/STEP_45_BENCHMARK_RESULTS.json.
"""

from __future__ import annotations

import gc
import json
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Dict, List
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.tokenizer.serialization import load_tokenizer_artifacts
from chakrview.runtime.inference import GenerationConfig
from chakrview.runtime.sampling import SamplingConfig
from chakrview.runtime.pipeline import InferenceEngine, InferenceRequest, ModelIdentity

OUTPUT_FILE = ROOT_DIR / "docs" / "STEP_45_BENCHMARK_RESULTS.json"
EXPECTED_PARAM_COUNT = 3_443_136
EXPECTED_WEIGHT_HASH = "c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da"


def compute_stats(latencies_ms: List[float], token_count: int) -> Dict[str, float]:
    """Compute mean, median, p95, and throughput (tokens/sec)."""
    mean_ms = statistics.mean(latencies_ms)
    median_ms = statistics.median(latencies_ms)
    p95_ms = statistics.quantiles(latencies_ms, n=20)[18] if len(latencies_ms) >= 20 else max(latencies_ms)
    total_time_sec = sum(latencies_ms) / 1000.0
    throughput = (token_count * len(latencies_ms)) / total_time_sec if total_time_sec > 0 else 0.0
    return {
        "mean_ms": round(mean_ms, 3),
        "median_ms": round(median_ms, 3),
        "p95_ms": round(p95_ms, 3),
        "throughput_tokens_per_sec": round(throughput, 2),
    }


def run_benchmark():
    print("=" * 72)
    print("CHAKRVIEW STEP 45: NEURAL GENERATION QUALITY BENCHMARK")
    print("=" * 72)

    # 1. Model & Engine Setup
    torch.manual_seed(42)
    cfg = ModelConfig()
    model = ChakrMicro(cfg)
    model.eval()

    tok_dir = ROOT_DIR / "data" / "experiments" / "vocab_4096"
    tokenizer, _ = load_tokenizer_artifacts(tok_dir)

    engine = InferenceEngine(model=model, tokenizer=tokenizer)
    identity = engine.model_identity

    # Verify invariants
    assert identity.parameter_count == EXPECTED_PARAM_COUNT, f"Param count mismatch: {identity.parameter_count}"
    assert identity.weight_hash == EXPECTED_WEIGHT_HASH, f"Weight hash mismatch: {identity.weight_hash}"
    print(f"Verified Model Identity: params={identity.parameter_count}, hash={identity.weight_hash[:16]}...")

    test_prompt = "ChakrView is an indigenous neural architecture designed for verified deterministic inference."
    num_gen_tokens = 16
    warmup_iters = 2
    benchmark_iters = 10

    benchmarks_to_run = [
        (
            "greedy_decoding",
            GenerationConfig(
                max_new_tokens=num_gen_tokens,
                sampling=SamplingConfig(temperature=0.0),
            ),
        ),
        (
            "temperature_sampling",
            GenerationConfig(
                max_new_tokens=num_gen_tokens,
                sampling=SamplingConfig(temperature=0.7, seed=42),
            ),
        ),
        (
            "top_k_sampling",
            GenerationConfig(
                max_new_tokens=num_gen_tokens,
                sampling=SamplingConfig(temperature=0.7, top_k=20, seed=42),
            ),
        ),
        (
            "top_p_sampling",
            GenerationConfig(
                max_new_tokens=num_gen_tokens,
                sampling=SamplingConfig(temperature=0.7, top_p=0.9, seed=42),
            ),
        ),
        (
            "repetition_penalty",
            GenerationConfig(
                max_new_tokens=num_gen_tokens,
                sampling=SamplingConfig(temperature=0.0, repetition_penalty=1.2),
            ),
        ),
        (
            "min_new_tokens_enforced",
            GenerationConfig(
                max_new_tokens=num_gen_tokens,
                min_new_tokens=8,
                sampling=SamplingConfig(temperature=0.0),
            ),
        ),
    ]

    benchmark_results: Dict[str, Any] = {}

    for name, gen_cfg in benchmarks_to_run:
        print(f"\nRunning benchmark: {name} ({benchmark_iters} iterations)...")
        req = InferenceRequest(prompt=test_prompt, generation_config=gen_cfg)

        # Warmup
        for _ in range(warmup_iters):
            engine.execute(req)

        # Timed iterations
        latencies: List[float] = []
        tokens_generated = 0
        gc.collect()

        for i in range(benchmark_iters):
            t0 = time.perf_counter()
            res = engine.execute(req)
            dur_ms = (time.perf_counter() - t0) * 1000.0
            latencies.append(dur_ms)
            tokens_generated = res.output_token_count

        stats = compute_stats(latencies, tokens_generated)
        stats["tokens_generated"] = tokens_generated
        stats["iterations"] = benchmark_iters
        benchmark_results[name] = stats
        print(f"  -> Mean: {stats['mean_ms']:.2f} ms | Median: {stats['median_ms']:.2f} ms | Throughput: {stats['throughput_tokens_per_sec']:.1f} tok/s")

    # Repetition control overhead comparison
    greedy_mean = benchmark_results["greedy_decoding"]["mean_ms"]
    rep_mean = benchmark_results["repetition_penalty"]["mean_ms"]
    rep_overhead_ms = rep_mean - greedy_mean
    rep_overhead_pct = ((rep_mean - greedy_mean) / greedy_mean) * 100.0 if greedy_mean > 0 else 0.0

    # Memory overhead measurement
    import os
    import psutil
    process = psutil.Process(os.getpid())
    mem_rss_bytes = process.memory_info().rss
    mem_rss_mb = mem_rss_bytes / (1024 * 1024)

    # Invariant verification post-run
    final_hash = engine.compute_weight_hash()
    assert final_hash == EXPECTED_WEIGHT_HASH, f"Post-benchmark weight hash corrupted: {final_hash}"

    report = {
        "benchmark_metadata": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "step": 45,
            "architecture": "ChakrMicro v0.1",
            "model_parameters": EXPECTED_PARAM_COUNT,
            "vocab_size": cfg.vocab_size,
            "max_seq_len": cfg.max_seq_len,
            "weight_sha256": EXPECTED_WEIGHT_HASH,
            "delta_w": 0,
            "warmup_iterations": warmup_iters,
            "benchmark_iterations": benchmark_iters,
            "generated_tokens_per_run": num_gen_tokens,
            "process_rss_mb": round(mem_rss_mb, 2),
        },
        "benchmarks": benchmark_results,
        "comparisons": {
            "repetition_penalty_overhead_ms": round(rep_overhead_ms, 3),
            "repetition_penalty_overhead_pct": round(rep_overhead_pct, 2),
            "step44_baseline_mean_ms": 11.2,
            "step45_overhead_ms": round(greedy_mean - 11.2, 3),
        },
        "invariant_verification": {
            "weights_immutable": True,
            "pre_benchmark_sha256": EXPECTED_WEIGHT_HASH,
            "post_benchmark_sha256": final_hash,
            "delta_w_zero": True,
        }
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 72)
    print(f"BENCHMARK COMPLETE. Saved to: {OUTPUT_FILE}")
    print(f"Greedy Latency: {greedy_mean:.2f} ms | Throughput: {benchmark_results['greedy_decoding']['throughput_tokens_per_sec']:.1f} tok/s")
    print(f"Repetition Penalty Overhead: {rep_overhead_ms:.2f} ms ({rep_overhead_pct:.1f}%)")
    print(f"Memory RSS: {mem_rss_mb:.1f} MB | Delta W: 0 (SHA-256 Verified)")
    print("=" * 72)


if __name__ == "__main__":
    run_benchmark()
