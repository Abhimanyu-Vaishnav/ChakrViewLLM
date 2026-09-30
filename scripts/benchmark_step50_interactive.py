"""
Benchmark: Step 50 Trained ChakrView Interactive Model Evaluation.

Measures:
  1. Startup time (ms)
  2. Checkpoint loading time (ms)
  3. Time to First Token (TTFT, ms)
  4. Token generation latency (ms/token)
  5. Throughput (tokens/sec)
  6. Memory RSS (MB)
  7. Context growth latency and token budgeting across turns
  8. Session reset latency (ms)
  9. Baseline immutability verification (Delta W = 0)

Saves results to: docs/STEP_50_BENCHMARK_RESULTS.json
"""

from __future__ import annotations

import gc
import json
import os
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Dict, List
import psutil
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from chakrview.runtime.interactive import (
    InteractiveModelSessionCoordinator,
    load_trained_checkpoint,
    compute_model_hash,
    instantiate_frozen_baseline,
    DEFAULT_TRAINED_CHECKPOINT,
    DEFAULT_TOKENIZER_DIR,
    EXPECTED_WEIGHT_HASH,
)

OUTPUT_FILE = ROOT_DIR / "docs" / "STEP_50_BENCHMARK_RESULTS.json"


def run_benchmark() -> Dict[str, Any]:
    print("=" * 72)
    print("CHAKRVIEW STEP 50: INTERACTIVE MODEL EVALUATION BENCHMARK")
    print("=" * 72)

    process = psutil.Process()
    gc.collect()
    rss_initial_mb = process.memory_info().rss / (1024.0 * 1024.0)

    # 1. Startup & Coordinator Initialization Latency
    print("\n[1/8] Measuring Coordinator Startup Latency...")
    t0 = time.perf_counter()
    coord = InteractiveModelSessionCoordinator(
        checkpoint_path=DEFAULT_TRAINED_CHECKPOINT,
        tokenizer_dir=DEFAULT_TOKENIZER_DIR,
        deterministic=True,
        seed=42,
    )
    startup_ms = (time.perf_counter() - t0) * 1000.0
    print(f"  Startup Latency: {startup_ms:.2f} ms")

    # 2. Checkpoint Loading Latency (isolated reload)
    print("\n[2/8] Measuring Isolated Checkpoint Loading Latency...")
    ckpt_times = []
    for _ in range(5):
        t0 = time.perf_counter()
        _model, _data, _hash = load_trained_checkpoint(DEFAULT_TRAINED_CHECKPOINT)
        ckpt_times.append((time.perf_counter() - t0) * 1000.0)
    ckpt_load_ms = statistics.mean(ckpt_times)
    print(f"  Checkpoint Load Latency (mean of 5): {ckpt_load_ms:.2f} ms")

    # 3. Time to First Token (TTFT)
    print("\n[3/8] Measuring Time to First Token (TTFT)...")
    prompt = "ChakrView is an indigenous neural architecture designed for"
    tokens = coord.tokenizer.encode(prompt, add_bos=True, add_eos=False)
    x = torch.tensor([tokens], dtype=torch.long)

    ttft_times = []
    for _ in range(10):
        t0 = time.perf_counter()
        with torch.no_grad():
            _logits = coord.trained_model(x)[:, -1, :]
        ttft_times.append((time.perf_counter() - t0) * 1000.0)
    ttft_mean_ms = statistics.mean(ttft_times)
    ttft_median_ms = statistics.median(ttft_times)
    print(f"  TTFT Mean: {ttft_mean_ms:.2f} ms | Median: {ttft_median_ms:.2f} ms")

    # 4. Token Generation Latency & Throughput
    print("\n[4/8] Measuring Token Generation Latency & Throughput (32 tokens)...")
    coord.gen_config.max_new_tokens = 32
    coord.gen_config.min_new_tokens = 32  # force full 32 tokens
    coord.reset_context()

    gen_latencies = []
    total_tokens_emitted = 0
    t_start_gen = time.perf_counter()

    for i in range(5):
        t0 = time.perf_counter()
        _text, metrics, _ = coord.generate_turn(f"Benchmark prompt turn {i + 1}")
        elapsed = time.perf_counter() - t0
        gen_latencies.append(elapsed * 1000.0)
        total_tokens_emitted += metrics.length

    total_gen_time_sec = time.perf_counter() - t_start_gen
    throughput_tok_per_sec = total_tokens_emitted / total_gen_time_sec if total_gen_time_sec > 0 else 0
    avg_ms_per_token = (sum(gen_latencies) / total_tokens_emitted) if total_tokens_emitted > 0 else 0

    print(f"  Avg Latency per Turn: {statistics.mean(gen_latencies):.2f} ms")
    print(f"  Avg Latency per Token: {avg_ms_per_token:.2f} ms/token")
    print(f"  Throughput: {throughput_tok_per_sec:.2f} tokens/sec")

    # 5. Context Growth Tracking
    print("\n[5/8] Measuring Context Growth Across Multi-Turn Dialogue...")
    coord.reset_context()
    context_snapshots = []
    for turn_idx in range(6):
        user_msg = f"Turn {turn_idx + 1}: Informative message containing key context item {turn_idx}."
        t0 = time.perf_counter()
        coord.generate_turn(user_msg)
        elapsed = (time.perf_counter() - t0) * 1000.0
        stats = coord.get_context_stats()
        context_snapshots.append({
            "turn_index": turn_idx + 1,
            "turns_count": stats["turns"],
            "tokens_in_context": stats["token_count"],
            "remaining_budget": stats["remaining_context_budget"],
            "turn_latency_ms": round(elapsed, 2),
        })
        print(f"  Turn {turn_idx + 1}: Context Tokens = {stats['token_count']} | Remaining = {stats['remaining_context_budget']} | Latency = {elapsed:.2f} ms")

    # 6. Session Reset Latency
    print("\n[6/8] Measuring Session Reset Latency...")
    reset_times = []
    for _ in range(20):
        t0 = time.perf_counter()
        coord.reset_context()
        reset_times.append((time.perf_counter() - t0) * 1000.0)
    reset_ms = statistics.mean(reset_times)
    print(f"  Reset Latency (mean of 20): {reset_ms:.4f} ms")

    # 7. Memory RSS Tracking
    print("\n[7/8] Measuring Peak Process RSS Memory...")
    gc.collect()
    rss_final_mb = process.memory_info().rss / (1024.0 * 1024.0)
    rss_delta_mb = rss_final_mb - rss_initial_mb
    print(f"  Initial RSS: {rss_initial_mb:.2f} MB")
    print(f"  Final RSS:   {rss_final_mb:.2f} MB (Delta: +{rss_delta_mb:.2f} MB)")

    # 8. Weight Invariance & Baseline Verification
    print("\n[8/8] Verifying Weight Invariance (Delta W = 0)...")
    trained_hash = compute_model_hash(coord.trained_model)
    baseline_hash = compute_model_hash(coord.baseline_model)

    assert baseline_hash == EXPECTED_WEIGHT_HASH, "Frozen baseline modified!"
    assert trained_hash == coord.trained_hash, "Trained model modified!"
    print(f"  Baseline Hash: {baseline_hash} (Delta W = 0 verified)")
    print(f"  Trained Hash:  {trained_hash} (Delta W = 0 verified)")

    results: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_architecture": "ChakrMicro v0.1",
        "parameters": 3_443_136,
        "vocabulary_size": 4_096,
        "max_context_length": 512,
        "selected_checkpoint": str(DEFAULT_TRAINED_CHECKPOINT),
        "trained_weight_hash": trained_hash,
        "frozen_baseline_hash": baseline_hash,
        "performance": {
            "startup_latency_ms": round(startup_ms, 2),
            "checkpoint_load_latency_ms": round(ckpt_load_ms, 2),
            "ttft_mean_ms": round(ttft_mean_ms, 3),
            "ttft_median_ms": round(ttft_median_ms, 3),
            "latency_ms_per_token": round(avg_ms_per_token, 2),
            "throughput_tokens_per_sec": round(throughput_tok_per_sec, 2),
            "session_reset_latency_ms": round(reset_ms, 4),
        },
        "memory": {
            "initial_rss_mb": round(rss_initial_mb, 2),
            "final_rss_mb": round(rss_final_mb, 2),
            "rss_delta_mb": round(rss_delta_mb, 2),
        },
        "context_growth": context_snapshots,
        "weight_invariance": {
            "baseline_hash_intact": (baseline_hash == EXPECTED_WEIGHT_HASH),
            "trained_hash_intact": (trained_hash == coord.trained_hash),
        },
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nBenchmark completed successfully. Saved to: {OUTPUT_FILE}")
    print("=" * 72)
    return results


if __name__ == "__main__":
    run_benchmark()
