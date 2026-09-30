"""
Benchmark: Local Model Runtime & Interactive Streaming Session Layer (Step 46).

Empirically measures:
1. Streaming Time to First Token (TTFT, ms)
2. Streaming Inter-Token Latency (ITL, ms)
3. Streaming generation throughput (tokens/sec)
4. Multi-turn conversation round-trip latency & throughput
5. Context compaction overhead under deep multi-turn dialogue
6. Process Memory RSS (MB)
7. Cryptographic weight invariance (ΔW = 0)

Results saved to docs/STEP_46_BENCHMARK_RESULTS.json.
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

from chakrview.runtime.inference import GenerationConfig
from chakrview.runtime.local_runtime import (
    EXPECTED_PARAM_COUNT,
    EXPECTED_WEIGHT_HASH,
    LocalModelRuntime,
)
from chakrview.runtime.sampling import SamplingConfig

OUTPUT_FILE = ROOT_DIR / "docs" / "STEP_46_BENCHMARK_RESULTS.json"


def compute_latency_stats(latencies_ms: List[float]) -> Dict[str, float]:
    """Compute mean, median, p95, min, max for a list of latencies."""
    if not latencies_ms:
        return {"mean_ms": 0.0, "median_ms": 0.0, "p95_ms": 0.0, "min_ms": 0.0, "max_ms": 0.0}
    mean_ms = statistics.mean(latencies_ms)
    median_ms = statistics.median(latencies_ms)
    p95_ms = statistics.quantiles(latencies_ms, n=20)[18] if len(latencies_ms) >= 20 else max(latencies_ms)
    return {
        "mean_ms": round(mean_ms, 3),
        "median_ms": round(median_ms, 3),
        "p95_ms": round(p95_ms, 3),
        "min_ms": round(min(latencies_ms), 3),
        "max_ms": round(max(latencies_ms), 3),
    }


def run_benchmark():
    print("=" * 72)
    print("CHAKRVIEW STEP 46: LOCAL MODEL RUNTIME BENCHMARK")
    print("=" * 72)

    # 1. Initialize Runtime
    runtime = LocalModelRuntime.from_default()
    runtime.verify_runtime_integrity()
    ident = runtime.model_identity
    print(f"Verified Model Identity: params={ident.parameter_count:,}, hash={ident.weight_hash[:16]}...")

    num_iterations = 10
    tokens_per_stream = 16
    warmup_iters = 2
    prompt = "ChakrView is an indigenous neural architecture designed for deterministic inference."

    # 2. Benchmark Streaming TTFT & Inter-Token Latency (ITL)
    print(f"\n[1/3] Benchmarking Streaming Latency ({num_iterations} runs, {tokens_per_stream} tokens/run)...")
    gen_cfg = GenerationConfig(
        max_new_tokens=tokens_per_stream,
        sampling=SamplingConfig(temperature=0.0),  # Greedy baseline
    )

    # Warmup
    for _ in range(warmup_iters):
        list(runtime.stream_generate(prompt=prompt, generation_config=gen_cfg))

    ttft_list: List[float] = []
    all_itl_list: List[float] = []
    stream_total_latencies: List[float] = []

    for _ in range(num_iterations):
        t_start = time.perf_counter()
        t_first: float = 0.0
        last_t = t_start
        itl_list: List[float] = []

        chunk_count = 0
        for chunk in runtime.stream_generate(prompt=prompt, generation_config=gen_cfg):
            now = time.perf_counter()
            if chunk_count == 0:
                t_first = now
                ttft_list.append((t_first - t_start) * 1000.0)
            else:
                itl_list.append((now - last_t) * 1000.0)
            last_t = now
            chunk_count += 1

        total_dur = (time.perf_counter() - t_start) * 1000.0
        stream_total_latencies.append(total_dur)
        all_itl_list.extend(itl_list)

    ttft_stats = compute_latency_stats(ttft_list)
    itl_stats = compute_latency_stats(all_itl_list)
    total_sec = sum(stream_total_latencies) / 1000.0
    stream_throughput = (num_iterations * tokens_per_stream) / total_sec if total_sec > 0 else 0.0

    print(f"  -> TTFT (Time to First Token): Mean {ttft_stats['mean_ms']:.2f} ms | Median {ttft_stats['median_ms']:.2f} ms")
    print(f"  -> ITL (Inter-Token Latency): Mean {itl_stats['mean_ms']:.2f} ms | Median {itl_stats['median_ms']:.2f} ms")
    print(f"  -> Streaming Throughput: {stream_throughput:.1f} tok/s")

    # 3. Benchmark Multi-Turn Stateful Chat
    print(f"\n[2/3] Benchmarking Multi-Turn Chat Conversation (8 continuous turns)...")
    session_id = "bench_multi_turn_session"
    runtime.reset_session(session_id)
    chat_turn_latencies: List[float] = []

    conversation_prompts = [
        "Explain causal transformers.",
        "How do tied embeddings save parameters?",
        "What are the benefits of Pre-RMSNorm over Post-LN?",
        "Describe RoPE positional embeddings.",
        "What is the mathematical formulation of SwiGLU?",
        "How does key-value caching optimize decoding?",
        "What guarantees does deterministic sampling provide?",
        "Summarize our architectural discussion.",
    ]

    for turn_idx, turn_prompt in enumerate(conversation_prompts):
        t0 = time.perf_counter()
        res = runtime.chat(
            session_id=session_id,
            prompt=turn_prompt,
            generation_config=GenerationConfig(max_new_tokens=12),
        )
        turn_ms = (time.perf_counter() - t0) * 1000.0
        chat_turn_latencies.append(turn_ms)

    chat_stats = compute_latency_stats(chat_turn_latencies)
    sess = runtime.get_or_create_session(session_id)
    print(f"  -> Completed {len(sess.turns)} turns in session")
    print(f"  -> Chat Turn Latency: Mean {chat_stats['mean_ms']:.2f} ms | Median {chat_stats['median_ms']:.2f} ms")

    # 4. Benchmark Context Compaction Overhead at Boundary
    print(f"\n[3/3] Benchmarking Context Compaction Overhead under Deep History...")
    compaction_session_id = "bench_deep_compaction"
    compaction_sess = runtime.get_or_create_session(compaction_session_id)

    # Pre-populate session with 20 turns
    for i in range(20):
        compaction_sess.add_turn(
            role="user" if i % 2 == 0 else "assistant",
            content=f"Synthetic history turn {i} containing domain knowledge about neural models.",
            token_count=25,
        )

    compaction_times_ms: List[float] = []
    for _ in range(50):
        t0 = time.perf_counter()
        compaction_sess.build_prompt_package(
            user_input="New turn requiring history compaction to fit within budget.",
            tokenizer=runtime.tokenizer,
            max_new_tokens=32,
        )
        compaction_times_ms.append((time.perf_counter() - t0) * 1000.0)

    compaction_stats = compute_latency_stats(compaction_times_ms)
    print(f"  -> Context Compaction Overhead: Mean {compaction_stats['mean_ms']:.3f} ms | Median {compaction_stats['median_ms']:.3f} ms")

    # 5. Measure Process Memory RSS & Verify Weight Immutability
    process = psutil.Process(os.getpid())
    mem_rss_mb = process.memory_info().rss / (1024 * 1024)

    post_hash = runtime.engine.compute_weight_hash()
    assert post_hash == EXPECTED_WEIGHT_HASH, f"Post-benchmark weight hash corrupted: {post_hash}"

    report = {
        "benchmark_metadata": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "step": 46,
            "architecture": "ChakrMicro v0.1",
            "model_parameters": EXPECTED_PARAM_COUNT,
            "vocab_size": runtime.model.config.vocab_size,
            "max_seq_len": runtime.model.config.max_seq_len,
            "weight_sha256": EXPECTED_WEIGHT_HASH,
            "delta_w": 0,
            "process_rss_mb": round(mem_rss_mb, 2),
        },
        "streaming_benchmarks": {
            "time_to_first_token_ms": ttft_stats,
            "inter_token_latency_ms": itl_stats,
            "throughput_tokens_per_sec": round(stream_throughput, 2),
            "tokens_per_stream": tokens_per_stream,
            "iterations": num_iterations,
        },
        "multi_turn_chat_benchmarks": {
            "turn_latency_ms": chat_stats,
            "turns_executed": len(conversation_prompts),
            "session_turns_recorded": len(sess.turns),
        },
        "context_compaction_benchmarks": {
            "compaction_overhead_ms": compaction_stats,
            "history_depth_turns": 20,
            "benchmark_iterations": 50,
        },
        "invariant_verification": {
            "weights_immutable": True,
            "pre_benchmark_sha256": EXPECTED_WEIGHT_HASH,
            "post_benchmark_sha256": post_hash,
            "delta_w_zero": True,
        }
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 72)
    print(f"BENCHMARK COMPLETE. Saved to: {OUTPUT_FILE}")
    print(f"TTFT: {ttft_stats['mean_ms']:.2f} ms | ITL: {itl_stats['mean_ms']:.2f} ms | Throughput: {stream_throughput:.1f} tok/s")
    print(f"Chat Turn Latency: {chat_stats['mean_ms']:.2f} ms | Compaction Overhead: {compaction_stats['mean_ms']:.3f} ms")
    print(f"Memory RSS: {mem_rss_mb:.1f} MB | Delta W: 0 (SHA-256 Verified)")
    print("=" * 72)


if __name__ == "__main__":
    run_benchmark()
