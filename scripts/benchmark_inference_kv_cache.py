"""
Benchmark: Full-Forward (O(N^2)) vs Persistent KV-Cache (O(N)) Incremental Inference (Step 10).

Empirically measures:
- Generation throughput (tokens/second)
- Per-token decoding latency (ms)
- First-token latency (ms)
- Cache memory consumption (bytes)
- Numerical equivalence (maximum logit delta between cached and full-forward)
across various prompt lengths and generation horizons.
"""

import gc
import json
from pathlib import Path
import sys
import time
from typing import Dict, List, Tuple, Any
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.brain.cache import KVCache
from chakrview.training.checkpoint import CheckpointManager



def generate_full_forward(
    model: ChakrMicro,
    prompt_tensor: torch.Tensor,
    num_tokens: int,
) -> Tuple[List[int], List[torch.Tensor], float]:
    """Generate tokens by recomputing the full sequence on every step (O(N^2))."""
    tokens = prompt_tensor.clone()
    logits_history: List[torch.Tensor] = []

    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_tokens):
            out = model(tokens)
            next_logits = out[:, -1, :]
            logits_history.append(next_logits.clone())
            next_id = torch.argmax(next_logits, dim=-1, keepdim=True)
            tokens = torch.cat([tokens, next_id], dim=1)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    return tokens[0, prompt_tensor.shape[1]:].tolist(), logits_history, latency_ms


def generate_kv_cache(
    model: ChakrMicro,
    prompt_tensor: torch.Tensor,
    num_tokens: int,
) -> Tuple[List[int], List[torch.Tensor], float, float, int]:
    """Generate tokens using persistent KV-cache incremental decoding (O(N))."""
    logits_history: List[torch.Tensor] = []
    generated_ids: List[int] = []

    t0 = time.perf_counter()
    with torch.no_grad():
        # Prefill
        prefill_out, cache = model.prefill(prompt_tensor)
        next_logits = prefill_out[:, -1, :]
        logits_history.append(next_logits.clone())
        next_id = torch.argmax(next_logits, dim=-1, keepdim=True)
        generated_ids.append(next_id.item())
        t_prefill = time.perf_counter()

        # Incremental decoding
        for _ in range(num_tokens - 1):
            out = model.decode_next(next_id, kv_cache=cache)
            next_logits = out[:, -1, :]
            logits_history.append(next_logits.clone())
            next_id = torch.argmax(next_logits, dim=-1, keepdim=True)
            generated_ids.append(next_id.item())

    t_end = time.perf_counter()
    total_latency_ms = (t_end - t0) * 1000.0
    prefill_latency_ms = (t_prefill - t0) * 1000.0
    cache_bytes = cache.total_memory_bytes

    return generated_ids, logits_history, total_latency_ms, prefill_latency_ms, cache_bytes


def run_benchmark():
    print("=" * 80)
    print("CHAKRVIEW STEP 10: INFERENCE ENGINE & KV-CACHE BENCHMARK")
    print("=" * 80)

    torch.set_num_threads(4)
    model = ChakrMicro(ModelConfig())

    # Load Step 8 trained weights if present, else test initialized
    ckpt_path = Path("checkpoints/stage_c_full_epoch/checkpoint_0006478.pt")
    if ckpt_path.is_file():
        payload = CheckpointManager.load(ckpt_path)
        model.load_state_dict(payload["model_state_dict"])
        print(f"Loaded trained Step 8 checkpoint: {ckpt_path} (step {payload['step']})")
    else:
        print("Using initialized ChakrMicro v0.1 weights")

    model.eval()

    # Benchmark configurations: (prompt_len, gen_len)
    scenarios = [
        (16, 16),
        (32, 32),
        (64, 64),
        (128, 64),
        (256, 64),
        (256, 128),
    ]

    results = []

    print("\nEvaluating benchmark scenarios...\n")
    print(f"{'Prompt':<8}{'Gen':<6}{'Full Fwd (ms)':<16}{'KV Cache (ms)':<16}{'Speedup':<10}{'KV Tok/s':<12}{'Cache RAM':<12}{'Max Logit Delta':<16}")
    print("-" * 96)


    for p_len, g_len in scenarios:
        # Create deterministic pseudo-prompt
        torch.manual_seed(42)
        prompt = torch.randint(3, 4095, (1, p_len), dtype=torch.long)

        # Warmup
        _ = model(prompt)

        # Run Method A (Full Forward)
        gc.collect()
        tokens_a, logits_a, lat_a = generate_full_forward(model, prompt, g_len)

        # Run Method B (KV Cache)
        gc.collect()
        tokens_b, logits_b, lat_b, prefill_lat, cache_bytes = generate_kv_cache(model, prompt, g_len)

        # Measure numerical discrepancy across all emitted token positions
        max_delta = 0.0
        for la, lb in zip(logits_a, logits_b):
            delta = torch.max(torch.abs(la - lb)).item()
            if delta > max_delta:
                max_delta = delta

        # Exact token match verification
        tokens_match = (tokens_a == tokens_b)
        speedup = lat_a / lat_b if lat_b > 0 else 1.0
        kv_tok_sec = (g_len / (lat_b / 1000.0)) if lat_b > 0 else 0.0
        cache_kb = cache_bytes / 1024.0

        print(
            f"{p_len:<8}{g_len:<6}{lat_a:<16.2f}{lat_b:<16.2f}{speedup:<8.2f}x  {kv_tok_sec:<12.1f}{cache_kb:<8.1f} KB  {max_delta:<14.2e}"
        )


        results.append({
            "prompt_length": p_len,
            "generation_length": g_len,
            "full_forward_latency_ms": lat_a,
            "kv_cache_latency_ms": lat_b,
            "speedup": speedup,
            "kv_cache_throughput_tok_sec": kv_tok_sec,
            "cache_memory_bytes": cache_bytes,
            "max_logit_discrepancy": max_delta,
            "tokens_identical": tokens_match,
        })

    # Summary
    avg_speedup = sum(r["speedup"] for r in results) / len(results)
    max_discrepancy_overall = max(r["max_logit_discrepancy"] for r in results)
    all_matched = all(r["tokens_identical"] for r in results)

    print("-" * 94)
    print(f"Average KV-Cache Speedup: {avg_speedup:.2f}x")
    print(f"Overall Maximum Numerical Discrepancy: {max_discrepancy_overall:.2e}")
    print(f"All Generated Tokens Match Exactly: {all_matched}")
    print("=" * 80)

    # Save benchmark artifact
    out_file = Path("docs/STEP_10_BENCHMARK_RESULTS.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "device": "cpu",
                "threads": 4,
                "model_parameters": 3443136,
                "scenarios": results,
                "summary": {
                    "average_speedup": avg_speedup,
                    "max_logit_discrepancy": max_discrepancy_overall,
                    "all_tokens_identical": all_matched,
                },
            },
            f,
            indent=2,
        )
    print(f"\nSaved benchmark results to {out_file}")


if __name__ == "__main__":
    run_benchmark()
