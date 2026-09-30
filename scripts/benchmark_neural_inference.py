"""
Step 44 Benchmark: End-to-End Neural Inference Pipeline.

Measures CPU performance for:
1. Tokenizer encode latency & throughput
2. Tokenizer decode latency & throughput
3. Context assembly & secret scanning overhead
4. Single-token forward pass (ChakrMicro)
5. 8-token autoregressive generation
6. 32-token autoregressive generation
7. 64-token autoregressive generation
8. End-to-end cognitive inference (Envelope -> Tokenize -> Model -> Detokenize)

Output: docs/STEP_44_BENCHMARK_RESULTS.json
"""

import json
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Callable, Dict, List
import torch

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.cognition.federation.cognitive.models import CognitiveContextEnvelope
from chakrview.runtime.inference import GenerationConfig
from chakrview.runtime.pipeline import (
    InferenceContextBuilder,
    InferenceEngine,
    InferenceRequest,
)
from chakrview.runtime.sampling import SamplingConfig
from chakrview.tokenizer.serialization import load_tokenizer_artifacts


def run_microbenchmark(
    func: Callable[[], Any],
    iterations: int,
    warmup: int = 5,
    label: str = "",
) -> Dict[str, Any]:
    """Execute warmup, measure execution timings, and compute statistics."""
    for _ in range(warmup):
        func()

    timings_ms: List[float] = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        func()
        t1 = time.perf_counter()
        timings_ms.append((t1 - t0) * 1000.0)

    total_ms = sum(timings_ms)
    mean_ms = statistics.mean(timings_ms)
    median_ms = statistics.median(timings_ms)
    stdev_ms = statistics.stdev(timings_ms) if len(timings_ms) > 1 else 0.0
    min_ms = min(timings_ms)
    max_ms = max(timings_ms)
    throughput = (iterations / (total_ms / 1000.0)) if total_ms > 0 else 0.0

    print(
        f"  [{label:<35}] "
        f"mean={mean_ms:8.4f}ms  "
        f"median={median_ms:8.4f}ms  "
        f"throughput={throughput:10.1f}/s  "
        f"(N={iterations})"
    )

    return {
        "label": label,
        "iterations": iterations,
        "total_ms": round(total_ms, 2),
        "mean_ms": round(mean_ms, 4),
        "median_ms": round(median_ms, 4),
        "stdev_ms": round(stdev_ms, 4),
        "min_ms": round(min_ms, 4),
        "max_ms": round(max_ms, 4),
        "throughput_per_sec": round(throughput, 1),
    }


def main() -> None:
    print("=" * 80)
    print("ChakrView Step 44: End-to-End Neural Inference Pipeline Benchmark")
    print("=" * 80)

    root_dir = Path(__file__).resolve().parent.parent
    tok_dir = root_dir / "data" / "experiments" / "vocab_4096"
    tokenizer, _ = load_tokenizer_artifacts(tok_dir)

    torch.manual_seed(42)
    model = ChakrMicro(ModelConfig())
    model.eval()

    engine = InferenceEngine(model=model, tokenizer=tokenizer)

    benchmarks: Dict[str, Any] = {}
    sample_text = (
        "ChakrView is an indigenous distributed AI operating system. "
        "The neural core ChakrMicro has 3,443,136 parameters across 6 transformer blocks."
    )
    tokens_64 = engine.encode(sample_text)[:64]
    if len(tokens_64) < 32:
        tokens_64 = tokens_64 * 3
    tokens_64 = tokens_64[:64]

    # 1. Tokenizer Encode
    benchmarks["tokenizer_encode"] = run_microbenchmark(
        func=lambda: engine.encode(sample_text, add_bos=True),
        iterations=500,
        label="tokenizer_encode",
    )

    # 2. Tokenizer Decode
    benchmarks["tokenizer_decode_64_tokens"] = run_microbenchmark(
        func=lambda: engine.decode(tokens_64, skip_special_tokens=True),
        iterations=500,
        label="tokenizer_decode_64_tokens",
    )

    # 3. Context Assembly & Secret Scan
    env = CognitiveContextEnvelope(
        envelope_id="env_bench_001",
        episode_id="ep_bench_001",
        tenant_id="tenant_bench",
        session_id="session_bench",
    )
    env.add_context_item("Verified proposition: ChakrMicro has 6 transformer blocks.")
    env.add_evidence("External documentation snippet: Maximum sequence horizon is 512 tokens.")
    req_context = InferenceRequest(
        prompt="Analyze architectural limits",
        context_envelope=env,
        tenant_id="tenant_bench",
    )

    benchmarks["context_assembly_and_scan"] = run_microbenchmark(
        func=lambda: InferenceContextBuilder.build_context(req_context, tokenizer),
        iterations=500,
        label="context_assembly_and_scan",
    )

    # 4. Single Forward Pass (Prefill 32 tokens)
    tokens_32 = tokens_64[:32]
    benchmarks["single_forward_pass_32_tokens"] = run_microbenchmark(
        func=lambda: engine.forward(tokens_32),
        iterations=50,
        warmup=2,
        label="single_forward_pass_32_tokens",
    )

    # 5. Greedy Generation: 8 Tokens
    cfg_8 = GenerationConfig(max_new_tokens=8, sampling=SamplingConfig(temperature=0.0))
    req_8 = InferenceRequest(prompt="Scientific exploration", generation_config=cfg_8)
    benchmarks["greedy_generation_8_tokens"] = run_microbenchmark(
        func=lambda: engine.execute(req_8),
        iterations=30,
        warmup=2,
        label="greedy_generation_8_tokens",
    )

    # 6. Greedy Generation: 32 Tokens
    cfg_32 = GenerationConfig(max_new_tokens=32, sampling=SamplingConfig(temperature=0.0))
    req_32 = InferenceRequest(prompt="Distributed cognition requires", generation_config=cfg_32)
    benchmarks["greedy_generation_32_tokens"] = run_microbenchmark(
        func=lambda: engine.execute(req_32),
        iterations=15,
        warmup=1,
        label="greedy_generation_32_tokens",
    )

    # 7. Greedy Generation: 64 Tokens
    cfg_64 = GenerationConfig(max_new_tokens=64, sampling=SamplingConfig(temperature=0.0))
    req_64 = InferenceRequest(prompt="Transformer neural networks", generation_config=cfg_64)
    benchmarks["greedy_generation_64_tokens"] = run_microbenchmark(
        func=lambda: engine.execute(req_64),
        iterations=10,
        warmup=1,
        label="greedy_generation_64_tokens",
    )

    # 8. End-to-End Cognitive Inference (Envelope -> Tokenize -> Model -> Detokenize)
    req_e2e = InferenceRequest(
        prompt="Synthesize cognitive architecture report",
        context_envelope=env,
        tenant_id="tenant_bench",
        generation_config=GenerationConfig(max_new_tokens=16, sampling=SamplingConfig(temperature=0.0)),
    )
    benchmarks["end_to_end_cognitive_inference"] = run_microbenchmark(
        func=lambda: engine.execute(req_e2e),
        iterations=20,
        warmup=2,
        label="end_to_end_cognitive_inference",
    )

    out_data = {
        "benchmark_suite": "Step 44: End-to-End Neural Inference Pipeline",
        "timestamp": time.time(),
        "environment": {
            "mode": "in-process loopback CPU execution",
            "cpu_only": True,
            "gpu_required": False,
            "neural_model": "ChakrMicro v0.1 (3,443,136 parameters, ΔW = 0)",
            "weight_hash": engine.expected_weight_hash,
            "max_context": engine.max_context,
            "vocab_size": engine.tokenizer.vocab_size,
        },
        "benchmarks": benchmarks,
    }

    out_file = root_dir / "docs" / "STEP_44_BENCHMARK_RESULTS.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2)

    print("=" * 80)
    print(f"Results written to: {out_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()
