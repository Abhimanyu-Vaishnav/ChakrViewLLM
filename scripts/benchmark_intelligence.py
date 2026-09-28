"""
Empirical Benchmark for ChakrView Step 20: Neural Reasoning Integration & Intelligence Loop.

Measures:
1. Context construction latency (with priority packing and token budgeting)
2. Token budget enforcement overhead
3. Neural inference latency (ChakrMicro autoregressive generation on CPU)
4. Learning record creation and lifecycle validation latency
5. Feedback processing and prompt injection filtering latency
6. Governed reasoning execution latency
7. End-to-end Neural Intelligence Loop latency and overhead breakdown

CRITICAL RULE:
No numbers are fabricated. All measurements are collected from live execution on CPU.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from typing import Dict, Any, List

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.tokenizer import load_experiment_artifacts
from chakrview.capability.bridge import get_standard_capability_registry
from chakrview.capability.gate import CapabilityGate
from chakrview.state.manager import CognitiveStateManager
from chakrview.state.epistemic import EpistemicStatus
from chakrview.intelligence.contracts import (
    LearningRecord,
    LearningRecordStatus,
    NeuralInferenceRequest,
)
from chakrview.intelligence.context import (
    ContextBudget,
    IntelligenceContextBuilder,
)
from chakrview.intelligence.inference import NeuralInferenceEngine
from chakrview.intelligence.feedback import (
    FeedbackCategory,
    FeedbackCollector,
    RuntimeObservation,
    RuntimeEvaluation,
)
from chakrview.intelligence.learning import (
    LearningPipeline,
    ModelUpdateManager,
)
from chakrview.intelligence.pipeline import NeuralIntelligenceLoop


def benchmark_context_construction(builder: IntelligenceContextBuilder, tokenizer: Any, num_iterations: int = 50) -> Dict[str, Any]:
    """Measure latency of context building with various payload sizes."""
    latencies = []
    token_counts = []
    dropped_counts = []

    for i in range(num_iterations):
        t0 = time.perf_counter()
        assembled = builder.build_context(
            task_objective="Analyze system security constraints and compute hash integrity.",
            system_identity="ChakrMicro Sovereign Neural Core v0.1",
            system_constraints=[
                "Local sovereignty; do not call external APIs.",
                "Verify all mathematical and structural invariants.",
                "DATA != AUTHORITY.",
            ],
            verified_knowledge=[
                f"Fact #{k}: Node {k} is operating in sovereign mode." for k in range(5)
            ],
            memories=[
                f"Memory #{k}: User preferred concise outputs." for k in range(3)
            ],
            uncertainties=[
                "Uncertainty: External network availability is unknown."
            ],
            user_assertions=[
                f"User assertion #{k}: Requesting verification check." for k in range(5)
            ],
            tokenizer=tokenizer,
            budget=ContextBudget(max_context=512, generation_budget=128),
        )
        t_el = (time.perf_counter() - t0) * 1000.0
        latencies.append(t_el)
        token_counts.append(assembled.token_count)
        dropped_counts.append(len(assembled.items_dropped))

    latencies.sort()
    return {
        "iterations": num_iterations,
        "mean_latency_ms": round(sum(latencies) / len(latencies), 4),
        "median_latency_ms": round(latencies[len(latencies) // 2], 4),
        "min_latency_ms": round(min(latencies), 4),
        "max_latency_ms": round(max(latencies), 4),
        "p95_latency_ms": round(latencies[int(len(latencies) * 0.95)], 4),
        "mean_prompt_tokens": round(sum(token_counts) / len(token_counts), 1),
        "items_dropped_mean": round(sum(dropped_counts) / len(dropped_counts), 1),
    }


def benchmark_neural_inference(engine: NeuralInferenceEngine, num_iterations: int = 10, max_new_tokens: int = 16) -> Dict[str, Any]:
    """Measure CPU autoregressive neural inference with ChakrMicro."""
    latencies = []
    throughputs = []

    # Warmup
    engine.infer(NeuralInferenceRequest(prompt_text="Warmup ChakrView", max_new_tokens=4))

    for i in range(num_iterations):
        req = NeuralInferenceRequest(
            prompt_text=f"Task query iteration {i}: evaluate invariant state.",
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            compute_uncertainty=True,
        )
        res = engine.infer(req)
        latencies.append(res.latency_ms)
        tok_per_sec = (res.generated_tokens_count / (res.latency_ms / 1000.0)) if res.latency_ms > 0 else 0.0
        throughputs.append(tok_per_sec)

    latencies.sort()
    throughputs.sort()
    return {
        "iterations": num_iterations,
        "max_new_tokens": max_new_tokens,
        "mean_latency_ms": round(sum(latencies) / len(latencies), 2),
        "median_latency_ms": round(latencies[len(latencies) // 2], 2),
        "min_latency_ms": round(min(latencies), 2),
        "max_latency_ms": round(max(latencies), 2),
        "mean_throughput_tok_per_sec": round(sum(throughputs) / len(throughputs), 2),
        "median_throughput_tok_per_sec": round(throughputs[len(throughputs) // 2], 2),
    }


def benchmark_feedback_and_learning(collector: FeedbackCollector, pipeline: LearningPipeline, num_iterations: int = 100) -> Dict[str, Any]:
    """Measure learning record creation, injection check, validation, and pipeline storage."""
    latencies = []

    obs = RuntimeObservation(
        raw_output="Result: 120 (verified by calculator capability)",
        task_id="task_bench",
        source_type="governed_agent",
        execution_time_ms=1.2,
    )
    eval_ok = RuntimeEvaluation(
        is_passed=True,
        quality_score=0.95,
        verification_notes="AST arithmetic checked",
        evaluator_id="eval_bench",
    )

    for i in range(num_iterations):
        t0 = time.perf_counter()
        rec = collector.process_feedback(
            input_context=f"Calculate 10 * 12 (iter {i})",
            observation=obs,
            evaluation=eval_ok,
            category=FeedbackCategory.VERIFIED_REASONING,
            owner_id="bench_user",
        )
        pipeline.add_record(rec)
        t_el = (time.perf_counter() - t0) * 1000.0
        latencies.append(t_el)

    latencies.sort()
    return {
        "iterations": num_iterations,
        "mean_latency_ms": round(sum(latencies) / len(latencies), 4),
        "median_latency_ms": round(latencies[len(latencies) // 2], 4),
        "p95_latency_ms": round(latencies[int(len(latencies) * 0.95)], 4),
        "records_created": len(pipeline.get_records("bench_user")),
    }


def benchmark_end_to_end_loop(loop: NeuralIntelligenceLoop, num_iterations: int = 10) -> Dict[str, Any]:
    """Measure complete end-to-end Neural Intelligence Loop latency."""
    latencies = []
    context_tokens = []
    gen_tokens = []

    # Warmup
    loop.run("Calculate 5 + 5", max_new_tokens=8)

    for i in range(num_iterations):
        prompt = f"Calculate {10 + i} * 4"
        res = loop.run(prompt, max_new_tokens=16, compute_uncertainty=True)
        latencies.append(res.execution_time_ms)
        context_tokens.append(res.assembled_context.token_count)
        gen_tokens.append(res.neural_result.generated_tokens_count)

    latencies.sort()
    return {
        "iterations": num_iterations,
        "mean_total_latency_ms": round(sum(latencies) / len(latencies), 2),
        "median_total_latency_ms": round(latencies[len(latencies) // 2], 2),
        "min_latency_ms": round(min(latencies), 2),
        "max_latency_ms": round(max(latencies), 2),
        "mean_context_tokens": round(sum(context_tokens) / len(context_tokens), 1),
        "mean_generated_tokens": round(sum(gen_tokens) / len(gen_tokens), 1),
    }


def main():
    print("=" * 70)
    print("CHAKRVIEW STEP 20: NEURAL INTELLIGENCE BENCHMARK")
    print("=" * 70)

    # 1. Setup Model and Tokenizer
    model_cfg = ModelConfig()
    model = ChakrMicro(model_cfg)
    model.eval()

    tok_dir = Path("data/tokenizer_experiments/v4096")
    tokenizer = load_experiment_artifacts(tok_dir)

    state_mgr = CognitiveStateManager(owner_id="bench_user", session_id="bench_sess")
    cap_registry = get_standard_capability_registry()
    cap_gate = CapabilityGate(cap_registry)
    ctx_builder = IntelligenceContextBuilder()
    inference_engine = NeuralInferenceEngine(model, tokenizer)
    feedback_collector = FeedbackCollector()
    learning_pipeline = LearningPipeline()
    model_mgr = ModelUpdateManager()

    loop = NeuralIntelligenceLoop(
        model=model,
        tokenizer=tokenizer,
        state_manager=state_mgr,
        capability_gate=cap_gate,
        capability_registry=cap_registry,
        context_builder=ctx_builder,
        inference_engine=inference_engine,
        feedback_collector=feedback_collector,
        learning_pipeline=learning_pipeline,
        model_update_manager=model_mgr,
    )

    # Invariants Verification
    param_count = sum(p.numel() for p in model.parameters())
    assert param_count == 3_443_136, f"Invariant violated: {param_count}"

    print(f"Model: ChakrMicro v0.1 ({param_count:,} params, {model_cfg.vocab_size} vocab, {model_cfg.max_seq_len} ctx)")
    print("Device: CPU")
    print("Running benchmarks...")

    # Benchmark 1: Context Construction
    ctx_bench = benchmark_context_construction(ctx_builder, tokenizer, num_iterations=50)
    print(f"[1/4] Context Construction: mean={ctx_bench['mean_latency_ms']}ms, p95={ctx_bench['p95_latency_ms']}ms")

    # Benchmark 2: Neural Inference
    inf_bench = benchmark_neural_inference(inference_engine, num_iterations=10, max_new_tokens=16)
    print(f"[2/4] Neural Inference: mean={inf_bench['mean_latency_ms']}ms, throughput={inf_bench['mean_throughput_tok_per_sec']} tok/s")

    # Benchmark 3: Feedback & Learning Pipeline
    feed_bench = benchmark_feedback_and_learning(feedback_collector, learning_pipeline, num_iterations=100)
    print(f"[3/4] Feedback & Learning Record: mean={feed_bench['mean_latency_ms']}ms, records={feed_bench['records_created']}")

    # Benchmark 4: End-to-End Intelligence Loop
    e2e_bench = benchmark_end_to_end_loop(loop, num_iterations=10)
    print(f"[4/4] End-to-End Loop: mean={e2e_bench['mean_total_latency_ms']}ms, median={e2e_bench['median_total_latency_ms']}ms")

    results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "step": 20,
        "environment": {
            "device": "CPU",
            "threads": torch.get_num_threads(),
            "torch_version": torch.__version__,
        },
        "model_invariants": {
            "parameters": param_count,
            "vocab_size": model_cfg.vocab_size,
            "max_seq_len": model_cfg.max_seq_len,
            "bos_token_id": model_cfg.bos_token_id,
            "eos_token_id": model_cfg.eos_token_id,
            "pad_token_id": model_cfg.pad_token_id,
        },
        "context_construction_benchmark": ctx_bench,
        "neural_inference_benchmark": inf_bench,
        "feedback_and_learning_benchmark": feed_bench,
        "end_to_end_intelligence_loop_benchmark": e2e_bench,
    }

    out_file = Path("docs/STEP_20_BENCHMARK_RESULTS.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nBenchmark successfully written to: {out_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()
