"""
Empirical Benchmark for ChakrView Step 21: Neural Thinking & Deliberation Foundation.

Measures:
1. Workspace creation and initialization latency
2. Attention / focus selection latency
3. Candidate generation latency (ChakrMicro autoregressive inference on CPU)
4. Critique engine evaluation latency
5. Revision engine planning latency
6. Stopping policy evaluation latency
7. Complete end-to-end deliberation latency
8. Average thinking cycles and revisions
9. Workspace and memory growth (bytes and thought step accounting)

CRITICAL RULE:
No numbers are fabricated. All measurements are collected from live execution on CPU.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import sys
import time
from typing import Dict, Any, List

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.tokenizer import load_experiment_artifacts, BOS_ID, EOS_ID, PAD_ID
from chakrview.tokenizer.tokenizer import BPETokenizer
from chakrview.capability.bridge import get_standard_capability_registry
from chakrview.capability.gate import CapabilityGate
from chakrview.state.manager import CognitiveStateManager

from chakrview.thinking.thought import ThoughtStep, ThoughtPurpose
from chakrview.thinking.policy import (
    ThinkingPolicy,
    get_standard_policy,
    get_strict_policy,
    get_fast_policy,
)
from chakrview.thinking.workspace import ThinkingWorkspace
from chakrview.thinking.attention import ThinkingAttention, FocusType
from chakrview.thinking.critique import CritiqueEngine, CritiqueVerdict, CritiqueResult
from chakrview.thinking.revision import RevisionEngine, RevisionPlan
from chakrview.thinking.stopping import ThinkingStoppingPolicy, StoppingCondition
from chakrview.thinking.trace import ThinkingTrace
from chakrview.thinking.deliberation import DeliberationEngine, DeliberationOutcome


def benchmark_workspace_lifecycle(num_iterations: int = 100) -> Dict[str, Any]:
    """Measure latency of ThinkingWorkspace creation, state updates, and serialization."""
    policy = get_standard_policy()
    creation_times = []
    step_times = []
    serialization_times = []
    serialized_sizes = []

    for i in range(num_iterations):
        t0 = time.perf_counter()
        ws = ThinkingWorkspace(
            task_id=f"bench_task_{i}",
            owner_id="bench_user",
            session_id="bench_session",
            objective="Evaluate thermodynamic efficiency in high-pressure steam turbines",
            policy=policy,
        )
        creation_times.append((time.perf_counter() - t0) * 1000.0)

        # Measure thought addition
        t1 = time.perf_counter()
        for s in range(5):
            ws.add_thought_step(
                purpose=ThoughtPurpose.HYPOTHESIZE,
                content=f"Deliberation step hypothesis #{s + 1} with supporting parameters.",
            )
        step_times.append(((time.perf_counter() - t1) * 1000.0) / 5.0)

        # Measure serialization
        t2 = time.perf_counter()
        data = ws.to_dict()
        s_json = json.dumps(data)
        serialization_times.append((time.perf_counter() - t2) * 1000.0)
        serialized_sizes.append(len(s_json.encode("utf-8")))

    return {
        "iterations": num_iterations,
        "mean_creation_latency_ms": round(statistics.mean(creation_times), 4),
        "median_creation_latency_ms": round(statistics.median(creation_times), 4),
        "mean_thought_step_addition_latency_ms": round(statistics.mean(step_times), 4),
        "mean_serialization_latency_ms": round(statistics.mean(serialization_times), 4),
        "mean_serialized_size_bytes": round(statistics.mean(serialized_sizes), 1),
        "max_serialized_size_bytes": max(serialized_sizes),
    }


def benchmark_attention_mechanism(num_iterations: int = 100) -> Dict[str, Any]:
    """Measure latency of ThinkingAttention focus selection."""
    attention = ThinkingAttention()
    ws = ThinkingWorkspace(
        objective="Investigate atmospheric composition and pressure variants",
    )
    ws.add_evidence("Evidence line A: nitrogen dominant.", source="sensor")
    ws.add_evidence("Evidence line B: carbon dioxide traces.", source="sensor")
    ws.record_weakness("Insufficient resolution on trace gases.")
    ws.unresolved_questions.append("What is the noble gas percentage?")

    latencies = []
    for _ in range(num_iterations):
        t0 = time.perf_counter()
        focus = attention.select_focus(ws)
        latencies.append((time.perf_counter() - t0) * 1000.0)

    return {
        "iterations": num_iterations,
        "mean_latency_ms": round(statistics.mean(latencies), 4),
        "median_latency_ms": round(statistics.median(latencies), 4),
        "p95_latency_ms": round(statistics.quantiles(latencies, n=20)[18], 4),
        "sample_focus_type": focus.focus_type.value,
        "is_heuristic": focus.is_heuristic,
    }


def benchmark_critique_engine(num_iterations: int = 100) -> Dict[str, Any]:
    """Measure latency of CritiqueEngine evaluations."""
    engine = CritiqueEngine()
    ws = ThinkingWorkspace(
        objective="Compute trajectory and orbit insertion vectors",
    )
    ws.add_evidence("Standard orbit altitude is 400 km.", source="orbital_mechanics")
    ws.record_hypothesis("Candidate burns at periapsis.", confidence=0.85)

    candidate_text = "Calculated burn duration is 120s with nominal insertion altitude 400 km."

    latencies = []
    for _ in range(num_iterations):
        t0 = time.perf_counter()
        critique = engine.critique(candidate_text, ws)
        latencies.append((time.perf_counter() - t0) * 1000.0)

    return {
        "iterations": num_iterations,
        "mean_latency_ms": round(statistics.mean(latencies), 4),
        "median_latency_ms": round(statistics.median(latencies), 4),
        "p95_latency_ms": round(statistics.quantiles(latencies, n=20)[18], 4),
        "sample_verdict": critique.verdict.value,
        "sample_score": round(critique.score, 4),
    }


def benchmark_revision_engine(num_iterations: int = 100) -> Dict[str, Any]:
    """Measure latency of RevisionEngine formulating structured revision directives."""
    rev_engine = RevisionEngine()
    critique = CritiqueResult(
        verdict=CritiqueVerdict.WEAK,
        score=0.55,
        findings=["Calculated burn duration exceeds fuel reserve envelope."],
        recommended_action="Recalculate with throttled thrust profile.",
        contradictions_detected=["Fuel mass constraint violated."],
    )

    latencies = []
    for i in range(num_iterations):
        # Create fresh workspace so revision budget is respected
        ws = ThinkingWorkspace(
            objective="Calculate orbit insertion",
            policy=ThinkingPolicy(max_revision_cycles=num_iterations + 10),
        )
        t0 = time.perf_counter()
        plan = rev_engine.plan_revision(ws, critique)
        latencies.append((time.perf_counter() - t0) * 1000.0)

    return {
        "iterations": num_iterations,
        "mean_latency_ms": round(statistics.mean(latencies), 4),
        "median_latency_ms": round(statistics.median(latencies), 4),
        "p95_latency_ms": round(statistics.quantiles(latencies, n=20)[18], 4),
        "focus_areas_count": len(plan.focus_areas),
    }


def benchmark_stopping_policy(num_iterations: int = 100) -> Dict[str, Any]:
    """Measure latency of ThinkingStoppingPolicy decisions."""
    stopping_policy = ThinkingStoppingPolicy()
    ws = ThinkingWorkspace(objective="Verify structural load integrity")
    ws.set_conclusion("Final Answer: Structural integrity verified under maximum aerodynamic pressure.")
    critique = CritiqueResult(verdict=CritiqueVerdict.PASS, score=0.95)

    latencies = []
    for _ in range(num_iterations):
        t0 = time.perf_counter()
        should_stop, condition, reason = stopping_policy.evaluate_stopping(
            workspace=ws,
            latest_critique=critique,
            verification_passed=True,
        )
        latencies.append((time.perf_counter() - t0) * 1000.0)

    return {
        "iterations": num_iterations,
        "mean_latency_ms": round(statistics.mean(latencies), 4),
        "median_latency_ms": round(statistics.median(latencies), 4),
        "p95_latency_ms": round(statistics.quantiles(latencies, n=20)[18], 4),
        "sample_condition": condition.value,
        "sample_should_stop": should_stop,
    }


def benchmark_end_to_end_deliberation(
    engine: DeliberationEngine,
    num_iterations: int = 10,
) -> Dict[str, Any]:
    """
    Measure end-to-end DeliberationEngine execution on representative objectives.
    """
    objectives = [
        ("Direct Mathematical Query", "Calculate 15 * 8", ThinkingPolicy(max_thought_steps=6, max_revision_cycles=1)),
        ("Multi-Step Analytical Query", "Analyze aerodynamic boundary layer separation", ThinkingPolicy(max_thought_steps=8, max_revision_cycles=2)),
        ("Constrained Decision Query", "Evaluate power distribution options for auxiliary subsystem", ThinkingPolicy(max_thought_steps=8, max_revision_cycles=2)),
    ]

    scenario_metrics = []
    all_latencies = []
    total_cycles_list = []
    total_revisions_list = []
    workspace_bytes_list = []

    for name, obj, pol in objectives:
        latencies = []
        cycles = []
        revisions = []
        for _ in range(num_iterations):
            t0 = time.perf_counter()
            outcome = engine.deliberate(
                objective=obj,
                owner_id="bench_user",
                session_id="bench_sess",
                policy=pol,
                max_new_tokens=16,
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            latencies.append(elapsed_ms)
            cycles.append(outcome.workspace.step_count)
            revisions.append(outcome.workspace.revision_count)
            ws_size = len(json.dumps(outcome.workspace.to_dict()).encode("utf-8"))
            workspace_bytes_list.append(ws_size)

        all_latencies.extend(latencies)
        total_cycles_list.extend(cycles)
        total_revisions_list.extend(revisions)

        scenario_metrics.append({
            "scenario": name,
            "objective": obj,
            "mean_latency_ms": round(statistics.mean(latencies), 2),
            "median_latency_ms": round(statistics.median(latencies), 2),
            "avg_steps": round(statistics.mean(cycles), 2),
            "avg_revisions": round(statistics.mean(revisions), 2),
            "stopping_condition": outcome.stopping_condition,
            "success": outcome.success,
        })

    return {
        "iterations_per_scenario": num_iterations,
        "scenarios_evaluated": len(objectives),
        "overall_mean_latency_ms": round(statistics.mean(all_latencies), 2),
        "overall_median_latency_ms": round(statistics.median(all_latencies), 2),
        "overall_p95_latency_ms": round(statistics.quantiles(all_latencies, n=20)[18], 2),
        "overall_min_latency_ms": round(min(all_latencies), 2),
        "overall_max_latency_ms": round(max(all_latencies), 2),
        "average_thinking_steps": round(statistics.mean(total_cycles_list), 2),
        "average_revisions": round(statistics.mean(total_revisions_list), 2),
        "average_workspace_bytes": round(statistics.mean(workspace_bytes_list), 1),
        "max_workspace_bytes": max(workspace_bytes_list),
        "scenarios": scenario_metrics,
        "weights_modified": False,
    }


def main():
    print("=" * 70)
    print("CHAKRVIEW STEP 21: NEURAL THINKING & DELIBERATION BENCHMARK")
    print("=" * 70)
    print(f"Device: CPU | PyTorch: {torch.__version__} | Threads: {torch.get_num_threads()}")

    # Initialize frozen model
    torch.manual_seed(42)
    model_cfg = ModelConfig()
    model = ChakrMicro(model_cfg)
    model.eval()

    param_count = sum(p.numel() for p in model.parameters())
    print(f"ChakrMicro parameters: {param_count:,} (Expected: 3,443,136)")
    assert param_count == 3443136, f"Invariant violation: param count {param_count} != 3443136"

    # Tokenizer
    tok_dir = Path(__file__).resolve().parent.parent / "data" / "tokenizer_experiments" / "v4096"
    if tok_dir.is_dir():
        tokenizer = load_experiment_artifacts(tok_dir)
    else:
        tokenizer = BPETokenizer()

    # Capabilities and Gate
    cap_registry = get_standard_capability_registry()
    cap_gate = CapabilityGate(cap_registry)

    # Deliberation Engine
    delib_engine = DeliberationEngine(
        model=model,
        tokenizer=tokenizer,
        capability_registry=cap_registry,
        capability_gate=cap_gate,
    )

    # Benchmark 1: Workspace Lifecycle
    ws_bench = benchmark_workspace_lifecycle(num_iterations=100)
    print(f"[1/6] Workspace Creation & Lifecycle: creation={ws_bench['mean_creation_latency_ms']}ms, step_add={ws_bench['mean_thought_step_addition_latency_ms']}ms, size={ws_bench['mean_serialized_size_bytes']}B")

    # Benchmark 2: Attention Selection
    att_bench = benchmark_attention_mechanism(num_iterations=100)
    print(f"[2/6] Attention Selection: mean={att_bench['mean_latency_ms']}ms, focus={att_bench['sample_focus_type']}")

    # Benchmark 3: Critique Engine
    crit_bench = benchmark_critique_engine(num_iterations=100)
    print(f"[3/6] Critique Evaluation: mean={crit_bench['mean_latency_ms']}ms, verdict={crit_bench['sample_verdict']}, score={crit_bench['sample_score']}")

    # Benchmark 4: Revision Planning
    rev_bench = benchmark_revision_engine(num_iterations=100)
    print(f"[4/6] Revision Planning: mean={rev_bench['mean_latency_ms']}ms, focus_areas={rev_bench['focus_areas_count']}")

    # Benchmark 5: Stopping Policy
    stop_bench = benchmark_stopping_policy(num_iterations=100)
    print(f"[5/6] Stopping Evaluation: mean={stop_bench['mean_latency_ms']}ms, condition={stop_bench['sample_condition']}")

    # Benchmark 6: End-to-End Deliberation Loop
    print("\nRunning live end-to-end deliberation scenarios on CPU...")
    e2e_bench = benchmark_end_to_end_deliberation(delib_engine, num_iterations=5)
    print(f"[6/6] End-to-End Deliberation: mean={e2e_bench['overall_mean_latency_ms']}ms, p95={e2e_bench['overall_p95_latency_ms']}ms, avg_steps={e2e_bench['average_thinking_steps']}")

    # Assemble comprehensive results
    results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "step": 21,
        "benchmark_title": "Neural Thinking & Deliberation Foundation Benchmark",
        "environment": {
            "device": "CPU",
            "threads": torch.get_num_threads(),
            "torch_version": torch.__version__,
        },
        "model_invariants": {
            "parameters": param_count,
            "vocab_size": model_cfg.vocab_size,
            "max_seq_len": model_cfg.max_seq_len,
            "bos_token_id": BOS_ID,
            "eos_token_id": EOS_ID,
            "pad_token_id": PAD_ID,
            "weights_modified": False,
        },
        "workspace_lifecycle_benchmark": ws_bench,
        "attention_benchmark": att_bench,
        "critique_benchmark": crit_bench,
        "revision_benchmark": rev_bench,
        "stopping_policy_benchmark": stop_bench,
        "end_to_end_deliberation_benchmark": e2e_bench,
    }

    out_file = Path("docs/STEP_21_BENCHMARK_RESULTS.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nBenchmark results successfully written to: {out_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()
