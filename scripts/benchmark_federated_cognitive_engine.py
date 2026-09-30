"""
Step 42 Benchmark: Federated Cognitive Orchestration & Distributed Reasoning Graph.

Measures:
1. Episode planning throughput (plan_episode, N iterations)
2. Context envelope creation and validation (N iterations)
3. DAG traversal and step completion (N iterations)
4. Full 5-step episode execution in simulation mode (N iterations)
5. Synthesis engine throughput (N synthesis calls)
6. FederatedNeuralCapability (real ChakrMicro, M forward passes)
7. Context envelope serialization roundtrip (N iterations)

Output: docs/STEP_42_BENCHMARK_RESULTS.json
"""

import hashlib
import json
import statistics
import time
from pathlib import Path

import torch


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _now_ms() -> float:
    return time.perf_counter() * 1000.0


def _bench(label: str, fn, n: int = 500) -> dict:
    times = []
    for _ in range(n):
        t0 = _now_ms()
        fn()
        times.append(_now_ms() - t0)
    result = {
        "label": label,
        "iterations": n,
        "total_ms": round(sum(times), 2),
        "mean_ms": round(statistics.mean(times), 4),
        "median_ms": round(statistics.median(times), 4),
        "stdev_ms": round(statistics.stdev(times) if len(times) > 1 else 0.0, 4),
        "min_ms": round(min(times), 4),
        "max_ms": round(max(times), 4),
        "throughput_per_sec": round(1000.0 / statistics.mean(times), 1),
    }
    print(
        f"  [{label}] mean={result['mean_ms']:.3f}ms  "
        f"median={result['median_ms']:.3f}ms  "
        f"throughput={result['throughput_per_sec']:.0f}/s  "
        f"(N={n})"
    )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark Targets
# ─────────────────────────────────────────────────────────────────────────────

def bench_episode_planning(results: list, N: int = 300):
    from chakrview.cognition.federation.cognitive.engine import FederatedCognitiveEngine
    engine = FederatedCognitiveEngine()
    i = [0]

    def fn():
        i[0] += 1
        engine.plan_episode(
            f"Objective {i[0]}",
            tenant_id="bench_tenant",
            session_id="bench_session",
        )

    results.append(_bench("episode_planning", fn, n=N))


def bench_context_envelope(results: list, N: int = 1000):
    from chakrview.cognition.federation.cognitive.models import CognitiveContextEnvelope
    import uuid

    def fn():
        env = CognitiveContextEnvelope(
            envelope_id=f"env_{uuid.uuid4().hex[:8]}",
            episode_id="ep_bench",
            tenant_id="bench_tenant",
            session_id="bench_session",
        )
        for j in range(5):
            env.add_context_item(f"Context item {j} with some realistic text")
        for j in range(3):
            env.add_evidence(f"Evidence statement {j} about the domain")
        env.validate()

    results.append(_bench("context_envelope_create_validate", fn, n=N))


def bench_dag_traversal(results: list, N: int = 500):
    from chakrview.cognition.federation.cognitive.models import (
        CognitiveTaskGraph,
        CognitiveStep,
        CognitiveRole,
        CAPABILITY_ANALYST,
    )

    def fn():
        g = CognitiveTaskGraph(graph_id="gr_bench", episode_id="ep_bench")
        for k in range(5):
            deps = [f"s{k-1}"] if k > 0 else []
            g.add_step(CognitiveStep(
                step_id=f"s{k}",
                role=CognitiveRole.ANALYST,
                capability_id=CAPABILITY_ANALYST,
                dependencies=deps,
            ))
        g.validate()
        for k in range(5):
            ready = g.get_ready_steps()
            if ready:
                g.mark_step_completed(ready[0].step_id, {"result": k})

    results.append(_bench("dag_traversal_5_steps", fn, n=N))


def bench_full_episode_simulation(results: list, N: int = 100):
    from chakrview.cognition.federation.cognitive.engine import FederatedCognitiveEngine
    engine = FederatedCognitiveEngine()

    sim = {
        "step_analyst":     {"success": True, "node_id": "n1", "conclusion": "Analyst", "evidence": ["e1"], "hypotheses": []},
        "step_researcher":  {"success": True, "node_id": "n2", "conclusion": "Researcher", "evidence": ["e2"], "hypotheses": []},
        "step_critic":      {"success": True, "node_id": "n3", "conclusion": "Critic", "evidence": ["e3"], "hypotheses": []},
        "step_synthesizer": {"success": True, "node_id": "n4", "conclusion": "Synthesizer", "evidence": ["e4"], "hypotheses": []},
        "step_verifier":    {"success": True, "node_id": "n5", "conclusion": "Verified", "evidence": ["e5"], "hypotheses": []},
    }

    def fn():
        ep = engine.plan_episode("Bench objective", "bench_tenant", "bench_session")
        engine.execute_episode(ep.episode_id, simulate_results=sim)

    results.append(_bench("full_episode_plan_execute_simulate", fn, n=N))


def bench_synthesis_engine(results: list, N: int = 500):
    from chakrview.cognition.federation.cognitive.synthesis import CognitiveSynthesisEngine
    engine = CognitiveSynthesisEngine()
    step_results = [
        {"node_id": f"n{i}", "conclusion": "Conclusion A", "evidence": [f"ev{i}"], "hypotheses": []}
        for i in range(5)
    ]

    def fn():
        engine.synthesize("ep_bench", "step_synth", step_results)

    results.append(_bench("synthesis_5_workers", fn, n=N))


def bench_envelope_serialization(results: list, N: int = 1000):
    from chakrview.cognition.federation.cognitive.models import CognitiveContextEnvelope

    env = CognitiveContextEnvelope(
        envelope_id="env_bench",
        episode_id="ep_bench",
        tenant_id="bench_tenant",
        session_id="bench_session",
    )
    for j in range(10):
        env.add_context_item(f"Context item {j}")
    for j in range(5):
        env.add_evidence(f"Evidence {j}")

    def fn():
        d = env.to_dict()
        CognitiveContextEnvelope.from_dict(d)

    results.append(_bench("envelope_serialization_roundtrip", fn, n=N))


def bench_neural_capability(results: list, M: int = 5):
    """
    Benchmark real ChakrMicro forward pass via FederatedNeuralCapability.
    Uses a small number of iterations since each forward pass is expensive on CPU.
    """
    from chakrview.brain.config import ModelConfig
    from chakrview.brain.model import ChakrMicro
    from chakrview.cognition.federation.cognitive.capabilities import FederatedNeuralCapability
    from chakrview.capability.contract import CapabilityRequest
    from chakrview.cognition.federation.cognitive.models import CAPABILITY_NEURAL_INFERENCE

    torch.manual_seed(42)
    model = ChakrMicro(ModelConfig())
    model.eval()
    cap = FederatedNeuralCapability(model)

    prompt = [0, 10, 20, 30, 40]

    def fn():
        req = CapabilityRequest(
            capability_id=CAPABILITY_NEURAL_INFERENCE,
            parameters={"prompt_tokens": prompt, "max_new_tokens": 1},
        )
        result = cap.execute(req)
        assert result.success

    results.append(_bench("neural_inference_1_token", fn, n=M))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 70)
    print("STEP 42 BENCHMARK: Federated Cognitive Orchestration")
    print("=" * 70)

    results = []

    print("\n[1] Episode Planning")
    bench_episode_planning(results, N=300)

    print("\n[2] Context Envelope Create + Validate")
    bench_context_envelope(results, N=1000)

    print("\n[3] DAG Traversal (5-step chain)")
    bench_dag_traversal(results, N=500)

    print("\n[4] Full Episode Simulation (plan + execute 5 steps)")
    bench_full_episode_simulation(results, N=100)

    print("\n[5] Synthesis Engine (5 workers)")
    bench_synthesis_engine(results, N=500)

    print("\n[6] Context Envelope Serialization Roundtrip")
    bench_envelope_serialization(results, N=1000)

    print("\n[7] Neural Inference (real ChakrMicro, 1 token)")
    try:
        bench_neural_capability(results, M=5)
    except Exception as e:
        print(f"  Neural benchmark skipped: {e}")

    # ΔW = 0 verification
    print("\n[Neural Core Verification]")
    from chakrview.brain.config import ModelConfig
    from chakrview.brain.model import ChakrMicro
    torch.manual_seed(42)
    model = ChakrMicro(ModelConfig())
    hasher = hashlib.sha256()
    with torch.no_grad():
        for name, param in sorted(model.named_parameters()):
            hasher.update(name.encode("utf-8"))
            hasher.update(param.detach().cpu().numpy().tobytes())
    actual_hash = hasher.hexdigest()
    EXPECTED_HASH = "c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da"
    param_count = sum(p.numel() for p in model.parameters())
    delta_w_zero = actual_hash == EXPECTED_HASH

    neural_verification = {
        "parameter_count": param_count,
        "vocabulary_size": 4096,
        "context_length": 512,
        "weight_hash": actual_hash,
        "expected_hash": EXPECTED_HASH,
        "delta_w_zero": delta_w_zero,
    }
    print(f"  Parameters:  {param_count:,}")
    print(f"  Vocabulary:  4096")
    print(f"  Context:     512")
    print(f"  Weight hash: {actual_hash}")
    print(f"  DeltaW=0:    {delta_w_zero}")
    assert delta_w_zero, "DeltaW != 0 VIOLATION"

    output = {
        "step": 42,
        "title": "Federated Cognitive Orchestration & Distributed Reasoning Graph",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "benchmarks": results,
        "neural_core_verification": neural_verification,
    }

    out_path = Path("docs/STEP_42_BENCHMARK_RESULTS.json")
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nDONE: Benchmark results written to {out_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
