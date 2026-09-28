"""
Empirical Benchmark for ChakrView Step 19: Governed Cognitive Reasoning Foundation.

Measures:
1. Task creation and lifecycle transitions
2. Task decomposition into subproblems
3. Evidence evaluation and store ingestion
4. Hypothesis evaluation
5. Structured inference construction
6. Contradiction detection and resolution evaluation
7. Decision candidate scoring and selection
8. Verification loop evaluation
9. Full governed reasoning loop end-to-end
10. Scaling with evidence volume (N = 10, 50, 100)
11. Scaling with reasoning depth (D = 1, 2, 4)

Outputs results to docs/STEP_19_BENCHMARK_RESULTS.json.
"""

import json
import os
import sys
import time
from typing import Dict, Any, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from chakrview.reasoning.task import ReasoningTask, ReasoningPhase, ReasoningTaskType
from chakrview.reasoning.decomposition import ProblemDecomposer, Subproblem
from chakrview.reasoning.evidence import EvidenceItem, EvidenceStore, EvidenceType
from chakrview.reasoning.hypothesis import Hypothesis, HypothesisEngine
from chakrview.reasoning.inference import InferenceEngine
from chakrview.reasoning.contradiction import ContradictionDetector
from chakrview.reasoning.decision import DecisionEngine, DecisionCandidate
from chakrview.reasoning.verification import VerificationEngine, VerificationCriteria
from chakrview.reasoning.engine import GovernedReasoningEngine
from chakrview.reasoning.policies import ReasoningPolicy
from chakrview.capability.registry import CapabilityRegistry
from chakrview.capability.provider import CalculatorCapabilityProvider
from chakrview.capability.gate import CapabilityGate


def benchmark_operation(func, iterations=1000):
    # Warmup
    for _ in range(max(5, iterations // 20)):
        func()
    t0 = time.perf_counter()
    for _ in range(iterations):
        func()
    elapsed = time.perf_counter() - t0
    avg_sec = elapsed / iterations
    ops_per_sec = iterations / elapsed if elapsed > 0 else 0
    return avg_sec, ops_per_sec


def run_all_benchmarks():
    print("=" * 70)
    print("CHAKRVIEW STEP 19: GOVERNED COGNITIVE REASONING BENCHMARK")
    print("=" * 70)

    results: Dict[str, Any] = {
        "step": 19,
        "name": "Governed Cognitive Reasoning Foundation",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "measurements": {},
        "scaling_evidence": {},
        "scaling_depth": {},
    }

    # 1. Task Creation
    def op_task():
        t = ReasoningTask(original_objective="Compute 10 + 20", task_type=ReasoningTaskType.MATHEMATICAL)
        t.transition_to(ReasoningPhase.DECOMPOSE)

    avg_s, ops = benchmark_operation(op_task, iterations=5000)
    results["measurements"]["task_creation"] = {
        "latency_us": round(avg_s * 1e6, 2),
        "throughput_ops_per_sec": int(ops),
    }
    print(f"[1] Task Creation: {avg_s * 1e6:.2f} µs/op ({int(ops):,} ops/sec)")

    # 2. Problem Decomposition
    decomposer = ProblemDecomposer()
    t_decomp = ReasoningTask(original_objective="calculate 25 * 4 + 50", task_type=ReasoningTaskType.MATHEMATICAL)

    def op_decomp():
        decomposer.decompose(t_decomp)

    avg_s, ops = benchmark_operation(op_decomp, iterations=2000)
    results["measurements"]["decomposition"] = {
        "latency_us": round(avg_s * 1e6, 2),
        "throughput_ops_per_sec": int(ops),
    }
    print(f"[2] Problem Decomposition: {avg_s * 1e6:.2f} µs/op ({int(ops):,} ops/sec)")

    # 3. Evidence Ingestion & Store Evaluation
    def op_evidence():
        store = EvidenceStore(max_items=50)
        e = EvidenceItem(evidence_type=EvidenceType.FACT, content="temperature is 24.5", confidence=1.0)
        store.add(e)
        store.get(e.evidence_id)

    avg_s, ops = benchmark_operation(op_evidence, iterations=5000)
    results["measurements"]["evidence_evaluation"] = {
        "latency_us": round(avg_s * 1e6, 2),
        "throughput_ops_per_sec": int(ops),
    }
    print(f"[3] Evidence Ingestion & Evaluation: {avg_s * 1e6:.2f} µs/op ({int(ops):,} ops/sec)")

    # 4. Hypothesis Evaluation
    hyp_engine = HypothesisEngine()
    store = EvidenceStore()
    e_sup = EvidenceItem(evidence_id="ev_s", confidence=1.0, reliability=1.0)
    store.add(e_sup)
    hyp = Hypothesis(statement="Subsystem is operational", supporting_evidence_ids=["ev_s"])

    def op_hyp():
        hyp_engine.evaluate_hypothesis(hyp, store)

    avg_s, ops = benchmark_operation(op_hyp, iterations=5000)
    results["measurements"]["hypothesis_evaluation"] = {
        "latency_us": round(avg_s * 1e6, 2),
        "throughput_ops_per_sec": int(ops),
    }
    print(f"[4] Hypothesis Evaluation: {avg_s * 1e6:.2f} µs/op ({int(ops):,} ops/sec)")

    # 5. Structured Inference Construction
    inf_engine = InferenceEngine()
    p1 = EvidenceItem(evidence_id="p1", confidence=0.9, reliability=0.9)
    p2 = EvidenceItem(evidence_id="p2", confidence=0.85, reliability=0.85)

    def op_inf():
        inf_engine.deduce([p1, p2], "modus_ponens", "Engine is nominal")

    avg_s, ops = benchmark_operation(op_inf, iterations=5000)
    results["measurements"]["inference_construction"] = {
        "latency_us": round(avg_s * 1e6, 2),
        "throughput_ops_per_sec": int(ops),
    }
    print(f"[5] Inference Construction: {avg_s * 1e6:.2f} µs/op ({int(ops):,} ops/sec)")

    # 6. Contradiction Detection
    detector = ContradictionDetector()
    c_store = EvidenceStore()
    ev1 = EvidenceItem(evidence_id="c1", content="sensor state: enabled", confidence=0.9)
    ev2 = EvidenceItem(evidence_id="c2", content="sensor state: disabled", confidence=0.9)
    c_store.add(ev1)
    c_store.add(ev2)

    def op_contra():
        c = detector.detect_contradiction(ev1, ev2, topic="sensor_state")
        if c:
            detector.evaluate_resolution(c, c_store)

    avg_s, ops = benchmark_operation(op_contra, iterations=5000)
    results["measurements"]["contradiction_detection"] = {
        "latency_us": round(avg_s * 1e6, 2),
        "throughput_ops_per_sec": int(ops),
    }
    print(f"[6] Contradiction Detection & Resolution: {avg_s * 1e6:.2f} µs/op ({int(ops):,} ops/sec)")

    # 7. Decision Generation & Selection
    dec_engine = DecisionEngine()
    cand1 = DecisionCandidate(candidate_id="c1", utility_score=0.8, risks=[], uncertainty_score=0.1)
    cand2 = DecisionCandidate(candidate_id="c2", utility_score=0.6, risks=["risk_timeout"], uncertainty_score=0.4)

    def op_dec():
        dec_engine.evaluate_and_select("task_bench", [cand1, cand2], store)

    avg_s, ops = benchmark_operation(op_dec, iterations=5000)
    results["measurements"]["decision_generation"] = {
        "latency_us": round(avg_s * 1e6, 2),
        "throughput_ops_per_sec": int(ops),
    }
    print(f"[7] Decision Selection: {avg_s * 1e6:.2f} µs/op ({int(ops):,} ops/sec)")

    # 8. Verification Loop
    v_engine = VerificationEngine()
    crit = [VerificationCriteria(expected_value=150.0, tolerance=1e-5, criterion_type="numerical_tolerance")]

    def op_vrf():
        v_engine.verify("dec_bench", expected_result=150.0, actual_result=150.0, criteria=crit)

    avg_s, ops = benchmark_operation(op_vrf, iterations=5000)
    results["measurements"]["verification"] = {
        "latency_us": round(avg_s * 1e6, 2),
        "throughput_ops_per_sec": int(ops),
    }
    print(f"[8] Verification Loop: {avg_s * 1e6:.2f} µs/op ({int(ops):,} ops/sec)")

    # 9. Full Reasoning Cycle
    registry = CapabilityRegistry()
    calc = CalculatorCapabilityProvider().get_capabilities()[0]
    registry.register(calc)
    gate = CapabilityGate(registry)
    engine = GovernedReasoningEngine()

    def op_full():
        engine.reason("Calculate 25 * 4 + 50", capability_registry=registry, capability_gate=gate)

    avg_s, ops = benchmark_operation(op_full, iterations=200)
    results["measurements"]["full_reasoning_cycle"] = {
        "latency_ms": round(avg_s * 1e3, 3),
        "throughput_ops_per_sec": int(ops),
    }
    print(f"[9] Full Governed Reasoning Cycle: {avg_s * 1e3:.3f} ms/cycle ({int(ops):,} cycles/sec)")

    # 10. Scaling with Evidence Volume (N = 10, 50, 100)
    print("\n[10] Scaling with Evidence Volume:")
    for n in (10, 50, 100):
        s_n = EvidenceStore(max_items=n + 10)
        for i in range(n):
            s_n.add(EvidenceItem(evidence_id=f"e_{i}", content=f"Fact {i}", confidence=0.8, reliability=0.8))

        def op_eval_store():
            items = s_n.all_items()
            for it in items[:10]:
                _ = it.effective_weight()

        avg_s, ops = benchmark_operation(op_eval_store, iterations=1000)
        results["scaling_evidence"][f"N_{n}"] = {
            "evidence_count": n,
            "latency_us": round(avg_s * 1e6, 2),
            "throughput_ops_per_sec": int(ops),
        }
        print(f"  - N={n} Evidence Items: {avg_s * 1e6:.2f} µs ({int(ops):,} ops/sec)")

    # 11. Scaling with Reasoning Depth (D = 1, 2, 4)
    print("\n[11] Scaling with Reasoning Depth:")
    for depth in (1, 2, 4):
        policy = ReasoningPolicy(max_depth=depth, max_subproblems=max(4, depth * 4))
        eng_d = GovernedReasoningEngine(policy=policy)

        def op_depth():
            eng_d.reason("Analyze and calculate 10 + 20")

        avg_s, ops = benchmark_operation(op_depth, iterations=100)
        results["scaling_depth"][f"Depth_{depth}"] = {
            "max_depth": depth,
            "latency_ms": round(avg_s * 1e3, 3),
            "throughput_ops_per_sec": int(ops),
        }
        print(f"  - Depth={depth}: {avg_s * 1e3:.3f} ms ({int(ops):,} cycles/sec)")

    # Save to json
    out_dir = os.path.join(os.path.dirname(__file__), "..", "docs")
    out_path = os.path.join(out_dir, "STEP_19_BENCHMARK_RESULTS.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    print(f"Benchmark results successfully written to {out_path}")
    print("=" * 70)


if __name__ == "__main__":
    run_all_benchmarks()
