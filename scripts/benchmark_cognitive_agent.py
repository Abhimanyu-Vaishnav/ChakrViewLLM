"""
Benchmark for Governed Cognitive Agent Execution Subsystem (Step 15).

Measures latency overhead on CPU across all cognitive layers:
- Task creation latency
- Bounded planning overhead
- Execution graph overhead (DAG construction, cycle validation, topological ordering)
- Skill lookup latency
- Governed tool-gate overhead (authorization + argument sanitization)
- Execution trace & sanitization overhead
- Complete end-to-end cognitive pipeline overhead
"""

import json
import os
from pathlib import Path
import statistics
import sys
import time
from typing import Dict, List, Any

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from chakrview.cognition.task import CognitiveTask, TaskStatus
from chakrview.cognition.planner import DeterministicRulePlanner
from chakrview.cognition.graph import ExecutionGraph
from chakrview.cognition.skill_selector import CognitiveSkillSelector
from chakrview.cognition.tool_gate import GovernedToolGate
from chakrview.cognition.trace import ExecutionTrace
from chakrview.cognition.controller import CognitiveController
from chakrview.cognition.profile import get_desktop_profile
from chakrview.runtime.skills import SkillPolicy, Skill, SkillDomain, get_standard_skill_registry


def benchmark_task_creation(iterations: int = 1000) -> Dict[str, float]:
    times = []
    for i in range(iterations):
        t0 = time.perf_counter()
        task = CognitiveTask(task_id=f"bench_{i}", user_request="Compute matrix multiplication")
        times.append((time.perf_counter() - t0) * 1e6)  # microseconds
    return {
        "mean_us": statistics.mean(times),
        "median_us": statistics.median(times),
        "stdev_us": statistics.stdev(times),
        "min_us": min(times),
        "max_us": max(times),
    }


def benchmark_planning_overhead(iterations: int = 500) -> Dict[str, float]:
    planner = DeterministicRulePlanner()
    skill_reg = get_standard_skill_registry()
    math_skill = skill_reg.get("skill_math_v1")
    task = CognitiveTask(task_id="t_plan", user_request="calculate 24 * 7 + 100")

    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        plan = planner.plan(task, active_skill=math_skill, available_tools=["calculator"])
        times.append((time.perf_counter() - t0) * 1e6)  # microseconds

    return {
        "mean_us": statistics.mean(times),
        "median_us": statistics.median(times),
        "stdev_us": statistics.stdev(times),
        "min_us": min(times),
        "max_us": max(times),
    }


def benchmark_execution_graph_overhead(iterations: int = 500) -> Dict[str, float]:
    planner = DeterministicRulePlanner()
    skill_reg = get_standard_skill_registry()
    math_skill = skill_reg.get("skill_math_v1")
    task = CognitiveTask(task_id="t_graph", user_request="calculate 50 * 2")
    plan = planner.plan(task, active_skill=math_skill, available_tools=["calculator"])

    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        graph = ExecutionGraph(plan)
        order = graph.get_topological_order()
        ready = graph.get_ready_steps()
        times.append((time.perf_counter() - t0) * 1e6)  # microseconds

    return {
        "mean_us": statistics.mean(times),
        "median_us": statistics.median(times),
        "stdev_us": statistics.stdev(times),
        "min_us": min(times),
        "max_us": max(times),
    }


def benchmark_skill_lookup(iterations: int = 500) -> Dict[str, float]:
    selector = CognitiveSkillSelector()
    queries = [
        "calculate 12 * 8",
        "write a python script to parse logs",
        "summarize financial statements for quarterly audit",
        "general question about ancient history",
    ]
    times = []
    for i in range(iterations):
        q = queries[i % len(queries)]
        t0 = time.perf_counter()
        match = selector.select_skill(q)
        times.append((time.perf_counter() - t0) * 1e6)  # microseconds

    return {
        "mean_us": statistics.mean(times),
        "median_us": statistics.median(times),
        "stdev_us": statistics.stdev(times),
        "min_us": min(times),
        "max_us": max(times),
    }


def benchmark_tool_gate(iterations: int = 500) -> Dict[str, float]:
    gate = GovernedToolGate()
    policy = SkillPolicy(allowed_tools=["calculator"])
    skill = Skill(
        skill_id="bench_skill",
        name="Bench Skill",
        version="1.0",
        domain=SkillDomain.MATHEMATICS,
        description="Bench",
        policy=policy,
    )
    args = {"expression": "128 + 256"}

    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        obs = gate.execute_governed("step_bench", "calculator", args, active_skill=skill)
        times.append((time.perf_counter() - t0) * 1e6)  # microseconds

    return {
        "mean_us": statistics.mean(times),
        "median_us": statistics.median(times),
        "stdev_us": statistics.stdev(times),
        "min_us": min(times),
        "max_us": max(times),
    }


def benchmark_trace_generation(iterations: int = 500) -> Dict[str, float]:
    times = []
    for i in range(iterations):
        t0 = time.perf_counter()
        trace = ExecutionTrace(task_id=f"task_bench_{i}")
        for s in range(4):
            trace.record_event("STEP_EVENT", step_id=f"step_{s}", details={"auth_token": "secret_abc", "value": s})
        trace.finalize()
        d = trace.to_dict()
        times.append((time.perf_counter() - t0) * 1e6)  # microseconds

    return {
        "mean_us": statistics.mean(times),
        "median_us": statistics.median(times),
        "stdev_us": statistics.stdev(times),
        "min_us": min(times),
        "max_us": max(times),
    }


def benchmark_full_pipeline(iterations: int = 100) -> Dict[str, float]:
    controller = CognitiveController()
    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        res = controller.execute_task("calculate 14 * 6 + 10")
        times.append((time.perf_counter() - t0) * 1e3)  # milliseconds

    return {
        "mean_ms": statistics.mean(times),
        "median_ms": statistics.median(times),
        "stdev_ms": statistics.stdev(times),
        "min_ms": min(times),
        "max_ms": max(times),
    }


def main():
    print("=" * 60)
    print("CHAKRVIEW STEP 15: COGNITIVE AGENT SUBSYSTEM BENCHMARK")
    print("Platform: CPU (Single Thread Execution Overhead)")
    print("=" * 60)

    print("\n1. Measuring CognitiveTask creation latency...")
    task_res = benchmark_task_creation()
    print(f"   Task Creation: {task_res['mean_us']:.2f} µs (median: {task_res['median_us']:.2f} µs)")

    print("\n2. Measuring Bounded Planning latency...")
    plan_res = benchmark_planning_overhead()
    print(f"   Planning: {plan_res['mean_us']:.2f} µs (median: {plan_res['median_us']:.2f} µs)")

    print("\n3. Measuring Execution Graph (DAG + Cycle Check) overhead...")
    graph_res = benchmark_execution_graph_overhead()
    print(f"   Execution Graph: {graph_res['mean_us']:.2f} µs (median: {graph_res['median_us']:.2f} µs)")

    print("\n4. Measuring Cognitive Skill Selection latency...")
    skill_res = benchmark_skill_lookup()
    print(f"   Skill Selection: {skill_res['mean_us']:.2f} µs (median: {skill_res['median_us']:.2f} µs)")

    print("\n5. Measuring Governed Tool-Gate execution latency...")
    gate_res = benchmark_tool_gate()
    print(f"   Governed Tool Gate: {gate_res['mean_us']:.2f} µs (median: {gate_res['median_us']:.2f} µs)")

    print("\n6. Measuring Execution Trace & Redaction overhead...")
    trace_res = benchmark_trace_generation()
    print(f"   Trace Logging: {trace_res['mean_us']:.2f} µs (median: {trace_res['median_us']:.2f} µs)")

    print("\n7. Measuring Full Cognitive Pipeline (Understand->Plan->Execute->Verify->Respond)...")
    pipe_res = benchmark_full_pipeline()
    print(f"   Full Cognitive Pipeline: {pipe_res['mean_ms']:.3f} ms (median: {pipe_res['median_ms']:.3f} ms)")

    results = {
        "step": 15,
        "name": "Cognitive Agent Subsystem Benchmark",
        "task_creation": task_res,
        "planning": plan_res,
        "execution_graph": graph_res,
        "skill_selection": skill_res,
        "tool_gate": gate_res,
        "trace_logging": trace_res,
        "full_pipeline": pipe_res,
        "timestamp": time.time(),
    }

    out_path = Path("docs") / "STEP_15_BENCHMARK_RESULTS.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n[OK] Benchmark results saved to: {out_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
