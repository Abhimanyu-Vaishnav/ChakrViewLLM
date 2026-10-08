"""
Execution Runner for Waves 581-590:
ChakrView v0.4.0 Candidate Cognitive Verification & Audit.
"""

from __future__ import annotations

import os
import sys
import json
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from chakrview.brain.canonical_hash import verify_canonical_baseline
from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro

from chakrview.cognition.v040_candidate.unified_benchmark import (
    UnifiedCognitiveBenchmarkRunner,
)
from chakrview.cognition.v040_candidate.v040_master_gate import (
    V040CandidateMasterGate,
)


def run_waves_581_590_candidate_audit():
    print("=" * 70)
    print("CHAKRVIEW WAVES 581-590: v0.4.0 CANDIDATE COGNITIVE VERIFICATION")
    print("=" * 70)

    # 1. Canonical Neural Baseline Check
    torch.manual_seed(42)
    model = ChakrMicro(ModelConfig())
    base_res = verify_canonical_baseline(model)
    print(f"CANONICAL SHA256: {base_res['digest']}")
    print(f"PARAMETERS:       {base_res['total_params']}")
    print(f"DELTA W:          {base_res['delta_w']}")
    print(f"IS BIT EXACT:     {base_res['is_bit_exact']}")

    if not base_res["is_bit_exact"] or base_res["delta_w"] != 0:
        print("BASELINE_INTEGRITY_FAILURE")
        sys.exit(1)

    # 2. Candidate Master Gate Audit
    print("\nRUNNING v0.4.0 CANDIDATE MASTER GATE AUDIT...")
    gate_report = V040CandidateMasterGate.evaluate_candidate()
    print(f"  Working Memory 2.0:           {'PASS' if gate_report.working_memory_verified else 'FAIL'}")
    print(f"  Long-Context Chunking:         {'PASS' if gate_report.long_context_chunking_verified else 'FAIL'}")
    print(f"  Hierarchical Planning:         {'PASS' if gate_report.hierarchical_planning_verified else 'FAIL'}")
    print(f"  Multi-Step Goal Execution:     {'PASS' if gate_report.multi_step_execution_verified else 'FAIL'}")
    print(f"  Memory-Reasoning Precedence:   {'PASS' if gate_report.memory_reasoning_verified else 'FAIL'}")
    print(f"  Experience-Reasoning Bridge:   {'PASS' if gate_report.experience_reasoning_verified else 'FAIL'}")
    print(f"  Language-Cognitive Bridge:     {'PASS' if gate_report.language_cognitive_verified else 'FAIL'}")
    print(f"  Adaptive Problem Solving:      {'PASS' if gate_report.adaptive_strategy_verified else 'FAIL'}")
    print(f"  Canonical Baseline Intact:     {'PASS' if gate_report.canonical_baseline_intact else 'FAIL'}")
    print(f"  Audit Verdict:                 {gate_report.audit_verdict}")

    # 3. Unified Cognitive Capability Benchmark
    print("\nRUNNING UNIFIED COGNITIVE BENCHMARK (HELD-OUT N=40)...")
    bench_res = UnifiedCognitiveBenchmarkRunner.evaluate_benchmark(partition="FINAL_HELD_OUT")
    print(f"  Passed Cases: {bench_res['passed_cases']}/{bench_res['total_cases']} ({bench_res['accuracy']*100:.1f}%)")

    # 4. Persist Artifact
    out_dir = "artifacts/waves_528_536_execution"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "wave581_590_v040_candidate_results.json")

    summary = {
        "timestamp": time.time(),
        "canonical_sha256": base_res["digest"],
        "parameter_count": base_res["total_params"],
        "delta_w": 0,
        "candidate_verdict": gate_report.audit_verdict,
        "benchmark_accuracy": bench_res["accuracy"],
        "total_benchmark_cases": bench_res["total_cases"],
        "hardware_adaptation": {
            "cpu_first": True,
            "streaming_chunking_supported": True,
            "bounded_working_memory": True,
        },
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved execution artifact to: {out_path}")
    print(f"\nSTATUS: {gate_report.audit_verdict}")


if __name__ == "__main__":
    run_waves_581_590_candidate_audit()
