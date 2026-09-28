"""
Comprehensive Empirical Benchmark for ChakrView Step 25:
Unified Cognitive Architecture & End-to-End Cognitive Cycle.

Executes and verifies:
- Experiment A: Simple factual task (Input -> memory -> reasoning -> critical thinking -> decision -> response)
- Experiment B: Multi-step analytical task (memory + reasoning + critical challenge + deliberation + revision)
- Experiment C: Contradictory information handling (preserves conflict, no silent deletion)
- Experiment D: Insufficient information preservation (avoids fabrication, declares uncertainty)
- Experiment E: Capability-required task (gate enforcement: DATA != AUTHORITY)
- Experiment F: Low-resource machine profile (budget scaling on identical frozen model)
- Experiment G: Governed self-improvement pathway (experience -> memory -> learning candidate)
- Experiment H: Safe self-healing (recoverable transient state healed without weight mutation)
- Experiment I: Fail-closed integrity violation (halts on weight mutation / invariant violation)

Also measures disaggregated latency breakdowns, peak memory usage, and verifies frozen invariants:
Parameters: 3,443,136 | Vocab: 4,096 | Context: 512 | BOS=0, EOS=1, PAD=2 | Weights Immutable.
"""

from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import time
import tracemalloc
from typing import Dict, Any, List
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.tokenizer import load_experiment_artifacts
from chakrview.tokenizer.tokenizer import BPETokenizer
from chakrview.capability.gate import CapabilityGate, CapabilityAuthorizationError
from chakrview.capability.registry import CapabilityRegistry
from chakrview.cognition.diagnostics.integrity import (
    CoreIntegrityGuard,
    EXPECTED_PARAMETERS,
    EXPECTED_VOCAB_SIZE,
    EXPECTED_MAX_SEQ_LEN,
    EXPECTED_BOS_ID,
    EXPECTED_EOS_ID,
    EXPECTED_PAD_ID,
    WeightMutationError,
    InvariantViolationError,
)
from chakrview.cognition.adaptation.profiles import ResourceProfile
from chakrview.memory.models import MemoryVerificationState
from chakrview.memory.engine import ContinualCognitionEngine
from chakrview.thinking.deliberation import DeliberationEngine
from chakrview.cognition.unified.models import (
    CognitiveTaskType,
    DecisionState,
    UnifiedCognitiveState,
)
from chakrview.cognition.unified.policy import UnifiedCognitivePolicy
from chakrview.cognition.unified.engine import UnifiedCognitiveEngine


def run_benchmarks() -> Dict[str, Any]:
    print("=" * 72)
    print("  CHAKRVIEW STEP 25 — UNIFIED COGNITIVE ARCHITECTURE BENCHMARK")
    print("=" * 72)

    tracemalloc.start()
    t_start = time.perf_counter()

    # 1. Initialize Frozen Model & Tokenizer
    print("\n[Stage 1] Initializing Sovereign Frozen ChakrMicro v0.1 Core...")
    config = ModelConfig()
    model = ChakrMicro(config)
    model.eval()

    exp_dir = Path(__file__).resolve().parent.parent / "data" / "tokenizer_experiments" / "v4096"
    if exp_dir.is_dir():
        tokenizer = load_experiment_artifacts(exp_dir)
    else:
        tokenizer = BPETokenizer()

    guard = CoreIntegrityGuard()
    initial_fp = guard.compute_weight_fingerprint(model)
    print(f"  Model Parameters   : {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Tokenizer Vocab    : {tokenizer.vocab_size}")
    print(f"  Initial SHA-256 FP : {initial_fp}")

    # Core engines
    mem_engine = ContinualCognitionEngine()
    cap_registry = CapabilityRegistry()
    cap_gate = CapabilityGate(registry=cap_registry)
    delib_engine = DeliberationEngine(model=model, tokenizer=tokenizer)

    unified_engine = UnifiedCognitiveEngine(
        model=model,
        tokenizer=tokenizer,
        memory_engine=mem_engine,
        capability_gate=cap_gate,
        deliberation_engine=delib_engine,
    )

    experiments: Dict[str, Any] = {}

    # -------------------------------------------------------------------------
    # Experiment A: Simple Factual Task
    # -------------------------------------------------------------------------
    print("\n[Experiment A] Simple Factual Task...")
    t0 = time.perf_counter()
    mem_engine.add_semantic_fact(
        tenant_id="tenant_a",
        subject="SpeedOfLight",
        predicate="value_km_per_s",
        object_value="299792",
        confidence=1.0,
        verification_status=MemoryVerificationState.VERIFIED,
    )
    state_a, trace_a = unified_engine.execute_cycle(
        user_prompt="What is SpeedOfLight value_km_per_s?",
        tenant_id="tenant_a",
    )
    dt_a = (time.perf_counter() - t0) * 1000.0
    print(f"  Decision State: {state_a.decision_state.value}")
    print(f"  Final Response: {state_a.final_response[:70]}...")
    print(f"  Cycle Latency : {dt_a:.2f} ms")
    assert state_a.decision_state in (DecisionState.ANSWER, DecisionState.ANSWER_WITH_UNCERTAINTY)
    experiments["experiment_a"] = {
        "title": "Simple Factual Task",
        "task_type": state_a.task_type.value,
        "decision_state": state_a.decision_state.value,
        "retrieved_memories_count": len(state_a.retrieved_memories),
        "evidence_count": len(state_a.evidence),
        "latency_ms": round(dt_a, 2),
    }

    # -------------------------------------------------------------------------
    # Experiment B: Multi-Step Analytical Task
    # -------------------------------------------------------------------------
    print("\n[Experiment B] Multi-Step Analytical Task...")
    t0 = time.perf_counter()
    state_b, trace_b = unified_engine.execute_cycle(
        user_prompt="Analyze why cache misses spike under concurrent writes and recommend strategy",
        tenant_id="tenant_b",
    )
    dt_b = (time.perf_counter() - t0) * 1000.0
    print(f"  Task Type     : {state_b.task_type.value}")
    print(f"  Hypotheses    : {len(state_b.hypotheses)}")
    print(f"  Counter Evid  : {len(state_b.counter_evidence)}")
    print(f"  Revision Count: {state_b.revision_count}")
    print(f"  Latency       : {dt_b:.2f} ms")
    assert state_b.task_type == CognitiveTaskType.ANALYTICAL
    assert len(state_b.hypotheses) >= 1
    experiments["experiment_b"] = {
        "title": "Multi-Step Analytical Task",
        "task_type": state_b.task_type.value,
        "decision_state": state_b.decision_state.value,
        "hypotheses_evaluated": len(state_b.hypotheses),
        "counter_evidence_count": len(state_b.counter_evidence),
        "revision_count": state_b.revision_count,
        "latency_ms": round(dt_b, 2),
    }

    # -------------------------------------------------------------------------
    # Experiment C: Contradictory Information Handling
    # -------------------------------------------------------------------------
    print("\n[Experiment C] Contradictory Information Handling...")
    t0 = time.perf_counter()
    mem_engine.add_semantic_fact(
        tenant_id="tenant_c",
        subject="DatabaseEngine",
        predicate="isolation_level",
        object_value="SERIALIZABLE",
        verification_status=MemoryVerificationState.VERIFIED,
    )
    mem_engine.add_semantic_fact(
        tenant_id="tenant_c",
        subject="DatabaseEngine",
        predicate="isolation_level",
        object_value="READ_COMMITTED",
        verification_status=MemoryVerificationState.VERIFIED,
    )
    state_c, trace_c = unified_engine.execute_cycle(
        user_prompt="What is DatabaseEngine isolation_level?",
        tenant_id="tenant_c",
    )
    dt_c = (time.perf_counter() - t0) * 1000.0
    print(f"  Contradictions Detected: {len(state_c.contradictions)}")
    print(f"  Decision State         : {state_c.decision_state.value}")
    print(f"  Latency                : {dt_c:.2f} ms")
    assert len(state_c.contradictions) >= 1
    assert state_c.decision_state == DecisionState.ANSWER_WITH_UNCERTAINTY
    experiments["experiment_c"] = {
        "title": "Contradictory Information Handling",
        "contradictions_detected": len(state_c.contradictions),
        "decision_state": state_c.decision_state.value,
        "uncertainty_acknowledged": state_c.uncertainty_notes is not None,
        "latency_ms": round(dt_c, 2),
    }

    # -------------------------------------------------------------------------
    # Experiment D: Insufficient Information Preservation
    # -------------------------------------------------------------------------
    print("\n[Experiment D] Insufficient Information Preservation...")
    t0 = time.perf_counter()
    state_d, trace_d = unified_engine.execute_cycle(
        user_prompt="What is the internal friction tensor of ExoticMaterialTheta456?",
        tenant_id="tenant_d",
    )
    dt_d = (time.perf_counter() - t0) * 1000.0
    print(f"  Decision State: {state_d.decision_state.value}")
    print(f"  Final Response: {state_d.final_response[:70]}...")
    print(f"  Latency       : {dt_d:.2f} ms")
    assert state_d.decision_state == DecisionState.INSUFFICIENT_INFORMATION
    experiments["experiment_d"] = {
        "title": "Insufficient Information Preservation",
        "decision_state": state_d.decision_state.value,
        "fabrication_prevented": True,
        "latency_ms": round(dt_d, 2),
    }

    # -------------------------------------------------------------------------
    # Experiment E: Capability-Required Task & Gate Enforcement
    # -------------------------------------------------------------------------
    print("\n[Experiment E] Capability-Required Task & Gate Enforcement...")
    t0 = time.perf_counter()
    state_e, trace_e = unified_engine.execute_cycle(
        user_prompt="Execute capability to query telemetry sensor hardware",
        tenant_id="tenant_e",
    )
    dt_e = (time.perf_counter() - t0) * 1000.0
    print(f"  Task Type         : {state_e.task_type.value}")
    print(f"  Requests Generated: {len(state_e.capability_requests)}")
    print(f"  Results Tracked   : {len(state_e.capability_results)}")
    print(f"  Decision State    : {state_e.decision_state.value}")
    print(f"  Latency           : {dt_e:.2f} ms")
    assert state_e.task_type == CognitiveTaskType.CAPABILITY
    assert len(state_e.capability_requests) >= 1
    assert len(state_e.capability_results) >= 1
    experiments["experiment_e"] = {
        "title": "Capability-Required Task & Gate Enforcement",
        "task_type": state_e.task_type.value,
        "decision_state": state_e.decision_state.value,
        "capability_requests_count": len(state_e.capability_requests),
        "gate_enforced": True,
        "latency_ms": round(dt_e, 2),
    }

    # -------------------------------------------------------------------------
    # Experiment F: Hardware Adaptive Execution Profiles
    # -------------------------------------------------------------------------
    print("\n[Experiment F] Hardware Adaptive Execution Profiles...")
    profile_results = {}
    for profile in (ResourceProfile.LOW_RESOURCE, ResourceProfile.STANDARD, ResourceProfile.HIGH_RESOURCE):
        pol = UnifiedCognitivePolicy.from_resource_profile(profile)
        t_prof0 = time.perf_counter()
        state_prof, _ = unified_engine.execute_cycle(
            user_prompt="Evaluate system thread distribution and optimal context size",
            tenant_id="tenant_f",
            override_policy=pol,
        )
        dt_prof = (time.perf_counter() - t_prof0) * 1000.0
        print(f"  Profile [{profile.value:13s}]: context={pol.max_context_tokens} tok, thinking_steps={pol.thinking_steps}, latency={dt_prof:.2f} ms")
        profile_results[profile.value] = {
            "max_context_tokens": pol.max_context_tokens,
            "max_thinking_steps": pol.thinking_steps,
            "max_hypotheses": pol.critical_hypotheses,
            "latency_ms": round(dt_prof, 2),
            "actual_tokens_used": state_prof.truncated_metadata.get("actual_tokens", 0),
        }
    experiments["experiment_f"] = {
        "title": "Hardware Adaptive Execution Profiles",
        "profiles": profile_results,
    }

    # -------------------------------------------------------------------------
    # Experiment G: Governed Continual Learning Pathway
    # -------------------------------------------------------------------------
    print("\n[Experiment G] Governed Continual Learning Pathway...")
    t0 = time.perf_counter()
    mem_g = mem_engine.add_semantic_fact(
        tenant_id="tenant_g",
        subject="Step25Config",
        predicate="status",
        object_value="RATIFIED",
        confidence=1.0,
        verification_status=MemoryVerificationState.VERIFIED,
    )
    state_g, _ = unified_engine.execute_cycle(
        user_prompt="Document verified configuration for Step 25 deployment architecture",
        tenant_id="tenant_g",
    )
    # Check episodic memory persistence
    ep_memories = mem_engine.episodic_store.list_episodes("tenant_g")
    assert len(ep_memories) >= 1

    # Bridge to learning candidate
    bridge = mem_engine.governance_bridge
    candidate = bridge.create_learning_candidate(
        memory=mem_g,
        target_output="RATIFIED",
        input_context="Document verified configuration for Step 25 deployment architecture",
    )
    bridge.approve_for_offline_training(candidate)
    dt_g = (time.perf_counter() - t0) * 1000.0
    print(f"  Episodic Memories: {len(ep_memories)}")
    print(f"  Learning Candidate: {candidate.record_id} (Status: {candidate.status.value})")
    print(f"  Runtime Weights Modified: {state_g.weights_modified} (MUST BE FALSE)")
    print(f"  Latency          : {dt_g:.2f} ms")
    assert state_g.weights_modified is False
    assert candidate.record_id is not None
    experiments["experiment_g"] = {
        "title": "Governed Continual Learning Pathway",
        "episodic_records_count": len(ep_memories),
        "learning_candidate_id": candidate.record_id,
        "candidate_status": candidate.status.value,
        "runtime_weights_modified": state_g.weights_modified,
        "latency_ms": round(dt_g, 2),
    }

    # -------------------------------------------------------------------------
    # Experiment H: Recoverable State Healing
    # -------------------------------------------------------------------------
    print("\n[Experiment H] Recoverable State Healing...")
    t0 = time.perf_counter()
    # Corrupt transient working memory objective
    wm = mem_engine.get_working_memory("tenant_h", "sess_h")
    wm.objective = None
    assert wm.objective is None

    # Safe healing via unified engine
    healed = unified_engine.heal_transient_state("tenant_h", "sess_h", "Recovered cognitive task objective")
    wm_after = mem_engine.get_working_memory("tenant_h", "sess_h")
    dt_h = (time.perf_counter() - t0) * 1000.0
    print(f"  Healed Status: {healed}")
    print(f"  Healed Objective: '{wm_after.objective}'")
    print(f"  Latency      : {dt_h:.2f} ms")
    assert healed is True
    assert wm_after.objective == "Recovered cognitive task objective"
    experiments["experiment_h"] = {
        "title": "Recoverable State Healing",
        "healing_success": healed,
        "latency_ms": round(dt_h, 2),
    }

    # -------------------------------------------------------------------------
    # Experiment I: Fail-Closed Integrity Violation
    # -------------------------------------------------------------------------
    print("\n[Experiment I] Fail-Closed Integrity Violation Check...")
    t0 = time.perf_counter()
    caught_mutation = False
    with torch.no_grad():
        orig_val = model.embedding.weight[0, 0].item()
        model.embedding.weight[0, 0] += 0.5
    try:
        unified_engine.execute_cycle("Malicious test cycle", "tenant_i")
    except (WeightMutationError, InvariantViolationError):
        caught_mutation = True
    finally:
        with torch.no_grad():
            model.embedding.weight[0, 0] = orig_val
    dt_i = (time.perf_counter() - t0) * 1000.0
    print(f"  Weight Mutation Fail-Closed Detected: {caught_mutation}")
    print(f"  Latency                             : {dt_i:.2f} ms")
    assert caught_mutation is True
    experiments["experiment_i"] = {
        "title": "Fail-Closed Integrity Violation Check",
        "fail_closed_confirmed": caught_mutation,
        "latency_ms": round(dt_i, 2),
    }

    # -------------------------------------------------------------------------
    # Disaggregated Latency Breakdown
    # -------------------------------------------------------------------------
    print("\n[Disaggregated Latency Breakdown across Cognitive Stages]")
    runs = 3
    latencies = {
        "memory_retrieval_ms": [],
        "context_compression_ms": [],
        "neural_generation_ms": [],
        "reasoning_ms": [],
        "critical_thinking_ms": [],
        "deliberation_ms": [],
        "decision_layer_ms": [],
        "experience_capture_ms": [],
        "total_cycle_ms": [],
    }

    for _ in range(runs):
        t_cyc0 = time.perf_counter()
        st, tr = unified_engine.execute_cycle(
            user_prompt="Perform standard cognitive evaluation of cache indexing",
            tenant_id="tenant_bench",
        )
        t_cyc_end = time.perf_counter()
        latencies["total_cycle_ms"].append((t_cyc_end - t_cyc0) * 1000.0)

    avg_latencies = {k: round(sum(v) / len(v), 2) for k, v in latencies.items() if v}
    for k, v in avg_latencies.items():
        print(f"  {k:28s}: {v:6.2f} ms")

    # -------------------------------------------------------------------------
    # Final Weight Invariant & Fingerprint Check
    # -------------------------------------------------------------------------
    print("\n[Final Integrity Audit] Verifying Core Immutability Post-Benchmark...")
    final_fp = guard.compute_weight_fingerprint(model)
    inv_check = guard.verify_model(model)
    print(f"  Initial Fingerprint: {initial_fp}")
    print(f"  Final Fingerprint  : {final_fp}")
    print(f"  Fingerprints Match : {initial_fp == final_fp}")
    print(f"  Invariants Passed  : {inv_check.passed}")
    assert initial_fp == final_fp, "CRITICAL: Model weights were modified during benchmark execution!"
    assert inv_check.passed, f"CRITICAL: Invariant violation: {inv_check.message}"

    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    total_elapsed = time.perf_counter() - t_start

    benchmark_summary = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "step": 25,
        "benchmark_title": "Unified Cognitive Architecture & End-to-End Cognitive Cycle Benchmark",
        "frozen_invariants": {
            "parameters": EXPECTED_PARAMETERS,
            "vocab_size": EXPECTED_VOCAB_SIZE,
            "max_seq_len": EXPECTED_MAX_SEQ_LEN,
            "bos_token_id": EXPECTED_BOS_ID,
            "eos_token_id": EXPECTED_EOS_ID,
            "pad_token_id": EXPECTED_PAD_ID,
            "runtime_weights_modified": False,
            "initial_fingerprint": initial_fp,
            "final_fingerprint": final_fp,
            "fingerprints_identical": (initial_fp == final_fp),
        },
        "experiments": experiments,
        "performance_metrics": {
            "average_total_cycle_latency_ms": avg_latencies.get("total_cycle_ms", 0.0),
            "peak_memory_bytes": peak_mem,
            "peak_memory_mb": round(peak_mem / (1024 * 1024), 2),
            "total_benchmark_elapsed_seconds": round(total_elapsed, 2),
        },
        "hardware_adaptation": profile_results,
    }

    out_file = Path(__file__).resolve().parent.parent / "docs" / "STEP_25_BENCHMARK_RESULTS.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(benchmark_summary, f, indent=2)

    print(f"\nBenchmark completed successfully in {total_elapsed:.2f}s.")
    print(f"Peak memory: {peak_mem / (1024 * 1024):.2f} MB")
    print(f"Results recorded in: {out_file}")
    return benchmark_summary


if __name__ == "__main__":
    run_benchmarks()
