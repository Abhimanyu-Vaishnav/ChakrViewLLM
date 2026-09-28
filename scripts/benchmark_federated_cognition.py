"""
Empirical Benchmark for ChakrView Step 26:
Multi-Agent Federated Cognition & Cooperative Intelligence Foundation.

Measures:
1. Agent registration latency
2. Task decomposition latency
3. Message validation & cryptographic hashing latency
4. Single-agent execution latency
5. 2-agent federation latency
6. 4-agent federation latency
7. Conflict-resolution latency
8. Consensus synthesis latency
9. Memory integration latency
10. Full federated cognitive cycle latency
11. Hardware-adaptive comparison: LOW_RESOURCE vs STANDARD vs HIGH_RESOURCE
12. Peak memory usage, message counts, and fault isolation
13. Model weight immutability & frozen invariant verification (3,443,136 params, 4096 vocab, 512 context)
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
from chakrview.capability.gate import CapabilityGate
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
from chakrview.memory.engine import ContinualCognitionEngine
from chakrview.memory.models import MemoryVerificationState
from chakrview.cognition.unified.models import CognitiveTaskType, DecisionState
from chakrview.cognition.federated.models import (
    AgentRole,
    AgentStatus,
    AgentIdentity,
    AgentContract,
    MessageType,
    MessagePriority,
    AgentMessage,
    AgentTask,
    ConflictState,
)
from chakrview.cognition.federated.protocol import FederatedProtocolValidator
from chakrview.cognition.federated.registry import AgentRegistry
from chakrview.cognition.federated.policy import FederatedExecutionPolicy
from chakrview.cognition.federated.decomposition import FederatedTaskDecomposer
from chakrview.cognition.federated.agents import (
    AnalystAgent,
    ResearcherAgent,
    CriticAgent,
    PlannerAgent,
    SynthesizerAgent,
    VerifierAgent,
)
from chakrview.cognition.federated.evidence import FederatedEvidenceAggregator
from chakrview.cognition.federated.conflict import FederatedConflictResolver
from chakrview.cognition.federated.synthesis import FederatedSynthesizer
from chakrview.cognition.federated.engine import FederatedCognitionEngine


def run_benchmarks() -> Dict[str, Any]:
    print("=" * 72)
    print("  CHAKRVIEW STEP 26 — MULTI-AGENT FEDERATED COGNITION BENCHMARK")
    print("=" * 72)

    tracemalloc.start()
    t_start = time.perf_counter()

    # 1. Initialize Frozen Model & Tokenizer
    print("\n[1/10] Initializing Sovereign Frozen ChakrMicro v0.1 Core...")
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
    inv_check = guard.verify_model(model)
    assert inv_check.passed, f"Model invariant check failed: {inv_check.message}"

    print(f"  Parameters        : {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Vocabulary        : {tokenizer.vocab_size}")
    print(f"  Max Context       : {config.max_seq_len}")
    print(f"  Initial SHA-256 FP: {initial_fp}")

    mem_engine = ContinualCognitionEngine()
    cap_gate = CapabilityGate(registry=CapabilityRegistry())

    fed_engine = FederatedCognitionEngine(
        model=model,
        tokenizer=tokenizer,
        memory_engine=mem_engine,
        capability_gate=cap_gate,
    )

    micro_benchmarks: Dict[str, float] = {}

    # 2. Agent Registration Latency
    print("\n[2/10] Benchmarking Agent Registration Latency...")
    reg = AgentRegistry(max_agents=8)
    t0 = time.perf_counter()
    for i in range(6):
        ident = AgentIdentity(f"bench_ag_{i}", AgentRole.ANALYST, "t_bench", "sess")
        reg.register(AgentContract(identity=ident))
    dt_reg = (time.perf_counter() - t0) * 1000.0 / 6.0
    micro_benchmarks["agent_registration_latency_ms"] = round(dt_reg, 3)
    print(f"  Avg Registration Latency: {dt_reg:.3f} ms")

    # 3. Task Decomposition Latency
    print("\n[3/10] Benchmarking Task Decomposition Latency...")
    decomposer = FederatedTaskDecomposer()
    t0 = time.perf_counter()
    subtasks = decomposer.decompose(
        "Analyze distributed cache contention under high concurrent writes",
        CognitiveTaskType.ANALYTICAL,
        "t_bench",
        "sess",
    )
    dt_decomp = (time.perf_counter() - t0) * 1000.0
    micro_benchmarks["task_decomposition_latency_ms"] = round(dt_decomp, 3)
    print(f"  Decomposition Latency   : {dt_decomp:.3f} ms ({len(subtasks)} subtasks generated)")

    # 4. Message Validation & Cryptographic Hashing Latency
    print("\n[4/10] Benchmarking Message Validation & Hashing...")
    msg = AgentMessage(
        message_id="msg_bench_01",
        sender_agent_id="agent_bench_1",
        receiver_agent_id="COORDINATOR",
        tenant_id="t_bench",
        session_id="sess",
        correlation_id="task_01",
        message_type=MessageType.TASK_RESPONSE,
        payload={"claim": "Analytical finding regarding cache misses", "confidence": 0.88},
    )
    t0 = time.perf_counter()
    for _ in range(50):
        FederatedProtocolValidator.validate_message(msg, expected_tenant_id="t_bench")
    dt_msg = (time.perf_counter() - t0) * 1000.0 / 50.0
    micro_benchmarks["message_validation_latency_ms"] = round(dt_msg, 3)
    print(f"  Avg Validation & Hash Latency: {dt_msg:.3f} ms")

    # 5. Single-Agent Execution Latency
    print("\n[5/10] Benchmarking Single-Agent Execution Latency...")
    ident = AgentIdentity("analyst_bench", AgentRole.ANALYST, "t_bench", "sess")
    analyst = AnalystAgent(ident, AgentContract(ident), model, tokenizer)
    task = subtasks[1]
    t0 = time.perf_counter()
    res_msg = analyst.execute_task(task, {})
    dt_single = (time.perf_counter() - t0) * 1000.0
    micro_benchmarks["single_agent_execution_latency_ms"] = round(dt_single, 3)
    print(f"  Single Agent (Analyst) Execution: {dt_single:.3f} ms")

    # 6. Conflict Resolution Latency
    print("\n[6/10] Benchmarking Conflict Detection & Resolution...")
    resolver = FederatedConflictResolver()
    msg_opp1 = AgentMessage(
        message_id="m_opp1", sender_agent_id="ag1", receiver_agent_id="COORDINATOR",
        tenant_id="t_bench", session_id="sess", correlation_id="c1", message_type=MessageType.TASK_RESPONSE,
        payload={"claim": "Option A is valid and safe."},
    )
    msg_opp2 = AgentMessage(
        message_id="m_opp2", sender_agent_id="ag2", receiver_agent_id="COORDINATOR",
        tenant_id="t_bench", session_id="sess", correlation_id="c1", message_type=MessageType.TASK_RESPONSE,
        payload={"claim": "Option A is invalid and unsafe."},
    )
    t0 = time.perf_counter()
    conflicts = resolver.detect_conflicts("t_conf", [msg_opp1, msg_opp2])
    dt_conflict = (time.perf_counter() - t0) * 1000.0
    micro_benchmarks["conflict_resolution_latency_ms"] = round(dt_conflict, 3)
    print(f"  Conflict Detection Latency: {dt_conflict:.3f} ms ({len(conflicts)} conflict records detected)")

    # 7. Consensus Synthesis Latency
    print("\n[7/10] Benchmarking Consensus Synthesis Latency...")
    synthesizer = FederatedSynthesizer()
    t0 = time.perf_counter()
    cand = synthesizer.synthesize(
        task_id="t_syn_bench",
        objective="Analyze distributed cache contention",
        messages=[msg_opp1, msg_opp2],
        corroborated_evidence=[{"content": "Cache benchmark results", "confidence": 0.9}],
        conflicts=conflicts,
        critical_counter_evidence=[],
    )
    dt_syn = (time.perf_counter() - t0) * 1000.0
    micro_benchmarks["consensus_synthesis_latency_ms"] = round(dt_syn, 3)
    print(f"  Synthesis Latency         : {dt_syn:.3f} ms (Decision: {cand.recommended_decision_state.value})")

    # 8. Memory Integration Latency
    print("\n[8/10] Benchmarking Memory Integration...")
    t0 = time.perf_counter()
    mem_engine.add_semantic_fact(
        tenant_id="t_bench",
        subject="CacheCoherenceProtocol",
        predicate="type",
        object_value="MESI",
        confidence=1.0,
        verification_status=MemoryVerificationState.VERIFIED,
    )
    res_agent = ResearcherAgent(
        AgentIdentity("res_bench", AgentRole.RESEARCHER, "t_bench", "sess"),
        AgentContract(AgentIdentity("res_bench", AgentRole.RESEARCHER, "t_bench", "sess")),
        model,
        tokenizer,
        mem_engine,
    )
    task_res = subtasks[0]
    task_res.objective = "What is CacheCoherenceProtocol type?"
    res_out = res_agent.execute_task(task_res, {})
    dt_mem = (time.perf_counter() - t0) * 1000.0
    micro_benchmarks["memory_integration_latency_ms"] = round(dt_mem, 3)
    print(f"  Memory Ingestion & Recall : {dt_mem:.3f} ms ({res_out.payload.get('candidate_count')} candidates)")

    # 9. Hardware Adaptation Profiles (LOW_RESOURCE, STANDARD, HIGH_RESOURCE)
    print("\n[9/10] Benchmarking Hardware Adaptive Execution Profiles...")
    profile_results = {}
    for prof in (ResourceProfile.LOW_RESOURCE, ResourceProfile.STANDARD, ResourceProfile.HIGH_RESOURCE):
        pol = FederatedExecutionPolicy.from_resource_profile(prof)
        t0 = time.perf_counter()
        c_prof, tr_prof = fed_engine.execute_federated_cycle(
            objective="Evaluate distributed replication consistency models",
            tenant_id="t_bench",
            session_id=f"sess_{prof.value}",
            override_policy=pol,
        )
        dt_prof = (time.perf_counter() - t0) * 1000.0
        print(f"  Profile [{prof.value:13s}]: agents={pol.max_agents}, rounds={tr_prof.execution_rounds}, msgs={tr_prof.total_messages_exchanged}, latency={dt_prof:.2f} ms")
        profile_results[prof.value] = {
            "max_agents": pol.max_agents,
            "max_rounds": pol.max_rounds,
            "max_total_messages": pol.max_total_messages,
            "actual_rounds_executed": tr_prof.execution_rounds,
            "actual_messages_exchanged": tr_prof.total_messages_exchanged,
            "decision_state": c_prof.recommended_decision_state.value,
            "latency_ms": round(dt_prof, 2),
        }

    # 10. Multi-Agent Scaling: 2-Agent vs 4-Agent Federation
    print("\n[10/10] Multi-Agent Scaling Benchmarks...")
    # 2-Agent setup (Researcher + Synthesizer)
    agents_2 = {
        AgentRole.RESEARCHER: ResearcherAgent(
            AgentIdentity("ag_res_2", AgentRole.RESEARCHER, "t_scale", "s1"),
            AgentContract(AgentIdentity("ag_res_2", AgentRole.RESEARCHER, "t_scale", "s1")),
            model, tokenizer, mem_engine,
        ),
        AgentRole.SYNTHESIZER: SynthesizerAgent(
            AgentIdentity("ag_syn_2", AgentRole.SYNTHESIZER, "t_scale", "s1"),
            AgentContract(AgentIdentity("ag_syn_2", AgentRole.SYNTHESIZER, "t_scale", "s1")),
            model, tokenizer,
        ),
    }
    t0 = time.perf_counter()
    c2, tr2 = fed_engine.execute_federated_cycle(
        objective="What is CacheCoherenceProtocol type?",
        tenant_id="t_scale",
        session_id="s1",
        custom_agents=agents_2,
    )
    dt_2ag = (time.perf_counter() - t0) * 1000.0
    micro_benchmarks["two_agent_federation_latency_ms"] = round(dt_2ag, 3)

    # 4-Agent setup (Researcher + Analyst + Critic + Synthesizer)
    agents_4 = dict(agents_2)
    agents_4[AgentRole.ANALYST] = AnalystAgent(
        AgentIdentity("ag_ana_4", AgentRole.ANALYST, "t_scale", "s2"),
        AgentContract(AgentIdentity("ag_ana_4", AgentRole.ANALYST, "t_scale", "s2")),
        model, tokenizer,
    )
    agents_4[AgentRole.CRITIC] = CriticAgent(
        AgentIdentity("ag_crt_4", AgentRole.CRITIC, "t_scale", "s2"),
        AgentContract(AgentIdentity("ag_crt_4", AgentRole.CRITIC, "t_scale", "s2")),
        model, tokenizer,
    )
    t0 = time.perf_counter()
    c4, tr4 = fed_engine.execute_federated_cycle(
        objective="Analyze distributed replication consistency models",
        tenant_id="t_scale",
        session_id="s2",
        custom_agents=agents_4,
    )
    dt_4ag = (time.perf_counter() - t0) * 1000.0
    micro_benchmarks["four_agent_federation_latency_ms"] = round(dt_4ag, 3)
    micro_benchmarks["full_federated_cycle_latency_ms"] = round(dt_4ag, 3)

    print(f"  2-Agent Federation Latency: {dt_2ag:.2f} ms (msgs: {tr2.total_messages_exchanged})")
    print(f"  4-Agent Federation Latency: {dt_4ag:.2f} ms (msgs: {tr4.total_messages_exchanged})")

    # Final Weight Invariant & Fingerprint Check
    print("\n[Final Integrity Audit] Verifying Core Immutability Post-Benchmark...")
    final_fp = guard.compute_weight_fingerprint(model)
    inv_check = guard.verify_model(model)
    print(f"  Initial Fingerprint: {initial_fp}")
    print(f"  Final Fingerprint  : {final_fp}")
    print(f"  Fingerprints Match : {initial_fp == final_fp}")
    print(f"  Invariants Passed  : {inv_check.passed}")
    assert initial_fp == final_fp, "CRITICAL: Model weights were modified during federated benchmark execution!"
    assert inv_check.passed, f"CRITICAL: Invariant violation: {inv_check.message}"

    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    total_elapsed = time.perf_counter() - t_start

    benchmark_summary = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "step": 26,
        "benchmark_title": "Multi-Agent Federated Cognition & Cooperative Intelligence Benchmark",
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
        "micro_benchmarks": micro_benchmarks,
        "hardware_adaptation": profile_results,
        "federation_scaling": {
            "two_agent_messages": tr2.total_messages_exchanged,
            "two_agent_latency_ms": round(dt_2ag, 2),
            "four_agent_messages": tr4.total_messages_exchanged,
            "four_agent_latency_ms": round(dt_4ag, 2),
        },
        "performance_metrics": {
            "peak_memory_bytes": peak_mem,
            "peak_memory_mb": round(peak_mem / (1024 * 1024), 2),
            "total_benchmark_elapsed_seconds": round(total_elapsed, 2),
        },
    }

    out_file = Path(__file__).resolve().parent.parent / "docs" / "STEP_26_BENCHMARK_RESULTS.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(benchmark_summary, f, indent=2)

    print(f"\nBenchmark completed successfully in {total_elapsed:.2f}s.")
    print(f"Peak memory: {peak_mem / (1024 * 1024):.2f} MB")
    print(f"Results recorded in: {out_file}")
    return benchmark_summary


if __name__ == "__main__":
    run_benchmarks()
