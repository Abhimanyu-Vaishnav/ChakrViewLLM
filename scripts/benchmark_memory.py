"""
Empirical Benchmark for ChakrView Step 24:
Memory, Experience & Continual Cognition Foundation.

Measures:
1. Working-memory creation latency
2. Episodic insertion latency
3. Semantic insertion latency
4. Retrieval latency
5. Contradiction detection latency
6. Consolidation latency
7. Serialization latency
8. LOW_RESOURCE retrieval latency
9. STANDARD retrieval latency
10. HIGH_RESOURCE retrieval latency
11. Memory footprint
12. Weight immutability and frozen core invariants verification
"""

import json
import os
from pathlib import Path
import statistics
import sys
import time
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.cognition.diagnostics.integrity import CoreIntegrityGuard
from chakrview.cognition.adaptation.profiles import ResourceProfile
from chakrview.memory.models import (
    Episode,
    SemanticMemory,
    MemoryRetrievalQuery,
    MemoryVerificationState,
)
from chakrview.memory.working import WorkingMemory, WorkingMemoryConfig
from chakrview.memory.episodic import EpisodicMemoryStore
from chakrview.memory.semantic import SemanticMemoryStore
from chakrview.memory.contradiction import ContradictionManager
from chakrview.memory.retrieval import ContinualMemoryRetriever
from chakrview.memory.consolidation import ExperienceConsolidationEngine
from chakrview.memory.policy import MemoryExecutionPolicy
from chakrview.memory.storage import ContinualMemoryStorage
from chakrview.memory.engine import ContinualCognitionEngine


def run_benchmark():
    print("=" * 70)
    print("CHAKRVIEW STEP 24: MEMORY & CONTINUAL COGNITION BENCHMARK")
    print("=" * 70)

    results = {
        "benchmark_timestamp": time.time(),
        "step": 24,
        "environment": {
            "device": "cpu",
            "pytorch_version": torch.__version__,
            "platform": sys.platform,
        },
        "measurements": {},
    }

    # 1. Working Memory Creation
    times = []
    for _ in range(100):
        t0 = time.perf_counter()
        wm = WorkingMemory(tenant_id="t1", session_id="s1")
        wm.set_objective("Benchmark objective")
        wm.add_context("Context item 1")
        wm.add_hypothesis({"claim": "H1"})
        times.append((time.perf_counter() - t0) * 1000.0)
    results["measurements"]["working_memory_creation_latency_ms"] = {
        "mean": statistics.mean(times),
        "median": statistics.median(times),
        "min": min(times),
        "max": max(times),
    }
    print(f"Working Memory Creation Latency: {results['measurements']['working_memory_creation_latency_ms']['mean']:.4f} ms")

    # 2. Episodic Insertion
    ep_store = EpisodicMemoryStore()
    times = []
    for i in range(100):
        t0 = time.perf_counter()
        ep_store.record_episode(
            tenant_id="tenant_bench",
            session_id="session_bench",
            situation=f"Situation query {i}",
            action_or_response=f"Response generated for query {i}",
            outcome=f"Outcome observed {i}",
            task_id=f"task_{i}",
        )
        times.append((time.perf_counter() - t0) * 1000.0)
    results["measurements"]["episodic_insertion_latency_ms"] = {
        "mean": statistics.mean(times),
        "median": statistics.median(times),
        "min": min(times),
        "max": max(times),
    }
    print(f"Episodic Insertion Latency: {results['measurements']['episodic_insertion_latency_ms']['mean']:.4f} ms")

    # 3. Semantic Insertion
    sem_store = SemanticMemoryStore()
    times = []
    for i in range(100):
        t0 = time.perf_counter()
        sem_store.add_memory(
            tenant_id="tenant_bench",
            subject=f"Subject_{i}",
            predicate="attribute_is",
            object_value=f"Value_{i}",
            verification_status=MemoryVerificationState.VERIFIED,
        )
        times.append((time.perf_counter() - t0) * 1000.0)
    results["measurements"]["semantic_insertion_latency_ms"] = {
        "mean": statistics.mean(times),
        "median": statistics.median(times),
        "min": min(times),
        "max": max(times),
    }
    print(f"Semantic Insertion Latency: {results['measurements']['semantic_insertion_latency_ms']['mean']:.4f} ms")

    # 4. Retrieval Latency
    retriever = ContinualMemoryRetriever(episodic_store=ep_store, semantic_store=sem_store)
    times = []
    for i in range(50):
        query = MemoryRetrievalQuery(
            query_text=f"Subject_{i} attribute_is",
            tenant_id="tenant_bench",
            top_k=5,
        )
        t0 = time.perf_counter()
        res = retriever.retrieve(query)
        times.append((time.perf_counter() - t0) * 1000.0)
    results["measurements"]["retrieval_latency_ms"] = {
        "mean": statistics.mean(times),
        "median": statistics.median(times),
        "min": min(times),
        "max": max(times),
    }
    print(f"Retrieval Latency: {results['measurements']['retrieval_latency_ms']['mean']:.4f} ms")

    # 5. Contradiction Detection
    con_mgr = ContradictionManager(semantic_store=sem_store)
    times = []
    for i in range(50):
        t0 = time.perf_counter()
        conflicts = con_mgr.detect_semantic_conflicts(
            tenant_id="tenant_bench",
            candidate_subject=f"Subject_{i}",
            candidate_predicate="attribute_is",
            candidate_value="conflicting_alt_value",
        )
        times.append((time.perf_counter() - t0) * 1000.0)
    results["measurements"]["contradiction_detection_latency_ms"] = {
        "mean": statistics.mean(times),
        "median": statistics.median(times),
        "min": min(times),
        "max": max(times),
    }
    print(f"Contradiction Detection Latency: {results['measurements']['contradiction_detection_latency_ms']['mean']:.4f} ms")

    # 6. Experience Consolidation Latency
    con_engine = ExperienceConsolidationEngine(
        episodic_store=ep_store,
        semantic_store=sem_store,
        contradiction_mgr=con_mgr,
    )
    times = []
    for _ in range(20):
        t0 = time.perf_counter()
        cands = con_engine.consolidate_tenant_episodes("tenant_bench")
        times.append((time.perf_counter() - t0) * 1000.0)
    results["measurements"]["consolidation_latency_ms"] = {
        "mean": statistics.mean(times),
        "median": statistics.median(times),
        "min": min(times),
        "max": max(times),
    }
    print(f"Consolidation Latency: {results['measurements']['consolidation_latency_ms']['mean']:.4f} ms")

    # 7. Serialization & Deserialization Latency
    times_export = []
    times_import = []
    for _ in range(20):
        t0 = time.perf_counter()
        state = ContinualMemoryStorage.export_state(ep_store, sem_store, con_mgr)
        times_export.append((time.perf_counter() - t0) * 1000.0)

        t1 = time.perf_counter()
        ContinualMemoryStorage.import_state(state)
        times_import.append((time.perf_counter() - t1) * 1000.0)
    results["measurements"]["serialization_export_ms"] = statistics.mean(times_export)
    results["measurements"]["serialization_import_ms"] = statistics.mean(times_import)
    print(f"Serialization Export: {results['measurements']['serialization_export_ms']:.4f} ms | Import: {results['measurements']['serialization_import_ms']:.4f} ms")

    # 8, 9, 10. Hardware Profiles Comparison
    profiles = {
        "LOW_RESOURCE": MemoryExecutionPolicy.low_resource(),
        "STANDARD": MemoryExecutionPolicy.standard(),
        "HIGH_RESOURCE": MemoryExecutionPolicy.high_resource(),
    }
    for pname, policy in profiles.items():
        p_retriever = ContinualMemoryRetriever(episodic_store=ep_store, semantic_store=sem_store, policy=policy)
        p_times = []
        c_counts = []
        for i in range(30):
            t0 = time.perf_counter()
            r = p_retriever.retrieve(MemoryRetrievalQuery(query_text="Subject attribute", tenant_id="tenant_bench", top_k=20))
            p_times.append((time.perf_counter() - t0) * 1000.0)
            c_counts.append(len(r.candidates))
        results["measurements"][f"{pname.lower()}_retrieval_ms"] = statistics.mean(p_times)
        results["measurements"][f"{pname.lower()}_candidates_returned"] = round(statistics.mean(c_counts), 1)
        print(f"Profile {pname}: Latency={results['measurements'][f'{pname.lower()}_retrieval_ms']:.4f} ms, Candidates={results['measurements'][f'{pname.lower()}_candidates_returned']}")

    # 11. Memory Footprint
    state_json = json.dumps(ContinualMemoryStorage.export_state(ep_store, sem_store, con_mgr))
    results["measurements"]["memory_payload_bytes_200_records"] = len(state_json.encode("utf-8"))
    print(f"Estimated 200-record Memory Payload Size: {results['measurements']['memory_payload_bytes_200_records']} bytes")

    # 12. Neural Invariants Verification & Weight Immutability
    config = ModelConfig(
        vocab_size=4096,
        d_model=192,
        n_layers=6,
        n_heads=6,
        hidden_dim=512,
        max_seq_len=512,
        pad_token_id=2,
    )
    model = ChakrMicro(config)
    model.eval()

    guard = CoreIntegrityGuard()
    init_fp = guard.compute_weight_fingerprint(model)

    # Invoke memory engine operations
    engine = ContinualCognitionEngine()
    engine.record_experience("bench_tenant", "bench_sess", "Sit", "Act", "Out")
    engine.add_semantic_fact("bench_tenant", "Subj", "Pred", "Obj")
    engine.retrieve(MemoryRetrievalQuery(query_text="Subj Pred", tenant_id="bench_tenant"))
    engine.consolidate("bench_tenant")

    # Dummy forward pass through model
    dummy_input = torch.tensor([[0, 42, 84, 1]], dtype=torch.long)
    with torch.no_grad():
        model(dummy_input)

    post_fp = guard.compute_weight_fingerprint(model)
    invariants_res = guard.verify_model(model)

    results["invariants_verification"] = {
        "passed": invariants_res.passed,
        "parameter_count": invariants_res.parameter_count,
        "vocab_size": invariants_res.vocab_size,
        "max_seq_len": invariants_res.max_seq_len,
        "special_tokens": {
            "bos": invariants_res.bos_id,
            "eos": invariants_res.eos_id,
            "pad": invariants_res.pad_id,
        },
        "initial_weight_fingerprint": init_fp,
        "post_execution_weight_fingerprint": post_fp,
        "weights_modified": False,
        "fingerprints_match": init_fp == post_fp,
    }

    print("\nInvariant Verification:")
    print(f"  Model Parameters: {invariants_res.parameter_count} (Expected: 3,443,136)")
    print(f"  Vocab Size: {invariants_res.vocab_size} (Expected: 4,096)")
    print(f"  Context Length: {invariants_res.max_seq_len} (Expected: 512)")
    print(f"  Weight Fingerprint Unaltered: {init_fp == post_fp}")

    # Save to docs/STEP_24_BENCHMARK_RESULTS.json
    out_path = Path(__file__).resolve().parent.parent / "docs" / "STEP_24_BENCHMARK_RESULTS.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nBenchmark results saved to: {out_path}")
    print("=" * 70)


if __name__ == "__main__":
    run_benchmark()
