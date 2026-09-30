"""
Step 43 Benchmark: Persistent Cognitive Memory, Knowledge Retrieval & Adaptive Planning Integration.

Measures:
1. Memory write latency (EpisodicMemoryStore.record_episode & SemanticMemoryStore.add_memory)
2. Memory retrieval latency (ContinualMemoryRetriever across semantic + episodic candidates)
3. Context assembly latency (PersistentCognitiveMemoryAdapter.retrieve_context + Envelope population)
4. RAG invocation overhead (GovernedKnowledgeRetrievalCapability BM25 query + evidence packaging)
5. Adaptive planning latency (AdaptiveTaskPlanner & DeterministicWorkloadClassifier vs standard graph)
6. Episode consolidation latency (consolidate_episode with episodic + semantic candidate creation)
7. State persistence overhead (ContinualMemoryStorage export + atomic JSON save to disk)
8. End-to-end cognitive episode latency (adaptive plan + memory retrieval + RAG + synthesis + consolidation)

Output: docs/STEP_43_BENCHMARK_RESULTS.json
"""

import json
from pathlib import Path
import statistics
import sys
import tempfile
import time
from typing import Any, Callable, Dict, List
import uuid

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chakrview.cognition.orchestration.models import WorkloadClass
from chakrview.cognition.federation.cognitive.engine import FederatedCognitiveEngine
from chakrview.cognition.federation.cognitive.memory import PersistentCognitiveMemoryAdapter
from chakrview.cognition.federation.cognitive.capabilities import GovernedKnowledgeRetrievalCapability
from chakrview.capability.contract import CapabilityRequest
from chakrview.memory.episodic import EpisodicMemoryStore
from chakrview.memory.semantic import SemanticMemoryStore
from chakrview.memory.models import (
    MemoryVerificationState,
    MemoryRetrievalQuery,
)
from chakrview.memory.storage import ContinualMemoryStorage


def _now_ms() -> float:
    return time.perf_counter() * 1000.0


def _bench(label: str, fn: Callable[[], None], n: int = 500) -> Dict[str, Any]:
    times: List[float] = []
    # Warmup
    for _ in range(min(5, n)):
        fn()
    for _ in range(n):
        t0 = _now_ms()
        fn()
        times.append(_now_ms() - t0)

    res = {
        "label": label,
        "iterations": n,
        "total_ms": round(sum(times), 2),
        "mean_ms": round(statistics.mean(times), 4),
        "median_ms": round(statistics.median(times), 4),
        "stdev_ms": round(statistics.stdev(times) if len(times) > 1 else 0.0, 4),
        "min_ms": round(min(times), 4),
        "max_ms": round(max(times), 4),
        "throughput_per_sec": round(1000.0 / statistics.mean(times), 1) if statistics.mean(times) > 0 else 0.0,
    }
    print(
        f"  [{label:<35}] mean={res['mean_ms']:>8.4f}ms  "
        f"median={res['median_ms']:>8.4f}ms  "
        f"throughput={res['throughput_per_sec']:>10.1f}/s  "
        f"(N={n})"
    )
    return res


def run_all_benchmarks() -> Dict[str, Any]:
    print("=" * 80)
    print("ChakrView Step 43: Persistent Cognitive Memory, RAG & Planning Benchmark")
    print("=" * 80)

    results: List[Dict[str, Any]] = []

    # 1. Memory Write Latency
    ep_store = EpisodicMemoryStore()
    sem_store = SemanticMemoryStore()
    i = [0]

    def bench_mem_write():
        i[0] += 1
        ep_store.record_episode(
            tenant_id="bench_tenant",
            session_id="bench_session",
            situation=f"Benchmark episode situation {i[0]}",
            action_or_response=f"Response {i[0]} with synthesized conclusion",
            outcome="committed",
            confidence=0.85,
        )
        sem_store.add_memory(
            tenant_id="bench_tenant",
            subject=f"Concept_{i[0]}",
            predicate="has_property",
            object_value=f"Value_{i[0]}",
            verification_status=MemoryVerificationState.VERIFIED,
            confidence=0.9,
        )

    results.append(_bench("memory_write_episodic_and_semantic", bench_mem_write, n=500))

    # 2. Memory Retrieval Latency
    adapter = PersistentCognitiveMemoryAdapter(episodic_store=ep_store, semantic_store=sem_store)

    def bench_mem_retrieve():
        adapter.retriever.retrieve(
            MemoryRetrievalQuery(
                query_text="Concept Benchmark situation property",
                tenant_id="bench_tenant",
                top_k=5,
                trusted_only=True,
            )
        )

    results.append(_bench("memory_retrieval_scoring", bench_mem_retrieve, n=500))

    # 3. Context Assembly Latency
    def bench_context_assembly():
        adapter.retrieve_context(
            objective="Analyze Concept Benchmark situation property",
            tenant_id="bench_tenant",
            top_k=5,
        )

    results.append(_bench("context_assembly_adapter", bench_context_assembly, n=500))

    # 4. RAG / BM25 Knowledge Retrieval Overhead
    rag_cap = GovernedKnowledgeRetrievalCapability()
    for d in range(20):
        rag_cap.ingest_text(
            text=f"Document {d} detailing architectural pattern for ChakrMicro transformer blocks and SwiGLU.",
            title=f"Doc {d}",
            doc_id=f"doc_{d}",
        )

    req = CapabilityRequest(
        capability_id=rag_cap.descriptor.capability_id,
        parameters={"query": "architectural pattern transformer SwiGLU", "top_k": 3},
    )

    def bench_rag_retrieval():
        rag_cap.execute(req)

    results.append(_bench("rag_bm25_retrieval_overhead", bench_rag_retrieval, n=500))

    # 5. Adaptive Planning Overhead
    engine = FederatedCognitiveEngine(memory_adapter=adapter, knowledge_capability=rag_cap)
    plan_idx = [0]

    def bench_adaptive_planning():
        plan_idx[0] += 1
        engine.plan_adaptive_episode(
            objective=f"Analyze performance scaling curve for transformer layer {plan_idx[0]}",
            tenant_id="bench_tenant",
            session_id="bench_session",
            workload_class=WorkloadClass.STANDARD,
        )

    results.append(_bench("adaptive_planning_standard_dag", bench_adaptive_planning, n=300))

    # 6. Episode Memory Consolidation
    episodes_to_consolidate = [
        engine.plan_adaptive_episode(
            objective=f"Consolidation benchmark task {k}",
            tenant_id="bench_tenant",
            session_id="bench_session",
            workload_class=WorkloadClass.SIMPLE,
        )
        for k in range(50)
    ]
    for ep in episodes_to_consolidate:
        engine.execute_episode(ep.episode_id)

    cons_idx = [0]

    def bench_consolidation():
        ep = episodes_to_consolidate[cons_idx[0] % len(episodes_to_consolidate)]
        cons_idx[0] += 1
        adapter.consolidate_episode(ep, extract_semantic=True)

    results.append(_bench("episode_memory_consolidation", bench_consolidation, n=300))

    # 7. State Persistence Overhead (Atomic Disk Export/Import)
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp_file:
        tmp_path = tmp_file.name

    def bench_persistence_save():
        adapter.save_to_storage(tmp_path)

    results.append(_bench("persistence_atomic_disk_save", bench_persistence_save, n=100))

    def bench_persistence_load():
        adapter.load_from_storage(tmp_path)

    results.append(_bench("persistence_disk_load_validate", bench_persistence_load, n=100))

    Path(tmp_path).unlink(missing_ok=True)

    # 8. End-to-End Cognitive Episode Latency
    e2e_engine = FederatedCognitiveEngine(memory_adapter=adapter, knowledge_capability=rag_cap)
    e2e_idx = [0]

    def bench_end_to_end_episode():
        e2e_idx[0] += 1
        ep = e2e_engine.plan_adaptive_episode(
            objective=f"Evaluate SwiGLU feed-forward layer efficiency test {e2e_idx[0]}",
            tenant_id="bench_tenant",
            session_id="bench_session",
            workload_class=WorkloadClass.STANDARD,
        )
        e2e_engine.execute_episode(ep.episode_id)

    results.append(_bench("end_to_end_cognitive_episode", bench_end_to_end_episode, n=100))

    benchmark_data = {
        "benchmark_suite": "Step 43: Persistent Cognitive Memory, RAG & Adaptive Planning",
        "timestamp": time.time(),
        "environment": {
            "mode": "in-process loopback simulation (CPU-only)",
            "cpu_only": True,
            "gpu_required": False,
            "neural_model": "ChakrMicro v0.1 (3,443,136 parameters, ΔW = 0)",
            "note": "In-process microbenchmark results. Real network deployments will exhibit additional transport latency.",
        },
        "benchmarks": {b["label"]: b for b in results},
    }

    output_path = Path("docs/STEP_43_BENCHMARK_RESULTS.json")
    output_path.write_text(json.dumps(benchmark_data, indent=2), encoding="utf-8")
    print("=" * 80)
    print(f"Results written to: {output_path}")
    print("=" * 80)
    return benchmark_data


if __name__ == "__main__":
    run_all_benchmarks()
