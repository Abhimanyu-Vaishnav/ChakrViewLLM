"""
Comprehensive Benchmark for Persistent Personal Memory Subsystem (Step 16).

Measures on CPU:
- Memory insertion latency
- Exact retrieval latency
- Semantic retrieval latency
- Hybrid retrieval latency
- Multi-tier deduplication latency
- Memory consolidation overhead
- Serialization & deserialization throughput
- Scaling benchmarks across scales: N = 100, 1,000, 10,000 records
"""

import json
import os
from pathlib import Path
import random
import statistics
import sys
import time
from typing import Dict, List, Any

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from chakrview.memory.record import MemoryRecord, MemoryType, MemoryProvenance
from chakrview.memory.store import InMemoryMemoryStore
from chakrview.memory.deduplication import MemoryDeduplicator
from chakrview.memory.consolidation import MemoryConsolidator
from chakrview.memory.retriever import PersistentMemoryRetriever
from chakrview.memory.manager import PersonalMemoryManager


def benchmark_insertion(store: InMemoryMemoryStore, n: int = 500) -> Dict[str, float]:
    times = []
    for i in range(n):
        rec = MemoryRecord(
            memory_id=f"bench_insert_{i}",
            memory_type=MemoryType.SEMANTIC,
            content=f"Synthetic business record {i}: client turnover is {i * 10} lakh in region {i % 5}.",
            owner_id="bench_user",
        )
        t0 = time.perf_counter()
        store.add(rec)
        times.append((time.perf_counter() - t0) * 1e6)  # microseconds

    return {
        "mean_us": statistics.mean(times),
        "median_us": statistics.median(times),
        "stdev_us": statistics.stdev(times),
        "min_us": min(times),
        "max_us": max(times),
    }


def benchmark_exact_retrieval(store: InMemoryMemoryStore, iterations: int = 500) -> Dict[str, float]:
    times = []
    for i in range(iterations):
        target_id = f"bench_insert_{i % 500}"
        t0 = time.perf_counter()
        rec = store.get(target_id, owner_id="bench_user")
        times.append((time.perf_counter() - t0) * 1e6)  # microseconds

    return {
        "mean_us": statistics.mean(times),
        "median_us": statistics.median(times),
        "stdev_us": statistics.stdev(times),
        "min_us": min(times),
        "max_us": max(times),
    }


def benchmark_hybrid_retrieval(retriever: PersistentMemoryRetriever, store: InMemoryMemoryStore, iterations: int = 100) -> Dict[str, float]:
    queries = [
        "client turnover region 2",
        "business record manufacturing",
        "turnover lakh Gujarat",
        "synthetic financial audit",
    ]
    times = []
    for i in range(iterations):
        q = queries[i % len(queries)]
        t0 = time.perf_counter()
        results = retriever.retrieve(q, owner_id="bench_user", store=store, top_k=5)
        times.append((time.perf_counter() - t0) * 1e3)  # milliseconds

    return {
        "mean_ms": statistics.mean(times),
        "median_ms": statistics.median(times),
        "stdev_ms": statistics.stdev(times),
        "min_ms": min(times),
        "max_ms": max(times),
    }


def benchmark_deduplication(deduplicator: MemoryDeduplicator, existing: List[MemoryRecord], iterations: int = 200) -> Dict[str, float]:
    times = []
    for i in range(iterations):
        # Alternate between duplicate and novel text
        text = existing[i % len(existing)].content if i % 2 == 0 else f"Novel unique statement {i}"
        t0 = time.perf_counter()
        res = deduplicator.check_duplicate(text, existing)
        times.append((time.perf_counter() - t0) * 1e6)  # microseconds

    return {
        "mean_us": statistics.mean(times),
        "median_us": statistics.median(times),
        "stdev_us": statistics.stdev(times),
        "min_us": min(times),
        "max_us": max(times),
    }


def benchmark_consolidation(consolidator: MemoryConsolidator, records: List[MemoryRecord], iterations: int = 50) -> Dict[str, float]:
    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        clusters = consolidator.cluster_memories(records[:100])
        for c in clusters:
            cand = consolidator.generate_candidate(c, owner_id="bench_user")
        times.append((time.perf_counter() - t0) * 1e3)  # milliseconds

    return {
        "mean_ms": statistics.mean(times),
        "median_ms": statistics.median(times),
        "stdev_ms": statistics.stdev(times),
        "min_ms": min(times),
        "max_ms": max(times),
    }


def benchmark_serialization(records: List[MemoryRecord], iterations: int = 500) -> Dict[str, float]:
    times_ser = []
    times_deser = []
    sample = records[:100]

    for i in range(iterations):
        rec = sample[i % len(sample)]
        # Serialization
        t0 = time.perf_counter()
        d = rec.to_dict()
        times_ser.append((time.perf_counter() - t0) * 1e6)

        # Deserialization
        t1 = time.perf_counter()
        restored = MemoryRecord.from_dict(d)
        times_deser.append((time.perf_counter() - t1) * 1e6)

    return {
        "serialization_mean_us": statistics.mean(times_ser),
        "serialization_median_us": statistics.median(times_ser),
        "deserialization_mean_us": statistics.mean(times_deser),
        "deserialization_median_us": statistics.median(times_deser),
    }


def benchmark_scaling(scales: List[int] = [100, 1000, 10000]) -> Dict[str, Any]:
    scaling_results = {}
    retriever = PersistentMemoryRetriever()

    for n in scales:
        store = InMemoryMemoryStore()
        # Populate N records
        for i in range(n):
            store.add(MemoryRecord(
                memory_id=f"scale_{n}_{i}",
                memory_type=MemoryType.SEMANTIC,
                content=f"Client entity {i}: GST number GSTIN{i:05d} operations in state {i % 10}",
                owner_id="scale_user",
            ))

        # Measure search latency at scale N
        times = []
        for _ in range(20):
            t0 = time.perf_counter()
            res = retriever.retrieve("GST operations state 5", owner_id="scale_user", store=store, top_k=5)
            times.append((time.perf_counter() - t0) * 1e3)  # ms

        scaling_results[str(n)] = {
            "record_count": n,
            "search_mean_ms": statistics.mean(times),
            "search_median_ms": statistics.median(times),
            "search_min_ms": min(times),
            "search_max_ms": max(times),
        }

    return scaling_results


def main():
    print("=" * 60)
    print("CHAKRVIEW STEP 16: PERSISTENT PERSONAL MEMORY BENCHMARK")
    print("Platform: CPU (Single Thread Execution Overhead)")
    print("=" * 60)

    store = InMemoryMemoryStore()

    print("\n1. Measuring Memory Insertion latency (500 records)...")
    ins_res = benchmark_insertion(store, n=500)
    print(f"   Insertion: {ins_res['mean_us']:.2f} µs (median: {ins_res['median_us']:.2f} µs)")

    print("\n2. Measuring Exact ID Lookup latency (500 lookups)...")
    get_res = benchmark_exact_retrieval(store)
    print(f"   Lookup: {get_res['mean_us']:.2f} µs (median: {get_res['median_us']:.2f} µs)")

    print("\n3. Measuring Multi-Tier Deduplication latency (200 checks)...")
    records = store.list_records(owner_id="bench_user", limit=500)
    dedup = MemoryDeduplicator()
    dedup_res = benchmark_deduplication(dedup, records)
    print(f"   Deduplication: {dedup_res['mean_us']:.2f} µs (median: {dedup_res['median_us']:.2f} µs)")

    print("\n4. Measuring Hybrid Memory Retrieval latency (100 queries)...")
    retriever = PersistentMemoryRetriever()
    ret_res = benchmark_hybrid_retrieval(retriever, store)
    print(f"   Hybrid Retrieval (N=500): {ret_res['mean_ms']:.3f} ms (median: {ret_res['median_ms']:.3f} ms)")

    print("\n5. Measuring Memory Consolidation Clustering & Synthesis...")
    consolidator = MemoryConsolidator()
    cons_res = benchmark_consolidation(consolidator, records)
    print(f"   Consolidation (100 records): {cons_res['mean_ms']:.3f} ms (median: {cons_res['median_ms']:.3f} ms)")

    print("\n6. Measuring Serialization / Deserialization...")
    ser_res = benchmark_serialization(records)
    print(f"   Serialization: {ser_res['serialization_mean_us']:.2f} µs (median: {ser_res['serialization_median_us']:.2f} µs)")
    print(f"   Deserialization: {ser_res['deserialization_mean_us']:.2f} µs (median: {ser_res['deserialization_median_us']:.2f} µs)")

    print("\n7. Measuring Scaling at N = [100, 1000, 10000] records...")
    scale_res = benchmark_scaling([100, 1000, 10000])
    for n, s in scale_res.items():
        print(f"   Scale N={n:>5}: Search Mean = {s['search_mean_ms']:.3f} ms (median: {s['search_median_ms']:.3f} ms)")

    results = {
        "step": 16,
        "name": "Persistent Personal Memory Subsystem Benchmark",
        "insertion": ins_res,
        "exact_lookup": get_res,
        "deduplication": dedup_res,
        "hybrid_retrieval": ret_res,
        "consolidation": cons_res,
        "serialization": ser_res,
        "scaling": scale_res,
        "timestamp": time.time(),
    }

    out_path = Path("docs") / "STEP_16_BENCHMARK_RESULTS.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n[OK] Benchmark results saved to: {out_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
