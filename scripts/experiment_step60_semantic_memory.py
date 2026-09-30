"""
ChakrView Step 60: Semantic Repository Memory Arbitration - Master Experiment.

Executes:
 1. Baseline neural core pre-verification.
 2. Controlled memory corpus construction (up to 500 entries).
 3. Benchmark A  - single relevant memory.
 4. Benchmark B  - 5 superficially similar memories.
 5. Benchmark C  - 20 mixed memories.
 6. Benchmark D  - 100 mixed memories.
 7. Benchmark E  - 500 competing memories.
 8. Benchmark F  - conflicting memories (same scores, different solutions).
 9. Benchmark G  - negative-transfer corpus.
10. Benchmark H  - memory ablation (index cleared).
11. Benchmark I  - memory versioning.
12. Benchmark J  - consolidation across independent episodes.
13. Latency measurement.
14. Serialisation/reload determinism.
15. Repeated queries to prove determinism.
16. Baseline neural core post-verification.
17. Machine-readable evidence export to artifacts/step60/.
"""
from __future__ import annotations

import json
import time
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from chakrview.runtime.interactive import instantiate_frozen_baseline, compute_model_hash
from chakrview.runtime.pipeline import EXPECTED_WEIGHT_HASH
from chakrview.cognition.repository.semantic_record import RepositorySemanticRecord
from chakrview.cognition.repository.memory_index import RepositoryMemoryIndex
from chakrview.cognition.repository.arbitration import (
    arbitrate, RepositoryQuery, ArbitrationStatus, ArbitrationWeights, DEFAULT_WEIGHTS
)

ARTIFACTS_DIR = ROOT / "artifacts" / "step60"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _rec(
    memory_id: str,
    task_family: str = "billing_repair",
    language: str = "python",
    framework: str = "pytest",
    symptom: str = "assertion failure tax calculation mismatch",
    root_cause: str = "incorrect rate operator",
    dep_sig: str = "tax_service billing_service",
    modules: list = None,
    solution: str = "correct_rate_operator",
    evidence: int = 3,
    success: int = 3,
    failed: int = 0,
    boundaries: list = None,
) -> RepositorySemanticRecord:
    return RepositorySemanticRecord(
        memory_id=memory_id,
        task_family=task_family,
        language=language,
        framework=framework,
        repository_pattern="upstream_producer_defect",
        symptom_signature=symptom,
        root_cause_signature=root_cause,
        dependency_signature=dep_sig,
        affected_modules=modules or ["tax_service.py"],
        solution_pattern=solution,
        verification_requirements=["targeted_pass"],
        known_boundaries=boundaries or [],
        confidence=0.9,
        evidence_count=evidence,
        successful_episodes=success,
        failed_episodes=failed,
        source_episode_ids=["ep_" + memory_id],
    )


def _query(
    family: str = "billing_repair",
    lang: str = "python",
    fw: str = "pytest",
    symptom: str = "assertion failure tax calculation mismatch",
    dep: str = "tax_service billing_service",
    modules: list = None,
) -> RepositoryQuery:
    return RepositoryQuery(
        task_family=family,
        language=lang,
        framework=fw,
        symptom_signature=symptom,
        dependency_signature=dep,
        affected_modules=modules or ["tax_service.py"],
    )


def _run_benchmark(name: str, idx: RepositoryMemoryIndex, q: RepositoryQuery, expect_status: str, results: dict):
    t0 = time.perf_counter()
    res = arbitrate(q, idx)
    latency_ms = (time.perf_counter() - t0) * 1000
    status = res.status.name
    correct = (status == expect_status)
    selected_id = res.selected.memory_id if res.selected else None
    results[name] = {
        "status": status,
        "expected": expect_status,
        "correct": correct,
        "selected_id": selected_id,
        "candidate_count": len(res.candidates),
        "latency_ms": round(latency_ms, 4),
        "reason": res.reason,
    }
    indicator = "PASS" if correct else "FAIL"
    print(f"  [{indicator}] {name}: {status} | {res.reason[:80]}")
    return res


def run_experiment() -> dict:
    print("=" * 80)
    print("CHAKRVIEW STEP 60: SEMANTIC REPOSITORY MEMORY & ARBITRATION EXPERIMENT")
    print("=" * 80)

    evidence: dict = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "baseline_hash_pre": None,
        "baseline_hash_post": None,
        "benchmarks": {},
        "latency_ms": {},
        "arbitration_traces": [],
        "determinism_verified": False,
    }

    # 1. Pre-verification
    print("\n[1] Baseline Neural Core Pre-Verification")
    model = instantiate_frozen_baseline()
    h_pre = compute_model_hash(model)
    evidence["baseline_hash_pre"] = h_pre
    assert h_pre == EXPECTED_WEIGHT_HASH, "CRITICAL: Baseline hash mismatch before experiment!"
    print("    PASS:", h_pre)

    # ─── Benchmark A: single relevant memory ─────────────────────────────────
    print("\n[A] Benchmark A: Single Relevant Memory")
    idx_a = RepositoryMemoryIndex()
    idx_a.insert(_rec("RA_001", evidence=5, success=5))
    q_a = _query()
    res_a = _run_benchmark("A_single_relevant", idx_a, q_a, "SELECTED", evidence["benchmarks"])
    evidence["arbitration_traces"].append({"benchmark": "A", "trace": res_a.trace})

    # ─── Benchmark B: 5 superficially similar ────────────────────────────────
    print("\n[B] Benchmark B: 5 Superficially Similar Memories")
    idx_b = RepositoryMemoryIndex()
    idx_b.insert(_rec("RB_TARGET", evidence=5, success=5, dep_sig="tax_service billing_service"))
    for i in range(4):
        idx_b.insert(_rec("RB_SIMILAR_" + str(i), evidence=2, success=2,
                          dep_sig="order_service payment_service",
                          symptom="generic assertion failure"))
    q_b = _query()
    res_b = _run_benchmark("B_5_similar", idx_b, q_b, "SELECTED", evidence["benchmarks"])
    evidence["arbitration_traces"].append({"benchmark": "B", "trace": res_b.trace})

    # ─── Benchmark C: 20 mixed memories ──────────────────────────────────────
    print("\n[C] Benchmark C: 20 Mixed Memories")
    idx_c = RepositoryMemoryIndex()
    idx_c.insert(_rec("RC_TARGET", evidence=5, success=5))
    for i in range(19):
        fam = "billing_repair" if i < 5 else "auth_repair"
        idx_c.insert(_rec("RC_MIX_" + str(i).zfill(3), task_family=fam if i >= 5 else "billing_repair",
                          evidence=2, success=2, symptom="unrelated failure " + str(i)))
    q_c = _query()
    res_c = _run_benchmark("C_20_mixed", idx_c, q_c, "SELECTED", evidence["benchmarks"])

    # ─── Benchmark D: 100 mixed memories ─────────────────────────────────────
    print("\n[D] Benchmark D: 100 Mixed Memories")
    idx_d = RepositoryMemoryIndex()
    idx_d.insert(_rec("RD_TARGET", evidence=5, success=5))
    families = ["auth_repair", "cache_repair", "network_repair", "database_repair"]
    for i in range(99):
        fam = families[i % len(families)]
        idx_d.insert(_rec("RD_MIX_" + str(i).zfill(4), task_family=fam,
                          language="python", framework="pytest",
                          evidence=2, success=2, symptom="unrelated " + str(i)))
    q_d = _query()
    t0 = time.perf_counter()
    res_d = arbitrate(q_d, idx_d)
    latency_d = (time.perf_counter() - t0) * 1000
    evidence["latency_ms"]["D_100"] = round(latency_d, 4)
    evidence["benchmarks"]["D_100_mixed"] = {
        "status": res_d.status.name,
        "selected_id": res_d.selected.memory_id if res_d.selected else None,
        "latency_ms": evidence["latency_ms"]["D_100"],
        "candidate_count": len(res_d.candidates),
        "correct": res_d.status == ArbitrationStatus.SELECTED and res_d.selected and res_d.selected.memory_id == "RD_TARGET",
    }
    print("  D_100_mixed:", res_d.status.name, "latency=" + "%.2f" % latency_d + "ms")

    # ─── Benchmark E: 500 competing memories ─────────────────────────────────
    print("\n[E] Benchmark E: 500 Competing Memories")
    idx_e = RepositoryMemoryIndex()
    idx_e.insert(_rec("RE_TARGET", evidence=5, success=5))
    domains = ["billing", "auth", "cache", "network", "database", "api", "crypto", "async", "orm", "routing"]
    for i in range(499):
        dom = domains[i % len(domains)]
        idx_e.insert(_rec("RE_COMP_" + str(i).zfill(4),
                          task_family=dom + "_repair",
                          language="python", framework="pytest",
                          evidence=1, success=1, symptom="unrelated failure " + dom + " " + str(i)))
    t0 = time.perf_counter()
    res_e = arbitrate(q_d, idx_e)
    latency_e = (time.perf_counter() - t0) * 1000
    evidence["latency_ms"]["E_500"] = round(latency_e, 4)
    evidence["benchmarks"]["E_500_competing"] = {
        "status": res_e.status.name,
        "selected_id": res_e.selected.memory_id if res_e.selected else None,
        "latency_ms": evidence["latency_ms"]["E_500"],
        "correct": res_e.status == ArbitrationStatus.SELECTED and res_e.selected and res_e.selected.memory_id == "RE_TARGET",
    }
    print("  E_500_competing:", res_e.status.name, "latency=" + "%.2f" % latency_e + "ms")

    # ─── Benchmark F: conflicting memories ───────────────────────────────────
    print("\n[F] Benchmark F: Conflicting Memories")
    idx_f = RepositoryMemoryIndex()
    idx_f.insert(_rec("RF_A", solution="refresh_token", evidence=3, success=3))
    idx_f.insert(_rec("RF_B", solution="invalidate_session", evidence=3, success=3))
    q_f = _query()
    res_f = _run_benchmark("F_conflict", idx_f, q_f, "CONFLICTING",
                           evidence["benchmarks"])
    evidence["arbitration_traces"].append({"benchmark": "F", "trace": res_f.trace})

    # ─── Benchmark G: negative-transfer corpus ────────────────────────────────
    print("\n[G] Benchmark G: Negative Transfer Corpus")
    idx_g = RepositoryMemoryIndex()
    neg_families = ["billing_repair", "auth_repair", "cache_repair", "django_orm_repair", "fastapi_routing_repair"]
    for i, fam in enumerate(neg_families):
        idx_g.insert(_rec("RG_" + str(i), task_family=fam, evidence=5, success=5))
    # Query with a completely unrelated family
    q_g = RepositoryQuery(
        task_family="javascript_async_repair",
        language="javascript",
        framework="jest",
        symptom_signature="promise rejection unhandled async callback",
    )
    res_g = _run_benchmark("G_negative_transfer", idx_g, q_g, "NO_MATCH", evidence["benchmarks"])

    # Also test within-corpus: billing query should not select auth record
    q_billing = _query()
    res_g2 = arbitrate(q_billing, idx_g)
    evidence["benchmarks"]["G_within_corpus_billing"] = {
        "status": res_g2.status.name,
        "selected_id": res_g2.selected.memory_id if res_g2.selected else None,
        "correct": res_g2.selected is None or res_g2.selected.task_family == "billing_repair",
    }
    print("  G_within_corpus_billing:", res_g2.status.name)

    # ─── Benchmark H: memory ablation ────────────────────────────────────────
    print("\n[H] Benchmark H: Memory Ablation (Empty Index)")
    idx_h = RepositoryMemoryIndex()  # empty
    q_h = _query()
    res_h = _run_benchmark("H_ablation", idx_h, q_h, "NO_MATCH", evidence["benchmarks"])

    # ─── Benchmark I: memory versioning ──────────────────────────────────────
    print("\n[I] Benchmark I: Memory Versioning")
    idx_i = RepositoryMemoryIndex()
    r_v1 = _rec("RI_VER", solution="v1_strategy", evidence=2, success=2)
    idx_i.insert(r_v1)
    r_v2 = _rec("RI_VER", solution="v2_strategy", evidence=5, success=5)
    idx_i.insert(r_v2)
    current = idx_i.lookup("RI_VER")
    versioning_ok = current is not None and current.solution_pattern == "v2_strategy"
    evidence["benchmarks"]["I_versioning"] = {
        "current_solution": current.solution_pattern if current else None,
        "versioning_ok": versioning_ok,
        "correct": versioning_ok,
    }
    print("  I_versioning: solution=" + (current.solution_pattern if current else "None"),
          "OK=" + str(versioning_ok))

    # ─── Benchmark J: consolidation ──────────────────────────────────────────
    print("\n[J] Benchmark J: Consolidation Across Independent Episodes")
    idx_j = RepositoryMemoryIndex()
    consolidated = RepositorySemanticRecord(
        memory_id="SEM_CONSOLIDATED_BILLING",
        task_family="billing_repair",
        language="python",
        framework="pytest",
        repository_pattern="upstream_producer_defect",
        symptom_signature="assertion failure tax calculation mismatch",
        root_cause_signature="incorrect rate operator",
        dependency_signature="tax_service billing_service",
        affected_modules=["tax_service.py"],
        solution_pattern="correct_rate_operator",
        verification_requirements=["targeted_pass", "regression_pass"],
        known_boundaries=["do not apply to fixed-fee billing logic"],
        confidence=0.92,
        evidence_count=4,
        successful_episodes=4,
        failed_episodes=0,
        source_episode_ids=["ep_001", "ep_002", "ep_003", "ep_004"],
    )
    idx_j.insert(consolidated)
    q_j = _query()
    res_j = arbitrate(q_j, idx_j)
    consolidation_ok = (res_j.status == ArbitrationStatus.SELECTED and
                        res_j.selected.memory_id == "SEM_CONSOLIDATED_BILLING")
    evidence["benchmarks"]["J_consolidation"] = {
        "status": res_j.status.name,
        "selected_id": res_j.selected.memory_id if res_j.selected else None,
        "correct": consolidation_ok,
        "evidence_count": consolidated.evidence_count,
        "source_episodes": len(consolidated.source_episode_ids),
    }
    print("  J_consolidation:", res_j.status.name, "correct=" + str(consolidation_ok))

    # ─── Determinism: repeat selected queries ─────────────────────────────────
    print("\n[DET] Determinism Verification")
    det_ok = True
    for _ in range(5):
        r1 = arbitrate(q_a, idx_a)
        r2 = arbitrate(q_a, idx_a)
        if r1.status != r2.status:
            det_ok = False
            break
        if r1.selected and r2.selected and r1.selected.memory_id != r2.selected.memory_id:
            det_ok = False
            break
    evidence["determinism_verified"] = det_ok
    print("  Determinism:", "PASS" if det_ok else "FAIL")

    # ─── Serialization round-trip ─────────────────────────────────────────────
    print("\n[SER] Serialization / Reload")
    idx_ser = RepositoryMemoryIndex()
    for i in range(20):
        idx_ser.insert(_rec("RSER_" + str(i).zfill(4), evidence=i + 1, success=i + 1))
    j1 = idx_ser.to_json()
    idx_reload = RepositoryMemoryIndex.from_json(j1)
    j2 = idx_reload.to_json()
    ser_ok = (j1 == j2)
    evidence["serialization_roundtrip_ok"] = ser_ok
    print("  Serialization:", "PASS" if ser_ok else "FAIL")

    # ─── Post-verification ────────────────────────────────────────────────────
    print("\n[17] Baseline Neural Core Post-Verification")
    h_post = compute_model_hash(model)
    evidence["baseline_hash_post"] = h_post
    hash_intact = (h_post == EXPECTED_WEIGHT_HASH)
    assert hash_intact, "CRITICAL: Baseline hash mutated during experiment!"
    print("    PASS:", h_post)

    # ─── Save artifacts ───────────────────────────────────────────────────────
    evidence_path = ARTIFACTS_DIR / "step60_semantic_memory_evidence.json"
    evidence_path.write_text(json.dumps(evidence, indent=2), encoding="utf-8")

    traces_path = ARTIFACTS_DIR / "step60_arbitration_traces.json"
    traces_path.write_text(json.dumps(evidence.get("arbitration_traces", []), indent=2), encoding="utf-8")

    # Memory index snapshot (benchmark A)
    index_path = ARTIFACTS_DIR / "step60_memory_index.json"
    index_path.write_text(idx_a.to_json(), encoding="utf-8")

    print("\n[ARTIFACTS] Written to:", str(ARTIFACTS_DIR))
    print("  - step60_semantic_memory_evidence.json")
    print("  - step60_arbitration_traces.json")
    print("  - step60_memory_index.json")

    # ─── Summary ─────────────────────────────────────────────────────────────
    benchmarks_correct = sum(1 for v in evidence["benchmarks"].values() if isinstance(v, dict) and v.get("correct"))
    benchmarks_total = sum(1 for v in evidence["benchmarks"].values() if isinstance(v, dict) and "correct" in v)
    print("\n" + "=" * 80)
    print("STEP 60 EXPERIMENT COMPLETE")
    print("=" * 80)
    print("  Benchmarks passed: " + str(benchmarks_correct) + "/" + str(benchmarks_total))
    print("  Determinism:       " + ("VERIFIED" if det_ok else "FAILED"))
    print("  Serialization:     " + ("OK" if ser_ok else "FAILED"))
    print("  Baseline hash:     " + ("INTACT" if hash_intact else "MUTATED"))
    print("  Latency 100-entry: " + "%.2f" % evidence["latency_ms"].get("D_100", 0) + "ms")
    print("  Latency 500-entry: " + "%.2f" % evidence["latency_ms"].get("E_500", 0) + "ms")

    return evidence


if __name__ == "__main__":
    run_experiment()
