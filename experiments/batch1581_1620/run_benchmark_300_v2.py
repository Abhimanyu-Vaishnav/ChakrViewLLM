"""
Benchmark Runner for Batches 1581–1620 Frozen 300-Task Suite (v2).
Location: experiments/batch1581_1620/run_benchmark_300_v2.py
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chakrview.runtime.safe_calculator import SafeCalculator
from chakrview.runtime.structured_verifier import (
    StructuredVerifier,
    StructuredClaim,
    VerificationVerdict,
)
from chakrview.runtime.conversation_state import (
    ConversationStateManager,
    ToolResultType,
)
from chakrview.runtime.sandboxed_python_tool import SandboxedPythonTool
from chakrview.runtime.oci_container_backend import (
    OciContainerBackend,
    ContainerExecutionRequest,
    ContainerExecutionStatus,
)
from chakrview.runtime.governed_tool_router import (
    GovernedToolRouter,
    CanonicalRoute,
)
from chakrview.runtime.local_knowledge_assistant import LocalKnowledgeAssistant

DATA_PATH = Path("experiments/batch1581_1620/data/evaluation_suite_300_v2.json")
CORPUS_DIR = Path("experiments/batch1141_1180/data/corpus")
OUT_DIR = Path("experiments/batch1581_1620/results")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_RESULTS = OUT_DIR / "BENCHMARK_RESULTS_300_V2.json"


def run_benchmark():
    print(f"Loading 300-task evaluation suite from {DATA_PATH}...")
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        suite = json.load(f)

    # Initialize Components
    calc = SafeCalculator()
    state_mgr = ConversationStateManager()
    sandbox = SandboxedPythonTool()
    container_backend = OciContainerBackend()

    print(f"Initializing LocalKnowledgeAssistant from {CORPUS_DIR}...")
    asst = LocalKnowledgeAssistant(extract_concise_span=True)
    asst.ingest_directory(CORPUS_DIR)

    verifier = StructuredVerifier(chunks=asst.chunks)
    router = GovernedToolRouter(
        calculator=calc,
        verifier=verifier,
        knowledge_assistant=asst,
        state_manager=state_mgr,
        container_backend=container_backend,
    )

    results = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "summary": {},
        "track1_container_policy": {},
        "track2_resource_limits": {},
        "track3_dependency_verification": {},
        "track4_multi_turn_state": {},
        "track5_end_to_end_routing": {},
    }

    # ========================================================
    # TRACK 1: CONTAINER POLICY & ISOLATION (60 Tasks)
    # ========================================================
    print("Executing Track 1: Container Policy & Isolation (60 tasks)...")
    t1_tasks = [t for t in suite if t["track"] == "container_policy"]
    t1_passed = 0
    t1_results = []

    for t in t1_tasks:
        tid = t["id"]
        req = ContainerExecutionRequest(
            task_id=t.get("task_id", tid),
            code=t["code"],
            image=t.get("image", "python:3.11-alpine"),
            network_enabled=t.get("network_enabled", False),
            mount_paths=t.get("mount_paths", []),
        )
        c_res = container_backend.execute(req)
        exp_status = t["expected_status"]
        is_pass = (c_res.status.value == exp_status)

        if is_pass:
            t1_passed += 1

        t1_results.append({
            "id": tid,
            "category": t["category"],
            "expected_status": exp_status,
            "actual_status": c_res.status.value,
            "passed": is_pass,
            "error": c_res.error,
            "latency_ms": c_res.execution_time_ms,
        })

    t1_acc = round(t1_passed / len(t1_tasks), 4) if t1_tasks else 0.0
    results["track1_container_policy"] = {
        "total_tasks": len(t1_tasks),
        "passed": t1_passed,
        "accuracy": t1_acc,
        "results": t1_results,
    }
    print(f"Track 1 Complete: {t1_passed}/{len(t1_tasks)} ({t1_acc * 100:.1f}%)")

    # ========================================================
    # TRACK 2: RESOURCE LIMITS & FAILURE RECOVERY (60 Tasks)
    # ========================================================
    print("Executing Track 2: Resource Limits & Failure Recovery (60 tasks)...")
    t2_tasks = [t for t in suite if t["track"] == "resource_limits"]
    t2_passed = 0
    t2_results = []

    for t in t2_tasks:
        tid = t["id"]
        cat = t["category"]
        exp_status = t["expected_status"]

        if cat == "invalid_memory_limit":
            req = ContainerExecutionRequest(
                task_id=t.get("task_id", tid),
                code=t["code"],
                memory_limit=t["memory_limit"],
            )
            c_res = container_backend.execute(req)
            is_pass = (c_res.status.value == exp_status)
            lat = c_res.execution_time_ms
            err = c_res.error
        else:
            # Subprocess sandbox fallback timeout / buffer capping
            timeout = t.get("timeout", 2.0)
            s_res = sandbox.execute_script(code=t["code"], timeout=timeout, execution_id=f"run_{tid}")
            is_pass = (s_res.status == exp_status)
            lat = s_res.execution_time_ms
            err = s_res.error

        if is_pass:
            t2_passed += 1

        t2_results.append({
            "id": tid,
            "category": cat,
            "expected_status": exp_status,
            "passed": is_pass,
            "error": err,
            "latency_ms": lat,
        })

    t2_acc = round(t2_passed / len(t2_tasks), 4) if t2_tasks else 0.0
    results["track2_resource_limits"] = {
        "total_tasks": len(t2_tasks),
        "passed": t2_passed,
        "accuracy": t2_acc,
        "results": t2_results,
    }
    print(f"Track 2 Complete: {t2_passed}/{len(t2_tasks)} ({t2_acc * 100:.1f}%)")

    # ========================================================
    # TRACK 3: DEPENDENCY-AWARE STRUCTURED VERIFICATION (100 Tasks)
    # ========================================================
    print("Executing Track 3: Dependency-Aware Structured Verification (100 tasks)...")
    t3_tasks = [t for t in suite if t["track"] == "dependency_verification"]
    t3_passed = 0
    t3_results = []

    for t in t3_tasks:
        tid = t["id"]
        ent = t["subject_entity"]
        attr = t["attribute_name"]
        val = t["claimed_value"]
        exp_verdict = t["expected_verdict"]

        claim = StructuredClaim(
            claim_id=tid,
            subject_entity=ent,
            attribute_name=attr,
            claimed_value=val,
        )

        v_rec = verifier.verify_claim(claim)
        is_pass = (v_rec.verdict.value == exp_verdict)

        if is_pass:
            t3_passed += 1

        t3_results.append({
            "id": tid,
            "subject_entity": ent,
            "attribute_name": attr,
            "claimed_value": val,
            "expected_verdict": exp_verdict,
            "actual_verdict": v_rec.verdict.value,
            "evidence_value": v_rec.evidence_value,
            "passed": is_pass,
            "latency_ms": v_rec.latency_ms,
        })

    t3_acc = round(t3_passed / len(t3_tasks), 4) if t3_tasks else 0.0
    results["track3_dependency_verification"] = {
        "total_tasks": len(t3_tasks),
        "passed": t3_passed,
        "accuracy": t3_acc,
        "results": t3_results,
    }
    print(f"Track 3 Complete: {t3_passed}/{len(t3_tasks)} ({t3_acc * 100:.1f}%)")

    # ========================================================
    # TRACK 4: MULTI-TURN STATE & ROUTING (40 Tasks)
    # ========================================================
    print("Executing Track 4: Multi-Turn State & Routing (40 tasks)...")
    t4_tasks = [t for t in suite if t["track"] == "multi_turn_state"]
    t4_passed = 0
    t4_results = []

    for t in t4_tasks:
        tid = t["id"]
        cat = t["category"]
        e1 = t["turn1_expression"]
        r1 = t["turn1_result"]
        t1_tenant = t.get("turn1_tenant", "tenant_a")
        cid = f"conv_t4_{tid}"

        # Turn 1
        state_mgr.record_tool_result(
            conversation_id=cid,
            tool_id="calculator",
            operation="evaluate",
            arguments={"expression": e1},
            result=r1,
            result_type=ToolResultType.NUMERIC,
            tenant_id=t1_tenant,
        )

        if t.get("do_reset"):
            state_mgr.reset_session(cid, tenant_id=t1_tenant)

        t2_tenant = t.get("turn2_tenant", t1_tenant)
        q2 = t["turn2_query"]
        res_ref = state_mgr.resolve_reference(q2, conversation_id=cid, tenant_id=t2_tenant)

        is_pass = (res_ref.resolved == t["should_resolve"])
        if is_pass and t["should_resolve"]:
            # Evaluate expression
            calc_res = calc.evaluate(res_ref.resolved_expression)
            is_pass = (calc_res.success and calc_res.result == t.get("expected_result"))

        if is_pass:
            t4_passed += 1

        t4_results.append({
            "id": tid,
            "category": cat,
            "passed": is_pass,
            "resolved": res_ref.resolved,
            "should_resolve": t["should_resolve"],
        })

    t4_acc = round(t4_passed / len(t4_tasks), 4) if t4_tasks else 0.0
    results["track4_multi_turn_state"] = {
        "total_tasks": len(t4_tasks),
        "passed": t4_passed,
        "accuracy": t4_acc,
        "results": t4_results,
    }
    print(f"Track 4 Complete: {t4_passed}/{len(t4_tasks)} ({t4_acc * 100:.1f}%)")

    # ========================================================
    # TRACK 5: END-TO-END GOVERNED ROUTING (40 Tasks)
    # ========================================================
    print("Executing Track 5: End-to-End Governed Routing (40 tasks)...")
    t5_tasks = [t for t in suite if t["track"] == "end_to_end_routing"]
    t5_passed = 0
    t5_results = []

    for t in t5_tasks:
        tid = t["id"]
        q = t["query"]
        exp_route = t["expected_route"]

        resp = router.route_request(q, request_id=f"req_{tid}")
        is_pass = (resp.route.value == exp_route)

        if is_pass:
            t5_passed += 1

        t5_results.append({
            "id": tid,
            "query": q,
            "expected_route": exp_route,
            "actual_route": resp.route.value,
            "passed": is_pass,
            "latency_ms": resp.latency_ms,
        })

    t5_acc = round(t5_passed / len(t5_tasks), 4) if t5_tasks else 0.0
    results["track5_end_to_end_routing"] = {
        "total_tasks": len(t5_tasks),
        "passed": t5_passed,
        "accuracy": t5_acc,
        "results": t5_results,
    }
    print(f"Track 5 Complete: {t5_passed}/{len(t5_tasks)} ({t5_acc * 100:.1f}%)")

    # Overall Summary
    total_passed = t1_passed + t2_passed + t3_passed + t4_passed + t5_passed
    results["summary"] = {
        "track1_container_policy_accuracy": t1_acc,
        "track2_resource_limits_accuracy": t2_acc,
        "track3_dependency_verification_accuracy": t3_acc,
        "track4_multi_turn_state_accuracy": t4_acc,
        "track5_end_to_end_routing_accuracy": t5_acc,
        "total_tasks_evaluated": len(suite),
        "total_tasks_passed": total_passed,
        "overall_success_rate": round(total_passed / len(suite), 4),
    }

    with open(OUT_RESULTS, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nBenchmark Complete! Results saved to {OUT_RESULTS}")
    print(f"Overall Success Rate: {results['summary']['overall_success_rate'] * 100:.2f}% ({total_passed}/{len(suite)})")


if __name__ == "__main__":
    run_benchmark()
