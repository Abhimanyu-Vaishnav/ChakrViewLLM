"""
Comprehensive 300-Task Benchmark Runner for Batches 1541–1580.
Location: experiments/batch1541_1580/run_benchmark_300.py

Executes:
1. Track 1: Multi-Turn Reference Resolution (80 tasks)
2. Track 2: Arithmetic Tool Chaining (60 tasks)
3. Track 3: Sandbox Execution & Isolation (60 tasks)
4. Track 4: Structured Proposition Verification (60 tasks)
5. Track 5: Routing & Recovery (40 tasks)

Saves machine-readable results to:
experiments/batch1541_1580/results/BENCHMARK_RESULTS_300.json
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
from chakrview.runtime.governed_tool_router import (
    GovernedToolRouter,
    CanonicalRoute,
)
from chakrview.runtime.local_knowledge_assistant import LocalKnowledgeAssistant

DATA_PATH = Path("experiments/batch1541_1580/data/evaluation_suite_300.json")
CORPUS_DIR = Path("experiments/batch1141_1180/data/corpus")
OUT_DIR = Path("experiments/batch1541_1580/results")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_RESULTS = OUT_DIR / "BENCHMARK_RESULTS_300.json"


def run_benchmark():
    print(f"Loading 300-task evaluation suite from {DATA_PATH}...")
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        suite = json.load(f)

    # Initialize Components
    calc = SafeCalculator()
    state_mgr = ConversationStateManager()
    sandbox = SandboxedPythonTool()

    print(f"Initializing LocalKnowledgeAssistant from {CORPUS_DIR}...")
    asst = LocalKnowledgeAssistant(extract_concise_span=True)
    asst.ingest_directory(CORPUS_DIR)

    verifier = StructuredVerifier(chunks=asst.chunks)
    router = GovernedToolRouter(
        calculator=calc,
        verifier=verifier,
        knowledge_assistant=asst,
        state_manager=state_mgr,
    )

    results = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "summary": {},
        "track1_multi_turn_reference": {},
        "track2_arithmetic_chaining": {},
        "track3_sandbox_execution": {},
        "track4_proposition_verification": {},
        "track5_routing_and_recovery": {},
    }

    # ========================================================
    # TRACK 1: MULTI-TURN REFERENCE RESOLUTION (80 Tasks)
    # ========================================================
    print("Executing Track 1: Multi-Turn Reference Resolution (80 tasks)...")
    t1_tasks = [t for t in suite if t["track"] == "multi_turn_reference"]
    t1_passed = 0
    t1_results = []

    for t in t1_tasks:
        tid = t["id"]
        cat = t["category"]
        e1 = t.get("turn1_expression")
        r1 = t.get("turn1_result")
        q2 = t["turn2_query"]
        exp_res = t.get("expected_result")
        should_resolve = t["should_resolve"]
        t_id = t.get("turn2_tenant", "default_tenant")
        cid = f"conv_t1_{tid}"

        # Setup Turn 1 if present
        if e1 and r1 is not None:
            t1_tenant = t.get("turn1_tenant", "default_tenant")
            state_mgr.record_tool_result(
                conversation_id=cid,
                tool_id="calculator",
                operation="evaluate",
                arguments={"expression": e1},
                result=r1,
                result_type=ToolResultType.NUMERIC,
                tenant_id=t1_tenant,
            )

        # Execute Turn 2
        resp = router.route_request(
            q2, request_id=f"req_{tid}", conversation_id=cid, tenant_id=t_id
        )

        is_pass = False
        if should_resolve:
            if resp.route == CanonicalRoute.CALCULATOR and resp.result_payload is not None:
                if exp_res is not None:
                    is_pass = (abs(float(resp.result_payload) - float(exp_res)) < 1e-4)
                else:
                    is_pass = True
        else:
            # Should NOT resolve (e.g. missing state or cross-tenant isolation)
            # If missing antecedent, router should fail reference or route to ABSTAIN/ERROR
            if resp.route != CanonicalRoute.CALCULATOR or resp.result_payload != exp_res:
                is_pass = True

        if is_pass:
            t1_passed += 1

        t1_results.append({
            "id": tid,
            "category": cat,
            "query": q2,
            "should_resolve": should_resolve,
            "actual_route": resp.route.value,
            "actual_result": resp.result_payload,
            "expected_result": exp_res,
            "passed": is_pass,
            "latency_ms": resp.latency_ms,
        })

    t1_acc = round(t1_passed / len(t1_tasks), 4) if t1_tasks else 0.0
    results["track1_multi_turn_reference"] = {
        "total_tasks": len(t1_tasks),
        "passed": t1_passed,
        "accuracy": t1_acc,
        "results": t1_results,
    }
    print(f"Track 1 Complete: {t1_passed}/{len(t1_tasks)} ({t1_acc * 100:.1f}%)")

    # ========================================================
    # TRACK 2: ARITHMETIC TOOL CHAINING (60 Tasks)
    # ========================================================
    print("Executing Track 2: Arithmetic Tool Chaining (60 tasks)...")
    t2_tasks = [t for t in suite if t["track"] == "arithmetic_chaining"]
    t2_passed = 0
    t2_results = []

    for t in t2_tasks:
        tid = t["id"]
        cat = t["category"]
        cid = f"conv_t2_{tid}"
        exp_succ = t["expected_success"]

        if cat == "multi_step_chain":
            # Step 1
            r1 = router.route_request(t["step1_expr"], conversation_id=cid)
            s1_ok = (r1.route == CanonicalRoute.CALCULATOR and r1.result_payload == t["step1_expected"])

            # Step 2
            r2 = router.route_request(t["step2_query"], conversation_id=cid)
            s2_ok = (r2.route == CanonicalRoute.CALCULATOR and r2.result_payload == t["step2_expected"])

            # Step 3
            r3 = router.route_request(t["step3_query"], conversation_id=cid)
            s3_ok = (r3.route == CanonicalRoute.CALCULATOR and r3.result_payload == t["step3_expected"])

            is_pass = (s1_ok and s2_ok and s3_ok)
            final_res = r3.result_payload

        else:
            # Chained domain error (e.g. divide that by 0)
            r1 = router.route_request(t["step1_expr"], conversation_id=cid)
            r2 = router.route_request(t["step2_query"], conversation_id=cid)
            is_pass = (r2.route == CanonicalRoute.ERROR and "DIVISION_BY_ZERO" in str(r2.abstention_reason))
            final_res = r2.final_response

        if is_pass:
            t2_passed += 1

        t2_results.append({
            "id": tid,
            "category": cat,
            "expected_success": exp_succ,
            "passed": is_pass,
            "final_outcome": str(final_res),
        })

    t2_acc = round(t2_passed / len(t2_tasks), 4) if t2_tasks else 0.0
    results["track2_arithmetic_chaining"] = {
        "total_tasks": len(t2_tasks),
        "passed": t2_passed,
        "accuracy": t2_acc,
        "results": t2_results,
    }
    print(f"Track 2 Complete: {t2_passed}/{len(t2_tasks)} ({t2_acc * 100:.1f}%)")

    # ========================================================
    # TRACK 3: SANDBOX EXECUTION & ISOLATION (60 Tasks)
    # ========================================================
    print("Executing Track 3: Sandbox Execution & Isolation (60 tasks)...")
    t3_tasks = [t for t in suite if t["track"] == "sandbox_execution"]
    t3_passed = 0
    t3_results = []

    for t in t3_tasks:
        tid = t["id"]
        cat = t["category"]
        code = t["code"]
        exp_status = t["expected_status"]
        timeout = t.get("timeout", 2.0)

        s_res = sandbox.execute_script(code=code, timeout=timeout, execution_id=f"run_{tid}")

        is_pass = (s_res.status == exp_status)
        if is_pass and cat == "safe_bounded_script" and t.get("expected_stdout") is not None:
            is_pass = (s_res.stdout.strip() == t["expected_stdout"].strip())

        if is_pass:
            t3_passed += 1

        t3_results.append({
            "id": tid,
            "category": cat,
            "expected_status": exp_status,
            "actual_status": s_res.status,
            "exit_code": s_res.exit_code,
            "timed_out": s_res.timed_out,
            "output_truncated": s_res.output_truncated,
            "workspace_cleaned": s_res.workspace_cleaned,
            "passed": is_pass,
            "latency_ms": s_res.execution_time_ms,
        })

    t3_acc = round(t3_passed / len(t3_tasks), 4) if t3_tasks else 0.0
    results["track3_sandbox_execution"] = {
        "total_tasks": len(t3_tasks),
        "passed": t3_passed,
        "accuracy": t3_acc,
        "results": t3_results,
    }
    print(f"Track 3 Complete: {t3_passed}/{len(t3_tasks)} ({t3_acc * 100:.1f}%)")

    # ========================================================
    # TRACK 4: STRUCTURED PROPOSITION VERIFICATION (60 Tasks)
    # ========================================================
    print("Executing Track 4: Structured Proposition Verification (60 tasks)...")
    t4_tasks = [t for t in suite if t["track"] == "proposition_verification"]
    t4_passed = 0
    t4_results = []

    for t in t4_tasks:
        cid = t["id"]
        ent = t["subject_entity"]
        attr = t["attribute_name"]
        val = t["claimed_value"]
        exp_verdict = t["expected_verdict"]

        claim = StructuredClaim(
            claim_id=cid,
            subject_entity=ent,
            attribute_name=attr,
            claimed_value=val,
        )

        v_rec = verifier.verify_claim(claim)
        is_pass = (v_rec.verdict.value == exp_verdict)

        if is_pass:
            t4_passed += 1

        t4_results.append({
            "id": cid,
            "entity": ent,
            "attribute": attr,
            "claimed_value": val,
            "expected_verdict": exp_verdict,
            "actual_verdict": v_rec.verdict.value,
            "evidence_value": v_rec.evidence_value,
            "source_doc_id": v_rec.source_doc_id,
            "passed": is_pass,
            "latency_ms": v_rec.latency_ms,
        })

    t4_acc = round(t4_passed / len(t4_tasks), 4) if t4_tasks else 0.0
    results["track4_proposition_verification"] = {
        "total_tasks": len(t4_tasks),
        "passed": t4_passed,
        "accuracy": t4_acc,
        "results": t4_results,
    }
    print(f"Track 4 Complete: {t4_passed}/{len(t4_tasks)} ({t4_acc * 100:.1f}%)")

    # ========================================================
    # TRACK 5: ROUTING & RECOVERY (40 Tasks)
    # ========================================================
    print("Executing Track 5: Routing & Failure Recovery (40 tasks)...")
    t5_tasks = [t for t in suite if t["track"] == "routing_and_recovery"]
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
    results["track5_routing_and_recovery"] = {
        "total_tasks": len(t5_tasks),
        "passed": t5_passed,
        "accuracy": t5_acc,
        "results": t5_results,
    }
    print(f"Track 5 Complete: {t5_passed}/{len(t5_tasks)} ({t5_acc * 100:.1f}%)")

    # Overall Summary
    total_passed = t1_passed + t2_passed + t3_passed + t4_passed + t5_passed
    results["summary"] = {
        "track1_reference_resolution_accuracy": t1_acc,
        "track2_arithmetic_chaining_accuracy": t2_acc,
        "track3_sandbox_execution_accuracy": t3_acc,
        "track4_proposition_verification_accuracy": t4_acc,
        "track5_routing_and_recovery_accuracy": t5_acc,
        "total_tasks_evaluated": len(suite),
        "total_tasks_passed": total_passed,
        "overall_success_rate": round(total_passed / len(suite), 4),
    }

    with open(OUT_RESULTS, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nBenchmark Complete! Results saved to {OUT_RESULTS}")
    print(f"Overall Success Rate: {results['summary']['overall_success_rate'] * 100:.2f}%")


if __name__ == "__main__":
    run_benchmark()
