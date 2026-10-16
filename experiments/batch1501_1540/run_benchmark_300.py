"""
Comprehensive 300-Task Benchmark Runner for Batches 1501–1540.
Location: experiments/batch1501_1540/run_benchmark_300.py

Executes:
1. Track 1: Safe Arithmetic Evaluation (100 tasks) vs Baseline Model
2. Track 2: Structured Verification Evaluation (100 tasks) vs Regex Baseline
3. Track 3: Tool Routing Classification (50 tasks)
4. Track 4: End-to-End Assistant Orchestration (50 tasks)

Saves machine-readable results to:
experiments/batch1501_1540/results/BENCHMARK_RESULTS_300.json
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
from chakrview.runtime.governed_tool_router import (
    GovernedToolRouter,
    CanonicalRoute,
)
from chakrview.runtime.local_knowledge_assistant import LocalKnowledgeAssistant

DATA_PATH = Path("experiments/batch1501_1540/data/evaluation_suite_300.json")
CORPUS_DIR = Path("experiments/batch1141_1180/data/corpus")
OUT_DIR = Path("experiments/batch1501_1540/results")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_RESULTS = OUT_DIR / "BENCHMARK_RESULTS_300.json"


def run_benchmark():
    print(f"Loading 300-task evaluation suite from {DATA_PATH}...")
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        suite = json.load(f)

    # Initialize Components
    calc = SafeCalculator()

    # Load Knowledge Assistant for indexing
    print(f"Initializing LocalKnowledgeAssistant and ingesting corpus from {CORPUS_DIR}...")
    asst = LocalKnowledgeAssistant(extract_concise_span=True)
    asst.ingest_directory(CORPUS_DIR)

    # Convert chunks to format verifier expects
    verifier = StructuredVerifier(chunks=asst.chunks)

    # Router
    router = GovernedToolRouter(calculator=calc, verifier=verifier, knowledge_assistant=asst)

    results = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "summary": {},
        "track1_arithmetic": {},
        "track2_verification": {},
        "track3_routing": {},
        "track4_e2e": {},
    }

    # ==========================================
    # RUN TRACK 1: ARITHMETIC (100 tasks)
    # ==========================================
    print("Executing Track 1: Safe Arithmetic Evaluation (100 tasks)...")
    t1_tasks = [t for t in suite if t["track"] == "arithmetic"]
    t1_passed = 0
    t1_results = []

    for t in t1_tasks:
        expr = t["expression"]
        exp_succ = t["expected_success"]
        exp_res = t["expected_result"]
        exp_err = t["expected_error"]

        c_res = calc.evaluate(expr)
        is_pass = False

        if exp_succ:
            # Expected mathematical success
            if c_res.success:
                # Compare numeric result
                if exp_res is not None and c_res.result is not None:
                    is_pass = (abs(c_res.result - exp_res) < 1e-4)
                else:
                    is_pass = (c_res.result == exp_res)
        else:
            # Expected rejection / domain error
            if not c_res.success:
                if exp_err:
                    is_pass = (c_res.error_code == exp_err)
                else:
                    is_pass = True

        if is_pass:
            t1_passed += 1

        t1_results.append({
            "id": t["id"],
            "expression": expr,
            "expected_success": exp_succ,
            "expected_result": exp_res,
            "expected_error": exp_err,
            "actual_success": c_res.success,
            "actual_result": c_res.result,
            "actual_error_code": c_res.error_code,
            "passed": is_pass,
            "latency_ms": c_res.execution_time_ms,
        })

    t1_acc = round(t1_passed / len(t1_tasks), 4) if t1_tasks else 0.0
    results["track1_arithmetic"] = {
        "total_tasks": len(t1_tasks),
        "passed": t1_passed,
        "accuracy": t1_acc,
        "results": t1_results,
    }
    print(f"Track 1 Complete: {t1_passed}/{len(t1_tasks)} ({t1_acc * 100:.1f}%)")

    # ==========================================
    # RUN TRACK 2: STRUCTURED VERIFICATION (100 tasks)
    # ==========================================
    print("Executing Track 2: Structured Verification Evaluation (100 tasks)...")
    t2_tasks = [t for t in suite if t["track"] == "verification"]
    t2_passed = 0
    t2_results = []

    for t in t2_tasks:
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
            t2_passed += 1

        t2_results.append({
            "id": cid,
            "entity": ent,
            "attribute": attr,
            "claimed_value": val,
            "expected_verdict": exp_verdict,
            "actual_verdict": v_rec.verdict.value,
            "evidence_value": v_rec.evidence_value,
            "source_doc_id": v_rec.source_doc_id,
            "passed": is_pass,
            "explanation": v_rec.explanation,
            "latency_ms": v_rec.latency_ms,
        })

    t2_acc = round(t2_passed / len(t2_tasks), 4) if t2_tasks else 0.0
    results["track2_verification"] = {
        "total_tasks": len(t2_tasks),
        "passed": t2_passed,
        "accuracy": t2_acc,
        "results": t2_results,
    }
    print(f"Track 2 Complete: {t2_passed}/{len(t2_tasks)} ({t2_acc * 100:.1f}%)")

    # ==========================================
    # RUN TRACK 3: ROUTING DISPATCH (50 tasks)
    # ==========================================
    print("Executing Track 3: Tool Routing Classification (50 tasks)...")
    t3_tasks = [t for t in suite if t["track"] == "routing"]
    t3_passed = 0
    t3_results = []

    for t in t3_tasks:
        tid = t["id"]
        q = t["query"]
        exp_route = t["expected_route"]

        route_dec = router.route_request(q, request_id=tid)
        is_pass = (route_dec.route.value == exp_route)

        if is_pass:
            t3_passed += 1

        t3_results.append({
            "id": tid,
            "query": q,
            "expected_route": exp_route,
            "actual_route": route_dec.route.value,
            "selection_reason": route_dec.selection_reason,
            "component_origin": route_dec.component_origin,
            "passed": is_pass,
            "latency_ms": route_dec.latency_ms,
        })

    t3_acc = round(t3_passed / len(t3_tasks), 4) if t3_tasks else 0.0
    results["track3_routing"] = {
        "total_tasks": len(t3_tasks),
        "passed": t3_passed,
        "accuracy": t3_acc,
        "results": t3_results,
    }
    print(f"Track 3 Complete: {t3_passed}/{len(t3_tasks)} ({t3_acc * 100:.1f}%)")

    # ==========================================
    # RUN TRACK 4: END-TO-END ASSISTANT (50 tasks)
    # ==========================================
    print("Executing Track 4: End-to-End Assistant Integration (50 tasks)...")
    t4_tasks = [t for t in suite if t["track"] == "e2e_assistant"]
    t4_passed = 0
    t4_results = []

    for t in t4_tasks:
        tid = t["id"]
        q = t["query"]
        exp_ans = t["expected_answer"]
        exp_route = t["expected_route"]

        resp = router.route_request(q, request_id=tid)

        # Correctness check: response must match expected route and contain answer
        route_ok = (resp.route.value == exp_route)
        ans_ok = False

        if exp_ans == "ABSTAIN":
            ans_ok = ("could not find sufficient" in resp.final_response.lower() or resp.route.value == "ABSTAIN")
        elif exp_ans in ("SUPPORTED", "CONTRADICTED"):
            ans_ok = (f"[{exp_ans}]" in resp.final_response or resp.result_payload == exp_ans)
        else:
            ans_ok = (str(exp_ans).lower() in resp.final_response.lower() or str(resp.result_payload) == str(exp_ans))

        is_pass = (route_ok and ans_ok)
        if is_pass:
            t4_passed += 1

        t4_results.append({
            "id": tid,
            "query": q,
            "expected_answer": exp_ans,
            "expected_route": exp_route,
            "actual_route": resp.route.value,
            "actual_response": resp.final_response,
            "component_origin": resp.component_origin,
            "citations": resp.citations,
            "passed": is_pass,
            "latency_ms": resp.latency_ms,
        })

    t4_acc = round(t4_passed / len(t4_tasks), 4) if t4_tasks else 0.0
    results["track4_e2e"] = {
        "total_tasks": len(t4_tasks),
        "passed": t4_passed,
        "accuracy": t4_acc,
        "results": t4_results,
    }
    print(f"Track 4 Complete: {t4_passed}/{len(t4_tasks)} ({t4_acc * 100:.1f}%)")

    # Overall Summary Matrix
    results["summary"] = {
        "track1_arithmetic_accuracy": t1_acc,
        "track2_verification_accuracy": t2_acc,
        "track3_routing_accuracy": t3_acc,
        "track4_e2e_accuracy": t4_acc,
        "total_tasks_evaluated": len(suite),
        "total_tasks_passed": t1_passed + t2_passed + t3_passed + t4_passed,
        "overall_success_rate": round(
            (t1_passed + t2_passed + t3_passed + t4_passed) / len(suite), 4
        ),
    }

    with open(OUT_RESULTS, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nAll Benchmark Runs Completed! Results saved to {OUT_RESULTS}")
    print(f"Overall Success Rate: {results['summary']['overall_success_rate'] * 100:.2f}%")


if __name__ == "__main__":
    run_benchmark()
