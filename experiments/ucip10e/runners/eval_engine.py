"""
UCIP-10E Evaluation Engine & CLI.

Provides commands:
- validate: Checks manifests and schemas.
- list: Lists registered benchmarks and capability domains.
- evaluate: Evaluates a model checkpoint on a benchmark suite.
- compare: Computes regression/progress delta between two evaluations.
- verify-artifacts: Checks SHA-256 of protected files and checkpoints.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import hashlib
from pathlib import Path
from typing import Dict, Any, List

REPO_ROOT = Path(r"D:\Project\ChakrView")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

BASE_DIR = REPO_ROOT / "experiments" / "ucip10e"

from experiments.ucip10e.runners.model_adapter import ModelAdapter
from experiments.ucip10e.evaluators.strict_evaluators import StrictEvaluators

def cmd_validate():
    print("========================================")
    print("UCIP-10E Manifest & Schema Validation")
    print("========================================")
    reg_file = BASE_DIR / "manifests" / "manifest_registry.json"
    if not reg_file.exists():
        print("FAIL: manifest_registry.json not found!")
        return 1
    reg = json.load(open(reg_file, encoding="utf-8"))
    for split, meta in reg["benchmark_suites"].items():
        p = REPO_ROOT / meta["path"]
        if not p.exists():
            print(f"FAIL: {split} suite missing at {p}!")
            return 1
        h = hashlib.sha256(open(p, "rb").read()).hexdigest()
        if h != meta["sha256"]:
            print(f"FAIL: {split} hash mismatch! Expected {meta['sha256']} got {h}")
            return 1
        data = json.load(open(p, encoding="utf-8"))
        print(f"PASS: {split.capitalize()} suite ({len(data)} probes, SHA-256: {h[:16]}...)")
    print("ALL MANIFESTS VALID.")
    return 0

def cmd_list():
    reg_file = BASE_DIR / "manifests" / "manifest_registry.json"
    reg = json.load(open(reg_file, encoding="utf-8"))
    print("\nRegistered Benchmark Suites:")
    for k, v in reg["benchmark_suites"].items():
        print(f"  - {k:15s}: {v['total_probes']} probes [{v['path']}]")
    print("\nCapability Domains:")
    domains = [
        "language", "mathematics", "copying", "logic", "generalization",
        "multi_turn", "self_correction", "uncertainty", "coding", "self_modeling"
    ]
    for d in domains:
        print(f"  - {d}")

def evaluate_probe(probe: Dict[str, Any], raw_output: str) -> Dict[str, Any]:
    eval_type = probe.get("evaluation_type")
    domain = probe.get("domain")

    if domain == "mathematics" or eval_type == "exact_math":
        return StrictEvaluators.evaluate_arithmetic(probe, raw_output)
    elif domain == "copying" or eval_type in ["verbatim_copy", "normalized_copy"]:
        return StrictEvaluators.evaluate_copying(probe, raw_output)
    elif domain == "language" or eval_type == "semantic_rubric":
        return StrictEvaluators.evaluate_language(probe, raw_output)
    elif domain == "logic" or eval_type == "deductive_logic":
        return StrictEvaluators.evaluate_logic(probe, raw_output)
    elif domain == "uncertainty" or eval_type == "abstention_calibration":
        return StrictEvaluators.evaluate_uncertainty(probe, raw_output)
    elif domain == "coding" or eval_type == "sandboxed_execution":
        return StrictEvaluators.evaluate_coding(probe, raw_output)
    elif domain == "self_modeling" or eval_type == "system_awareness":
        return StrictEvaluators.evaluate_self_modeling(probe, raw_output)
    else:
        # Fallback
        return StrictEvaluators.evaluate_language(probe, raw_output)

def cmd_evaluate(checkpoint_path: str, suite_name: str = "sealed", category: str | None = None, max_probes: int | None = None):
    reg_file = BASE_DIR / "manifests" / "manifest_registry.json"
    reg = json.load(open(reg_file, encoding="utf-8"))
    if suite_name not in reg["benchmark_suites"]:
        print(f"Error: suite '{suite_name}' not in registry!")
        return 1

    meta = reg["benchmark_suites"][suite_name]
    suite_path = REPO_ROOT / meta["path"]
    probes = json.load(open(suite_path, encoding="utf-8"))

    if category:
        probes = [p for p in probes if p.get("domain") == category]
        print(f"Filtered to category '{category}': {len(probes)} probes.")
    if max_probes and max_probes < len(probes):
        probes = probes[:max_probes]
        print(f"Limited evaluation to first {max_probes} probes.")

    adapter = ModelAdapter(checkpoint_path)

    total_passed = 0
    total_score = 0.0
    domain_stats: Dict[str, Dict[str, Any]] = {}
    records = []

    t0 = time.time()
    for idx, p in enumerate(probes):
        dom = p["domain"]
        if dom not in domain_stats:
            domain_stats[dom] = {"total": 0, "passed": 0, "score_sum": 0.0}
        domain_stats[dom]["total"] += 1

        prompt = p["prompt"]
        raw = adapter.generate(prompt, max_new_tokens=48)
        res = evaluate_probe(p, raw)

        passed = res.get("passed", False)
        score = res.get("score", 1.0 if passed else 0.0)

        if passed:
            total_passed += 1
            domain_stats[dom]["passed"] += 1
        total_score += score
        domain_stats[dom]["score_sum"] += score

        records.append({
            "id": p["id"],
            "domain": dom,
            "subdomain": p.get("subdomain"),
            "prompt": prompt,
            "raw_output": raw,
            "passed": passed,
            "score": score,
            "reason": res.get("reason"),
            "extracted": res.get("extracted")
        })

    elapsed = time.time() - t0
    acc = (total_passed / len(probes)) * 100 if probes else 0.0

    summary = {
        "checkpoint": str(Path(checkpoint_path).name),
        "checkpoint_sha256": adapter.sha256,
        "suite": suite_name,
        "suite_sha256": meta["sha256"],
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": elapsed,
        "total_probes": len(probes),
        "total_passed": total_passed,
        "accuracy_pct": acc,
        "domain_breakdown": domain_stats
    }

    out_name = f"eval_{Path(checkpoint_path).stem}_{suite_name}.json"
    out_file = BASE_DIR / "results" / out_name
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "evaluations": records}, f, indent=2, ensure_ascii=False)

    print("\n=======================================================")
    print(f"EVALUATION COMPLETE: {summary['checkpoint']} on {suite_name}")
    print(f"Total Passed: {total_passed}/{len(probes)} ({acc:.2f}%) | Time: {elapsed:.1f}s")
    print("Domain Breakdown:")
    for d, s in domain_stats.items():
        d_acc = (s["passed"] / s["total"]) * 100 if s["total"] else 0.0
        print(f"  {d:20s}: {s['passed']:3d}/{s['total']:3d} ({d_acc:5.1f}%)")
    print(f"Saved results to: {out_file}")
    return 0

def cmd_compare(baseline_res: str, candidate_res: str):
    b_data = json.load(open(baseline_res, encoding="utf-8"))
    c_data = json.load(open(candidate_res, encoding="utf-8"))

    b_sum = b_data["summary"]
    c_sum = c_data["summary"]

    print("==========================================================================")
    print("UCIP-10E CAPABILITY COMPARISON & REGRESSION AUDIT")
    print(f"Baseline:  {b_sum['checkpoint']} (Acc: {b_sum['accuracy_pct']:.2f}%)")
    print(f"Candidate: {c_sum['checkpoint']} (Acc: {c_sum['accuracy_pct']:.2f}%)")
    delta_total = c_sum["accuracy_pct"] - b_sum["accuracy_pct"]
    print(f"Overall Delta: {delta_total:+.2f}%")
    print("==========================================================================")
    print(f"{'Domain':22s} | {'Baseline':12s} | {'Candidate':12s} | {'Delta':8s} | {'Status'}")
    print("-" * 74)

    all_domains = sorted(list(set(b_sum["domain_breakdown"].keys()) | set(c_sum["domain_breakdown"].keys())))
    has_regression = False
    for d in all_domains:
        b_d = b_sum["domain_breakdown"].get(d, {"passed": 0, "total": 1})
        c_d = c_sum["domain_breakdown"].get(d, {"passed": 0, "total": 1})
        b_acc = (b_d["passed"] / b_d["total"]) * 100
        c_acc = (c_d["passed"] / c_d["total"]) * 100
        delta = c_acc - b_acc
        if delta < -2.0:
            status = "REGRESSION"
            has_regression = True
        elif delta > 2.0:
            status = "IMPROVED"
        else:
            status = "PARITY"
        print(f"{d:22s} | {b_acc:5.1f}% ({b_d['passed']}/{b_d['total']}) | {c_acc:5.1f}% ({c_d['passed']}/{c_d['total']}) | {delta:+6.1f}% | {status}")

    print("-" * 74)
    print(f"Regression Verdict: {'CRITICAL REGRESSIONS DETECTED' if has_regression else 'NO SEVERE REGRESSION'}")

def main():
    parser = argparse.ArgumentParser(description="ChakrView UCIP-10E Evaluation Infrastructure CLI")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("validate")
    sub.add_parser("list")

    p_eval = sub.add_parser("evaluate")
    p_eval.add_argument("--checkpoint", required=True)
    p_eval.add_argument("--suite", default="sealed")
    p_eval.add_argument("--category", default=None)
    p_eval.add_argument("--max_probes", type=int, default=None)

    p_comp = sub.add_parser("compare")
    p_comp.add_argument("--baseline", required=True)
    p_comp.add_argument("--candidate", required=True)

    args = parser.parse_args()
    if args.command == "validate":
        sys.exit(cmd_validate())
    elif args.command == "list":
        cmd_list()
    elif args.command == "evaluate":
        sys.exit(cmd_evaluate(args.checkpoint, args.suite, args.category, args.max_probes))
    elif args.command == "compare":
        cmd_compare(args.baseline, args.candidate)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
