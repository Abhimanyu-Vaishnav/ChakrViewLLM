"""
UCIP-10E-R2 Evaluator Runner Engine.
Evaluates checkpoints on R2 Benchmark Suites using StrictEvaluatorsR2.
"""

import sys
import json
import time
from pathlib import Path

REPO_ROOT = Path(r"D:\Project\ChakrView")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

BASE_DIR = REPO_ROOT / "experiments" / "ucip10e_r2"

from experiments.ucip10e.runners.model_adapter import ModelAdapter
from experiments.ucip10e_r2.evaluators.strict_evaluators_r2 import StrictEvaluatorsR2

def evaluate_probe_r2(probe, raw_output):
    eval_type = probe.get("evaluation_type")
    domain = probe.get("domain")

    if domain == "mathematics" or eval_type == "exact_math":
        return StrictEvaluatorsR2.evaluate_arithmetic_r2(probe, raw_output)
    elif domain == "copying" or eval_type in ["verbatim_copy", "normalized_copy"]:
        return StrictEvaluatorsR2.evaluate_copying_r2(probe, raw_output)
    elif domain == "language" or eval_type == "semantic_rubric":
        return StrictEvaluatorsR2.evaluate_language_r2(probe, raw_output)
    elif domain == "logic" or eval_type == "deductive_logic":
        return StrictEvaluatorsR2.evaluate_logic_r2(probe, raw_output)
    elif domain == "uncertainty" or eval_type == "abstention_calibration":
        return StrictEvaluatorsR2.evaluate_uncertainty_r2(probe, raw_output)
    elif domain == "coding" or eval_type == "sandboxed_execution":
        return StrictEvaluatorsR2.evaluate_coding_r2(probe, raw_output)
    elif domain == "self_modeling" or eval_type == "system_awareness":
        return StrictEvaluatorsR2.evaluate_runtime_awareness_r2(probe, raw_output)
    else:
        return StrictEvaluatorsR2.evaluate_language_r2(probe, raw_output)

def run_r2_evaluation(checkpoint_path: str, output_subdir: str):
    ckpt_p = Path(checkpoint_path)
    suite_p = BASE_DIR / "datasets" / "sealed" / "benchmark_r2_sealed.json"
    probes = json.load(open(suite_p, encoding="utf-8"))

    print(f"\n=======================================================")
    print(f"Starting UCIP-10E-R2 CLEAN BENCHMARK EVALUATION: {ckpt_p.name}")
    print(f"Total Probes: {len(probes)} (Zero Contamination)")
    print(f"=======================================================")

    adapter = ModelAdapter(ckpt_p)

    total_passed = 0
    total_score = 0.0
    domain_stats = {}
    records = []

    t0 = time.time()
    for idx, p in enumerate(probes):
        dom = p["domain"]
        domain_stats.setdefault(dom, {"total": 0, "passed": 0, "score_sum": 0.0})
        domain_stats[dom]["total"] += 1

        prompt = p["prompt"]
        raw = adapter.generate(prompt, max_new_tokens=48)
        res = evaluate_probe_r2(p, raw)

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

        if (idx + 1) % 50 == 0 or (idx + 1) == len(probes):
            elapsed = time.time() - t0
            print(f"Progress: {idx + 1:3d}/{len(probes)} | Passed: {total_passed} ({total_passed/(idx+1)*100:.1f}%) | Elapsed: {elapsed:.1f}s")

    elapsed = time.time() - t0
    acc = (total_passed / len(probes)) * 100

    summary = {
        "checkpoint": ckpt_p.name,
        "checkpoint_sha256": adapter.sha256,
        "suite": "benchmark_r2_sealed",
        "total_probes": len(probes),
        "total_passed": total_passed,
        "accuracy_pct": acc,
        "elapsed_seconds": elapsed,
        "domain_breakdown": domain_stats
    }

    out_dir = BASE_DIR / "results" / output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"eval_r2_{ckpt_p.stem}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "evaluations": records}, f, indent=2, ensure_ascii=False)

    print("\n=======================================================")
    print(f"R2 EVALUATION COMPLETE: {ckpt_p.name}")
    print(f"Passed: {total_passed}/{len(probes)} ({acc:.2f}%) in {elapsed:.1f}s")
    print("Domain Breakdown:")
    for d, s in domain_stats.items():
        pct = (s["passed"] / s["total"]) * 100
        print(f"  {d:20s}: {s['passed']:3d}/{s['total']:3d} ({pct:5.1f}%)")
    print(f"Saved: {out_file}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python run_r2_eval.py <checkpoint_path> <output_subdir>")
        sys.exit(1)
    run_r2_evaluation(sys.argv[1], sys.argv[2])
