"""
UCIP-10E-R1 Full 840-Probe Evaluator Runner.
Runs the complete 840-probe sealed suite on a specified checkpoint
and outputs full results to experiments/ucip10e_r1/results/full_840/
"""

import sys
import json
import time
from pathlib import Path

REPO_ROOT = Path(r"D:\Project\ChakrView")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.ucip10e.runners.model_adapter import ModelAdapter
from experiments.ucip10e.runners.eval_engine import evaluate_probe

def run_full_840(checkpoint_path: str):
    ckpt_p = Path(checkpoint_path)
    suite_p = REPO_ROOT / "experiments" / "ucip10e" / "datasets" / "sealed" / "benchmark_sealed.json"
    probes = json.load(open(suite_p, encoding="utf-8"))

    print(f"\n=======================================================")
    print(f"Starting FULL 840-PROBE EVALUATION: {ckpt_p.name}")
    print(f"Total Probes: {len(probes)}")
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

        if (idx + 1) % 100 == 0 or (idx + 1) == len(probes):
            elapsed = time.time() - t0
            print(f"Progress: {idx + 1:3d}/{len(probes)} | Passed: {total_passed} ({total_passed/(idx+1)*100:.1f}%) | Elapsed: {elapsed:.1f}s")

    elapsed = time.time() - t0
    acc = (total_passed / len(probes)) * 100

    summary = {
        "checkpoint": ckpt_p.name,
        "checkpoint_sha256": adapter.sha256,
        "suite": "sealed_840",
        "total_probes": len(probes),
        "total_passed": total_passed,
        "accuracy_pct": acc,
        "elapsed_seconds": elapsed,
        "domain_breakdown": domain_stats
    }

    out_file = REPO_ROOT / "experiments" / "ucip10e_r1" / "results" / "full_840" / f"eval_full840_{ckpt_p.stem}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "evaluations": records}, f, indent=2, ensure_ascii=False)

    print("\n=======================================================")
    print(f"FULL EVALUATION COMPLETE: {ckpt_p.name}")
    print(f"Passed: {total_passed}/{len(probes)} ({acc:.2f}%) in {elapsed:.1f}s")
    print("Domain Breakdown:")
    for d, s in domain_stats.items():
        pct = (s["passed"] / s["total"]) * 100
        print(f"  {d:20s}: {s['passed']:3d}/{s['total']:3d} ({pct:5.1f}%)")
    print(f"Saved: {out_file}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_full_840.py <checkpoint_path>")
        sys.exit(1)
    run_full_840(sys.argv[1])
