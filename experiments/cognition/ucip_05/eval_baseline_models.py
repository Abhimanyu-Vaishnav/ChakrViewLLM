import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

import json
from scripts.run_m3_evaluations import DirectNeuralRunner
from experiments.cognition.ucip_05.evaluator import evaluate_response

benchmark_path = REPO_ROOT / "experiments/cognition/ucip_05/baseline_benchmark.json"
with open(benchmark_path, "r", encoding="utf-8") as f:
    benchmark = json.load(f)

for label, ckpt_rel in [
    ("UCIP-04 Micro", "releases/ucip_04_candidate/chakrview_ucip04_model.pt"),
    ("UCIP-04 Medium", "releases/ucip_04_candidate/chakrview_ucip04_medium_model.pt")
]:
    runner = DirectNeuralRunner(REPO_ROOT / ckpt_rel)
    total_passed = 0
    cat_stats = {}
    details = []
    
    for probe in benchmark:
        cat = probe["category"]
        if cat not in cat_stats:
            cat_stats[cat] = {"passed": 0, "total": 0}
        cat_stats[cat]["total"] += 1
        
        prompt = f"User: {probe['input']}\nAssistant:"
        out = runner.generate(prompt, max_new_tokens=40)
        res = evaluate_response(probe, out["text"])
        if res["passed"]:
            total_passed += 1
            cat_stats[cat]["passed"] += 1
        details.append({
            "probe_id": probe["probe_id"],
            "input": probe["input"],
            "category": cat,
            "generated": out["text"],
            "eval": res
        })
            
    print(f"=== {label} Baseline Benchmark Results ===")
    acc = total_passed / len(benchmark) * 100.0
    print(f"Total Score: {total_passed}/{len(benchmark)} ({acc:.1f}%)")
    for cat, s in cat_stats.items():
        cat_acc = s["passed"] / s["total"] * 100.0
        print(f"  {cat:28s}: {s['passed']:2d}/{s['total']:2d} ({cat_acc:.1f}%)")
    print()

    # Save details
    out_name = f"baseline_eval_{label.lower().replace(' ', '_')}.json"
    with open(REPO_ROOT / f"experiments/cognition/ucip_05/{out_name}", "w", encoding="utf-8") as f:
        json.dump({"total_passed": total_passed, "total": len(benchmark), "cat_stats": cat_stats, "details": details}, f, indent=2, ensure_ascii=False)
