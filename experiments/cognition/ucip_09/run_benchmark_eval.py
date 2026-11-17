import sys
import json
import argparse
from pathlib import Path

REPO_ROOT = Path(r"D:\Project\ChakrView")
sys.path.insert(0, str(REPO_ROOT))

from scripts.run_m3_evaluations import DirectNeuralRunner
from experiments.cognition.ucip_09.evaluator_sealed_500 import evaluate_probe

def run_evaluation(ckpt_path: Path, out_path: Path, max_tokens: int = 40):
    bench_path = REPO_ROOT / "experiments" / "cognition" / "ucip_09" / "sealed_language_intelligence_500.json"
    bench_data = json.load(open(bench_path, encoding="utf-8"))
    probes = bench_data["probes"]

    print(f"Loading checkpoint from: {ckpt_path.name}")
    runner = DirectNeuralRunner(ckpt_path)

    results = []
    cat_stats = {}
    total_passed = 0

    for idx, p in enumerate(probes):
        cat = p["category"]
        if cat not in cat_stats:
            cat_stats[cat] = {"total": 0, "passed": 0}
        cat_stats[cat]["total"] += 1

        formatted_prompt = f"User: {p['prompt']}\nAssistant:"
        res = runner.generate(formatted_prompt, max_new_tokens=max_tokens, temperature=0.0)
        gen_text = res["text"].strip()

        eval_res = evaluate_probe(p, gen_text)
        is_pass = eval_res["passed"]
        if is_pass:
            total_passed += 1
            cat_stats[cat]["passed"] += 1

        results.append({
            "id": p["id"],
            "category": cat,
            "prompt": p["prompt"],
            "generated": gen_text,
            "passed": is_pass,
            "status": eval_res["status"],
            "reason": eval_res["reason"]
        })

    acc = (total_passed / len(probes)) * 100.0

    print(f"\n--- Benchmark Results: {ckpt_path.name} ---")
    print(f"Overall Accuracy: {total_passed}/{len(probes)} ({acc:.1f}%)")
    for cat, s in cat_stats.items():
        c_acc = (s["passed"] / s["total"]) * 100.0 if s["total"] > 0 else 0.0
        print(f"  {cat:<25}: {s['passed']:>2}/{s['total']:<2} ({c_acc:>5.1f}%)")

    report = {
        "checkpoint": ckpt_path.name,
        "total_probes": len(probes),
        "total_passed": total_passed,
        "accuracy_percent": acc,
        "categories": cat_stats,
        "evaluations": results
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Saved evaluation report to: {out_path.name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    run_evaluation(Path(args.checkpoint), Path(args.out))
