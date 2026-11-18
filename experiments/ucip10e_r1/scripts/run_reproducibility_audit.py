import sys
import json
import random
import time
from pathlib import Path

REPO_ROOT = Path(r"D:\Project\ChakrView")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.ucip10e.runners.model_adapter import ModelAdapter
from experiments.ucip10e.runners.eval_engine import evaluate_probe

def run_reproducibility():
    core_p = REPO_ROOT / "experiments" / "ucip10e" / "datasets" / "sealed" / "benchmark_sealed_core100.json"
    probes = json.load(open(core_p, encoding="utf-8"))

    # Select 30 deterministic probes across all domains
    rng = random.Random(42)
    sample_indices = sorted(rng.sample(range(len(probes)), 30))
    sample_probes = [probes[i] for i in sample_indices]

    ckpts = {
        "ucip09": REPO_ROOT / "releases" / "ucip_09_candidates" / "chakrview_ucip09_medium_model.pt",
        "ucip09r": REPO_ROOT / "releases" / "ucip_09r_candidates" / "chakrview_ucip09r_medium_model.pt",
        "ucip09s": REPO_ROOT / "experiments" / "ucip09s" / "checkpoints" / "ucip09s_balanced_pilot.pt"
    }

    repro_results = {}
    for name, path in ckpts.items():
        print(f"\n--- Running Reproducibility Test: {name} (30 Probes) ---")
        adapter = ModelAdapter(path)
        
        # Run 1
        t0 = time.time()
        run1 = []
        for p in sample_probes:
            raw = adapter.generate(p["prompt"], max_new_tokens=48)
            ev = evaluate_probe(p, raw)
            run1.append({"id": p["id"], "raw": raw, "passed": ev.get("passed", False)})
            
        # Run 2 (To test exact determinism at temperature 0.0)
        run2 = []
        for p in sample_probes:
            raw = adapter.generate(p["prompt"], max_new_tokens=48)
            ev = evaluate_probe(p, raw)
            run2.append({"id": p["id"], "raw": raw, "passed": ev.get("passed", False)})

        # Check identical outputs
        exact_matches = sum(1 for r1, r2 in zip(run1, run2) if r1["raw"] == r2["raw"])
        score_matches = sum(1 for r1, r2 in zip(run1, run2) if r1["passed"] == r2["passed"])
        
        pass_count = sum(1 for r in run1 if r["passed"])
        acc = pass_count / len(sample_probes) * 100

        repro_results[name] = {
            "total_samples": len(sample_probes),
            "passed": pass_count,
            "accuracy": acc,
            "deterministic_token_match": (exact_matches == len(sample_probes)),
            "match_ratio": f"{exact_matches}/{len(sample_probes)}",
            "score_consistency": (score_matches == len(sample_probes)),
            "details": [{"id": r1["id"], "r1": r1["raw"], "r2": r2["raw"], "passed": r1["passed"]} for r1, r2 in zip(run1, run2)]
        }
        print(f"Results for {name}: {pass_count}/30 passed ({acc:.1f}%). Exact output match: {exact_matches}/30")

    out_file = REPO_ROOT / "experiments" / "ucip10e_r1" / "results" / "reproducibility" / "reproducibility_audit_30.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(repro_results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved reproducibility audit to: {out_file}")

if __name__ == "__main__":
    run_reproducibility()
