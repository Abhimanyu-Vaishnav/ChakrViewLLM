import sys
import json
import random
from pathlib import Path

REPO_ROOT = Path(r"D:\Project\ChakrView")
BASE_DIR = REPO_ROOT / "experiments" / "ucip10e_r2"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.ucip10e.runners.model_adapter import ModelAdapter
from experiments.ucip10e_r2.scripts.run_r2_eval import evaluate_probe_r2

def run_r2_reproducibility():
    suite_p = BASE_DIR / "datasets" / "sealed" / "benchmark_r2_sealed.json"
    probes = json.load(open(suite_p, encoding="utf-8"))

    # Sample 30 deterministic probes across domains
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
        print(f"\n--- Running R2 Reproducibility Audit: {name} (30 Probes) ---")
        adapter = ModelAdapter(path)

        run1 = []
        for p in sample_probes:
            raw = adapter.generate(p["prompt"], max_new_tokens=48)
            ev = evaluate_probe_r2(p, raw)
            run1.append({"id": p["id"], "raw": raw, "passed": ev.get("passed", False)})

        run2 = []
        for p in sample_probes:
            raw = adapter.generate(p["prompt"], max_new_tokens=48)
            ev = evaluate_probe_r2(p, raw)
            run2.append({"id": p["id"], "raw": raw, "passed": ev.get("passed", False)})

        exact_matches = sum(1 for r1, r2 in zip(run1, run2) if r1["raw"] == r2["raw"])
        score_matches = sum(1 for r1, r2 in zip(run1, run2) if r1["passed"] == r2["passed"])
        pass_count = sum(1 for r in run1 if r["passed"])

        repro_results[name] = {
            "total_samples": len(sample_probes),
            "passed": pass_count,
            "deterministic_token_match": (exact_matches == len(sample_probes)),
            "match_ratio": f"{exact_matches}/{len(sample_probes)}",
            "score_consistency": (score_matches == len(sample_probes)),
            "details": [{"id": r1["id"], "r1": r1["raw"], "r2": r2["raw"], "passed": r1["passed"]} for r1, r2 in zip(run1, run2)]
        }
        print(f"Results for {name}: {pass_count}/30 passed. Exact token matches: {exact_matches}/30")

    out_file = BASE_DIR / "results" / "reproducibility" / "reproducibility_r2_audit_30.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(repro_results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved R2 reproducibility audit to: {out_file}")

if __name__ == "__main__":
    run_r2_reproducibility()
