import json
from pathlib import Path

checkpoints = {
    "ucip09": "experiments/ucip10e_r2/results/uc09/eval_r2_chakrview_ucip09_medium_model.json",
    "ucip09r": "experiments/ucip10e_r2/results/uc09r/eval_r2_chakrview_ucip09r_medium_model.json",
    "ucip09s": "experiments/ucip10e_r2/results/uc09s/eval_r2_ucip09s_balanced_pilot.json"
}

summary_results = {}

for name, rel_path in checkpoints.items():
    with open(rel_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    evaluations = data["evaluations"]
    total = len(evaluations)
    
    passed_count = sum(1 for e in evaluations if e["passed"] is True)
    failed_count = sum(1 for e in evaluations if e["passed"] is False)
    
    # Domain breakdown recalculation
    domain_stats = {}
    for e in evaluations:
        d = e["domain"]
        if d not in domain_stats:
            domain_stats[d] = {"total": 0, "passed": 0, "score_sum": 0.0}
        domain_stats[d]["total"] += 1
        if e["passed"]:
            domain_stats[d]["passed"] += 1
        domain_stats[d]["score_sum"] += float(e.get("score", 0.0))
    
    accuracy = (passed_count / total * 100.0) if total > 0 else 0.0
    
    # Verify reported vs recalculated
    rep_summary = data.get("summary", {})
    rep_passed = rep_summary.get("total_passed")
    rep_total = rep_summary.get("total_probes")
    
    summary_results[name] = {
        "file": rel_path,
        "checkpoint": rep_summary.get("checkpoint"),
        "sha256": rep_summary.get("checkpoint_sha256"),
        "total": total,
        "passed": passed_count,
        "failed": failed_count,
        "accuracy_pct": accuracy,
        "matches_reported_summary": (passed_count == rep_passed and total == rep_total),
        "domain_stats": domain_stats
    }

print(json.dumps(summary_results, indent=2))

with open("experiments/ucip10e_r2v/evaluations/recalculated_scores.json", "w", encoding="utf-8") as f:
    json.dump(summary_results, f, indent=2)
