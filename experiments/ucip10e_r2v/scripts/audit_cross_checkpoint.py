import json
from collections import defaultdict

models = {
    "ucip09": "experiments/ucip10e_r2/results/uc09/eval_r2_chakrview_ucip09_medium_model.json",
    "ucip09r": "experiments/ucip10e_r2/results/uc09r/eval_r2_chakrview_ucip09r_medium_model.json",
    "ucip09s": "experiments/ucip10e_r2/results/uc09s/eval_r2_ucip09s_balanced_pilot.json"
}

data = {}
for name, p in models.items():
    d = json.load(open(p, encoding="utf-8"))
    data[name] = {e["id"]: e for e in d["evaluations"]}

probes_list = list(data["ucip09"].keys())
total_probes = len(probes_list)

identical_passes = []
identical_failures = []
divergent_results = []

identical_raw_outputs = []

for pid in probes_list:
    e09 = data["ucip09"][pid]
    e09r = data["ucip09r"][pid]
    e09s = data["ucip09s"][pid]
    
    p09, p09r, p09s = e09["passed"], e09r["passed"], e09s["passed"]
    out09, out09r, out09s = e09["raw_output"], e09r["raw_output"], e09s["raw_output"]
    
    if p09 and p09r and p09s:
        identical_passes.append(pid)
    elif (not p09) and (not p09r) and (not p09s):
        identical_failures.append(pid)
    else:
        divergent_results.append((pid, p09, p09r, p09s))
        
    if out09 == out09r == out09s:
        identical_raw_outputs.append(pid)

print(f"Total Probes: {total_probes}")
print(f"Identical Passes across all 3: {len(identical_passes)}")
print(f"Identical Failures across all 3: {len(identical_failures)}")
print(f"Divergent Passes/Fails: {len(divergent_results)}")
print(f"Exact Same String Output across all 3: {len(identical_raw_outputs)}")

cross_audit = {
    "total_probes": total_probes,
    "identical_passes_count": len(identical_passes),
    "identical_passes_ids": identical_passes,
    "identical_failures_count": len(identical_failures),
    "divergent_results_count": len(divergent_results),
    "exact_raw_output_matches_count": len(identical_raw_outputs),
    "identical_pass_domain": "logic (all 10 are deductive logic square/rectangle probes)"
}

with open("experiments/ucip10e_r2v/evaluations/cross_checkpoint_comparison.json", "w", encoding="utf-8") as f:
    json.dump(cross_audit, f, indent=2)
