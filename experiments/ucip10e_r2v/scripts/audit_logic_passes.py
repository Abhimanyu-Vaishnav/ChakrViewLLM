import json

bench_sealed = json.load(open("experiments/ucip10e_r2/datasets/sealed/benchmark_r2_sealed.json", encoding="utf-8"))
bench_map = {p["id"]: p for p in bench_sealed}

checkpoints = {
    "ucip09": "experiments/ucip10e_r2/results/uc09/eval_r2_chakrview_ucip09_medium_model.json",
    "ucip09r": "experiments/ucip10e_r2/results/uc09r/eval_r2_chakrview_ucip09r_medium_model.json",
    "ucip09s": "experiments/ucip10e_r2/results/uc09s/eval_r2_ucip09s_balanced_pilot.json"
}

audit_records = []

for name, path in checkpoints.items():
    data = json.load(open(path, encoding="utf-8"))
    passes = [e for e in data["evaluations"] if e["passed"]]
    for p in passes:
        meta = bench_map.get(p["id"], {})
        audit_records.append({
            "model": name,
            "probe_id": p["id"],
            "domain": p["domain"],
            "subdomain": p.get("subdomain"),
            "prompt": p["prompt"],
            "expected_eval_type": meta.get("evaluation_type"),
            "raw_output": p["raw_output"],
            "score": p["score"],
            "reason": p["reason"]
        })

print(f"Total logic passes collected across 3 models: {len(audit_records)}")
with open("experiments/ucip10e_r2v/evaluations/logic_passes_forensics.json", "w", encoding="utf-8") as f:
    json.dump(audit_records, f, indent=2)

# Check uniqueness of prompts among passes
unique_prompts = set(r["prompt"] for r in audit_records)
unique_outputs = set(r["raw_output"] for r in audit_records)
print(f"Unique prompts among passes: {len(unique_prompts)}")
print(f"Prompts: {unique_prompts}")
print(f"Unique outputs among passes: {len(unique_outputs)}")
print(f"Outputs: {unique_outputs}")
