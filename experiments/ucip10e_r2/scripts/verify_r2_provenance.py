import json
import collections
from pathlib import Path

REPO_ROOT = Path(r"D:\Project\ChakrView")
BASE_DIR = REPO_ROOT / "experiments" / "ucip10e_r2"

dev_path = BASE_DIR / "datasets" / "development" / "benchmark_r2_development.json"
sealed_path = BASE_DIR / "datasets" / "sealed" / "benchmark_r2_sealed.json"

dev_probes = json.load(open(dev_path, encoding="utf-8"))
sealed_probes = json.load(open(sealed_path, encoding="utf-8"))

train_files = {
    "ucip08_train": REPO_ROOT / "experiments" / "cognition" / "ucip_08" / "data" / "ucip08_train.jsonl",
    "ucip09_train": REPO_ROOT / "experiments" / "cognition" / "ucip_09" / "data" / "ucip09_train.jsonl",
    "ucip09r_train": REPO_ROOT / "experiments" / "cognition" / "ucip_09r" / "data" / "ucip09r_train.jsonl",
    "ucip09s_train": REPO_ROOT / "experiments" / "ucip09s" / "datasets" / "ucip09s_balanced_train.jsonl",
    "ucip10e_sealed_old": REPO_ROOT / "experiments" / "ucip10e" / "datasets" / "sealed" / "benchmark_sealed.json"
}

training_prompts = set()
for name, p in train_files.items():
    if p.exists():
        if p.suffix == ".jsonl":
            for line in open(p, encoding="utf-8"):
                if not line.strip(): continue
                training_prompts.add(json.loads(line).get("prompt", "").strip().lower())
        elif p.suffix == ".json":
            for r in json.load(open(p, encoding="utf-8")):
                training_prompts.add(r.get("prompt", "").strip().lower())

def audit_r2_suite(name, probes):
    prompts = [p["prompt"].strip() for p in probes]
    prompt_counts = collections.Counter(prompts)
    unique_prompts = len(prompt_counts)
    exact_dups = sum(c - 1 for c in prompt_counts.values())
    
    contaminated = []
    for p in probes:
        pr = p["prompt"].strip().lower()
        if pr in training_prompts:
            contaminated.append((p["id"], p["domain"], p["prompt"]))

    domain_counts = collections.Counter(p["domain"] for p in probes)
    return {
        "suite": name,
        "total_probes": len(probes),
        "unique_prompts": unique_prompts,
        "duplicate_prompts": exact_dups,
        "domain_distribution": dict(domain_counts),
        "training_contamination_count": len(contaminated),
        "contaminated_samples": contaminated
    }

dev_res = audit_r2_suite("r2_development", dev_probes)
sealed_res = audit_r2_suite("r2_sealed", sealed_probes)

dev_set = set(p["prompt"].strip().lower() for p in dev_probes)
sealed_set = set(p["prompt"].strip().lower() for p in sealed_probes)
overlap = dev_set.intersection(sealed_set)

res = {
    "dev_audit": dev_res,
    "sealed_audit": sealed_res,
    "dev_sealed_prompt_overlap_count": len(overlap),
    "overlap_samples": list(overlap)
}

with open(BASE_DIR / "reports" / "clean_benchmark_provenance_data.json", "w", encoding="utf-8") as f:
    json.dump(res, f, indent=2)

print("=== R2 CLEAN BENCHMARK PROVENANCE AUDIT ===")
print("R2 Dev Suite:", dev_res)
print("\nR2 Sealed Suite:", sealed_res)
print(f"\nExact Prompt Overlap between R2 Dev and R2 Sealed: {len(overlap)}")
