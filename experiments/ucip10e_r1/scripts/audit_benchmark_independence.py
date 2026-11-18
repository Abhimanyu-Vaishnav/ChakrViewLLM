import json
import collections
from pathlib import Path

REPO_ROOT = Path(r"D:\Project\ChakrView")
BASE_DIR = REPO_ROOT / "experiments" / "ucip10e"

dev_path = BASE_DIR / "datasets" / "development" / "benchmark_dev.json"
sealed_path = BASE_DIR / "datasets" / "sealed" / "benchmark_sealed.json"
core100_path = BASE_DIR / "datasets" / "sealed" / "benchmark_sealed_core100.json"

dev_probes = json.load(open(dev_path, encoding="utf-8"))
sealed_probes = json.load(open(sealed_path, encoding="utf-8"))
core100_probes = json.load(open(core100_path, encoding="utf-8"))

# Training datasets to check for contamination
train_files = {
    "ucip08_train": REPO_ROOT / "experiments" / "cognition" / "ucip_08" / "data" / "ucip08_train.jsonl",
    "ucip09_train": REPO_ROOT / "experiments" / "cognition" / "ucip_09" / "data" / "ucip09_train.jsonl",
    "ucip09r_train": REPO_ROOT / "experiments" / "cognition" / "ucip_09r" / "data" / "ucip09r_train.jsonl",
    "ucip09s_train": REPO_ROOT / "experiments" / "ucip09s" / "datasets" / "ucip09s_balanced_train.jsonl"
}

training_prompts = set()
training_targets = set()
for name, p in train_files.items():
    if p.exists():
        for line in open(p, encoding="utf-8"):
            if not line.strip(): continue
            rec = json.loads(line)
            training_prompts.add(rec.get("prompt", "").strip().lower())
            training_targets.add(rec.get("target", "").strip().lower())

def audit_suite(name, probes):
    prompts = [p["prompt"].strip() for p in probes]
    prompt_counts = collections.Counter(prompts)
    unique_prompts = len(prompt_counts)
    exact_dups = sum(c - 1 for c in prompt_counts.values())
    
    # Check contamination against training sets
    contaminated_prompts = []
    for p in probes:
        prompt_str = p["prompt"].strip().lower()
        # Direct match or target match
        if prompt_str in training_prompts:
            contaminated_prompts.append((p["id"], p["domain"], p["prompt"]))
        else:
            # Check if format 'User: <prompt>\nAssistant:' was in training
            wrapped = f"user: {prompt_str}\nassistant:"
            if wrapped in training_prompts:
                contaminated_prompts.append((p["id"], p["domain"], p["prompt"]))

    domain_counts = collections.Counter(p["domain"] for p in probes)
    return {
        "suite": name,
        "total_probes": len(probes),
        "unique_prompts": unique_prompts,
        "duplicate_prompts": exact_dups,
        "domain_distribution": dict(domain_counts),
        "contaminated_prompts_count": len(contaminated_prompts),
        "contaminated_samples": contaminated_prompts[:10]
    }

dev_audit = audit_suite("development_840", dev_probes)
sealed_audit = audit_suite("sealed_840", sealed_probes)
core_audit = audit_suite("sealed_core100", core100_probes)

# Check overlap between dev and sealed
dev_prompt_set = set(p["prompt"].strip().lower() for p in dev_probes)
sealed_prompt_set = set(p["prompt"].strip().lower() for p in sealed_probes)
dev_sealed_overlap = dev_prompt_set.intersection(sealed_prompt_set)

res = {
    "dev_audit": dev_audit,
    "sealed_audit": sealed_audit,
    "core100_audit": core_audit,
    "dev_sealed_exact_prompt_overlap_count": len(dev_sealed_overlap),
    "dev_sealed_overlap_samples": list(dev_sealed_overlap)[:10]
}

with open(REPO_ROOT / "experiments" / "ucip10e_r1" / "results" / "benchmark_independence_audit.json", "w", encoding="utf-8") as f:
    json.dump(res, f, indent=2, ensure_ascii=False)

print("=== BENCHMARK INDEPENDENCE AUDIT ===")
print("Dev Suite:", dev_audit)
print("\nSealed Suite:", sealed_audit)
print("\nCore100 Suite:", core_audit)
print(f"\nExact Prompt Overlap between Dev and Sealed: {len(dev_sealed_overlap)}")
