import json
from pathlib import Path

REPO_ROOT = Path(r"D:\Project\ChakrView")

CORPORA = {
    "ucip06_train": REPO_ROOT / "experiments" / "cognition" / "ucip_06" / "data" / "ucip06_train.jsonl",
    "ucip07_train": REPO_ROOT / "experiments" / "cognition" / "ucip_07" / "data" / "ucip07_train.jsonl",
    "ucip08_train": REPO_ROOT / "experiments" / "cognition" / "ucip_08" / "data" / "ucip08_train.jsonl",
}

print(f"{'Corpus':<15} | {'Records':<8} | {'Unique Prompts':<15} | {'Unique Targets':<15} | {'Dup Rate':<10}")
print("-" * 75)

for name, path in CORPORA.items():
    if not path.exists():
        print(f"{name:<15} | NOT FOUND")
        continue
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    prompts = [r.get("prompt", "") for r in records]
    targets = [r.get("target", "") for r in records]
    
    u_prompts = len(set(prompts))
    u_targets = len(set(targets))
    dup_rate = (len(records) - u_prompts) / len(records) * 100.0 if records else 0.0
    
    print(f"{name:<15} | {len(records):<8} | {u_prompts:<15} | {u_targets:<15} | {dup_rate:>8.1f}%")
