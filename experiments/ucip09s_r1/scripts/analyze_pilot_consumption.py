import json
import collections
from pathlib import Path

REPO_ROOT = Path(r"D:\Project\ChakrView")

train_path = REPO_ROOT / "experiments" / "ucip09s" / "datasets" / "ucip09s_balanced_train.jsonl"
train_data = [json.loads(line) for line in open(train_path, encoding="utf-8") if line.strip()]

# Replicate trainer sampling with seed 42
import random
random.seed(42)

indices = list(range(len(train_data)))
random.shuffle(indices)

steps = 80
batch_size = 4
total_exposures = steps * batch_size # 320 samples

consumed_categories = collections.Counter()
consumed_targets = []
data_idx = 0

for step in range(1, steps + 1):
    for _ in range(batch_size):
        if data_idx >= len(indices):
            random.shuffle(indices)
            data_idx = 0
        ex = train_data[indices[data_idx]]
        data_idx += 1
        cat = ex.get("category", "unknown")
        consumed_categories[cat] += 1
        consumed_targets.append(ex.get("target", ""))

print(f"Total training exposures: {len(consumed_targets)} across {steps} steps")
print("Consumed Category Distribution:")
for cat, count in consumed_categories.most_common():
    print(f"  {cat:25s}: {count:3d} ({count / total_exposures * 100:.1f}%)")

with open(REPO_ROOT / "experiments" / "ucip09s_r1" / "evaluations" / "pilot_consumed_exposure.json", "w", encoding="utf-8") as f:
    json.dump({
        "total_exposures": total_exposures,
        "unique_samples_in_dataset": len(train_data),
        "steps": steps,
        "batch_size": batch_size,
        "consumed_categories": dict(consumed_categories)
    }, f, indent=2)
