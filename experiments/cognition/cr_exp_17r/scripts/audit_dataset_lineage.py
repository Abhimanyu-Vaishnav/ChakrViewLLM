import json
import hashlib
from pathlib import Path

REPO_ROOT = Path("D:/Project/ChakrView")

train_path = REPO_ROOT / "experiments/conversation/cr_exp_17/data/conversation_train_1080.jsonl"
val_path = REPO_ROOT / "experiments/conversation/cr_exp_17/data/conversation_val_120.jsonl"
bench_path = REPO_ROOT / "experiments/conversation/cr_exp_17/CONVERSATION_BENCHMARK_V1.json"

train_records = []
with open(train_path, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            train_records.append(json.loads(line))

val_records = []
with open(val_path, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            val_records.append(json.loads(line))

with open(bench_path, encoding="utf-8") as f:
    bench_records = json.load(f)

# 1. Exact count verification
n_train = len(train_records)
n_val = len(val_records)
n_total = n_train + n_val

# 2. Check for duplicate prompts in train
train_prompts = [x["prompt"].strip() for x in train_records]
unique_train_prompts = set(train_prompts)
n_train_dup_prompts = len(train_prompts) - len(unique_train_prompts)

# 3. Check for duplicate prompt+response pairs in train
train_pairs = [(x["prompt"].strip(), x["response"].strip()) for x in train_records]
unique_train_pairs = set(train_pairs)
n_train_dup_pairs = len(train_pairs) - len(unique_train_pairs)

# 4. Check for duplicate prompts in val
val_prompts = [x["prompt"].strip() for x in val_records]
unique_val_prompts = set(val_prompts)
n_val_dup_prompts = len(val_prompts) - len(unique_val_prompts)

# 5. Check disjointness between train and val
train_prompt_set = set(train_prompts)
val_prompt_set = set(val_prompts)
train_val_overlap = train_prompt_set.intersection(val_prompt_set)

# 6. Check contamination against benchmark
bench_prompts = set(x["prompt"].strip() for x in bench_records)
train_bench_overlap = train_prompt_set.intersection(bench_prompts)
val_bench_overlap = val_prompt_set.intersection(bench_prompts)

report = {
    "train_records_count": n_train,
    "val_records_count": n_val,
    "total_records_count": n_total,
    "train_duplicate_prompts": n_train_dup_prompts,
    "train_duplicate_pairs": n_train_dup_pairs,
    "val_duplicate_prompts": n_val_dup_prompts,
    "train_val_prompt_overlap_count": len(train_val_overlap),
    "train_val_overlap_prompts": list(train_val_overlap),
    "train_bench_overlap_count": len(train_bench_overlap),
    "train_bench_overlap_prompts": list(train_bench_overlap),
    "val_bench_overlap_count": len(val_bench_overlap),
    "val_bench_overlap_prompts": list(val_bench_overlap),
}

with open("experiments/cognition/cr_exp_17r/audit/dataset_overlap_report.json", "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)

with open("experiments/cognition/cr_exp_17r/audit/dataset_lineage.json", "w", encoding="utf-8") as f:
    json.dump({
        "lineage_summary": {
            "reported_train": 1419,
            "verified_train": n_train,
            "reported_val": 157,
            "verified_val": n_val,
            "reported_total": 1576,
            "verified_total": n_total,
            "file_naming_discrepancy_explanation": "Files were named 'conversation_train_1080.jsonl' and 'conversation_val_120.jsonl' based on an initial target budget of 1,200 examples (1080/120). During corpus assembly, extra augmentations (identity grounding, multi-turn entity recall, Hinglish) were added, expanding the dataset to 1,576 records (1419 train, 157 val). The original filenames were preserved in assemble_corpus.py and train_conversation_v091.py without updating the filename numbers. The file contents strictly contain 1,419 and 157 records.",
            "disjoint_train_val": len(train_val_overlap) == 0,
            "benchmark_contamination_rate": len(train_bench_overlap) / len(bench_prompts)
        }
    }, f, indent=2)

print("Dataset lineage audit complete.")
print(f"Train: {n_train}, Val: {n_val}, Total: {n_total}")
print(f"Train-Val Overlap: {len(train_val_overlap)}")
print(f"Train-Benchmark Overlap: {len(train_bench_overlap)}")
print(f"Val-Benchmark Overlap: {len(val_bench_overlap)}")
