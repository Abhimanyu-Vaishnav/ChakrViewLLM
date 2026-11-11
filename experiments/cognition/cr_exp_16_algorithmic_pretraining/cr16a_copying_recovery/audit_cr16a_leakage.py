"""CR-EXP-16A Leakage Auditor
Audits CR16A data against frozen benchmark.
"""
import json
from pathlib import Path

REPO_ROOT = Path("D:/Project/ChakrView")
BENCH_PATH = REPO_ROOT / "experiments/cognition/cr_exp_16_algorithmic_pretraining/benchmark_data/cr16_synthetic_benchmark_1000.json"
TRAIN_PATH = REPO_ROOT / "experiments/cognition/cr_exp_16_algorithmic_pretraining/cr16a_copying_recovery/data/cr16a_train.json"
VAL_PATH = REPO_ROOT / "experiments/cognition/cr_exp_16_algorithmic_pretraining/cr16a_copying_recovery/data/cr16a_val.json"

bench = json.load(open(BENCH_PATH, encoding="utf-8"))
train = json.load(open(TRAIN_PATH, encoding="utf-8"))
val = json.load(open(VAL_PATH, encoding="utf-8"))

bench_targets = set(p["expected_target"].strip().lower() for p in bench)
train_seqs = set(s["seq"].strip().lower() for s in train)
val_seqs = set(s["seq"].strip().lower() for s in val)

ov_train = train_seqs.intersection(bench_targets)
ov_val = val_seqs.intersection(bench_targets)

print(f"Overlap Train vs Benchmark: {len(ov_train)}")
print(f"Overlap Val vs Benchmark:   {len(ov_val)}")
assert len(ov_train) == 0 and len(ov_val) == 0, "Leakage detected!"
print("ZERO LEAKAGE CONFIRMED!")

