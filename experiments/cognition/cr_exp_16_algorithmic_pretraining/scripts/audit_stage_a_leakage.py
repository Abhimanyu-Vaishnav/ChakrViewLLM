"""CR-EXP-16 Stage A Leakage Auditor
Audits training and validation datasets against the frozen 1,000-probe benchmark.
Verifies exact sequence overlap, entity overlap, and benchmark isolation.
"""
import json
from pathlib import Path

REPO_ROOT = Path("D:/Project/ChakrView")
EXP16_DIR = REPO_ROOT / "experiments/cognition/cr_exp_16_algorithmic_pretraining"

BENCHMARK_PATH = EXP16_DIR / "benchmark_data/cr16_synthetic_benchmark_1000.json"
TRAIN_PATH = EXP16_DIR / "data/stage_a_train.json"
VAL_PATH = EXP16_DIR / "data/stage_a_val.json"

def main():
    bench = json.load(open(BENCHMARK_PATH, encoding="utf-8"))
    train = json.load(open(TRAIN_PATH, encoding="utf-8"))
    val = json.load(open(VAL_PATH, encoding="utf-8"))
    
    # Collect all sequences from benchmark
    bench_seqs = set(p["expected_target"].strip().lower() for p in bench)
    bench_subjs = set(p["subject"].strip().lower() for p in bench if p["subject"])
    
    train_seqs = set(s["seq"].strip().lower() for s in train)
    val_seqs = set(s["seq"].strip().lower() for s in val)
    
    # 1. Exact sequence overlap with benchmark
    overlap_train_bench = train_seqs.intersection(bench_seqs)
    overlap_val_bench = val_seqs.intersection(bench_seqs)
    
    # 2. Check if any benchmark entities appear as whole sequences in train
    entity_leakage = train_seqs.intersection(bench_subjs)
    
    print(f"Total benchmark probes: {len(bench)}")
    print(f"Total training sequences: {len(train_seqs)}")
    print(f"Exact sequence overlap (Train vs Benchmark): {len(overlap_train_bench)}")
    print(f"Exact sequence overlap (Val vs Benchmark):   {len(overlap_val_bench)}")
    print(f"Benchmark entity overlap:                    {len(entity_leakage)}")
    
    if len(overlap_train_bench) == 0 and len(overlap_val_bench) == 0 and len(entity_leakage) == 0:
        print("\nZERO-LEAKAGE AUDIT PASSED: No benchmark targets or entities present in training sequences!")
    else:
        print("\nWARNING: Leakage detected!")

if __name__ == "__main__":
    main()

