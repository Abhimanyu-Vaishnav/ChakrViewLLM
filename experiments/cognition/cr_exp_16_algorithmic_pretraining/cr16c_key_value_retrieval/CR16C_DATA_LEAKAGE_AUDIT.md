# CR-EXP-16C Data Leakage Audit

## 1. Leakage Verification Scope
A strict cross-split collision audit was performed across:
- `cr16c_sealed_benchmark_1500.json` (1,500 probes)
- `cr16c_retrieval_train.json` (3,000 samples)
- `cr16c_retrieval_val.json` (500 samples)

## 2. Collision Check Results

| Check Description | Tested Dataset Pair | Size A | Size B | Collisions Found | Leakage Rate |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Benchmark vs Train Prompts** | Sealed Benchmark vs Training Set | 1,500 | 3,000 | **0** | **0.00%** |
| **Benchmark vs Val Prompts** | Sealed Benchmark vs Validation Set | 1,500 | 500 | **0** | **0.00%** |
| **Train vs Val Prompts** | Training Set vs Validation Set | 3,000 | 500 | **0** | **0.00%** |
| **Held-out Keys Exclusion** | Unseen Key Pool vs Train Facts | 8 keys | 3,000 | **0** | **0.00%** |

## 3. Audit Verdict
All 1,500 benchmark probes are 100% disjoint from all training and validation sequences. Zero leakage exists.

