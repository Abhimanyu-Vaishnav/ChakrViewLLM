# CR-16D Leakage Audit & Contamination Prevention

### 1. Leakage Audit Objectives
To ensure that benchmark results reflect authentic in-context algorithmic generalization rather than memorized sequences, training data synthesis in CR-EXP-16D incorporated an automated strict anti-leakage filter.

### 2. Contamination Verification Process
Before generating training splits, all benchmark suites were loaded into memory, and their raw text prompts were added to an unalterable contamination filter:
1. `CR-16 Synthetic Copying Benchmark` (1,000 probes, SHA-256: `0b61e73d...`)
2. `CR-16C Sealed Retrieval Benchmark` (1,500 probes, SHA-256: `033f807b...`)
3. `CR-16D Compositional Generalization Benchmark` (1,000 probes, SHA-256: `ed19e0d0...`)
4. Held-out validation split (`cr16d_multitask_val.json`, 600 samples)

During synthetic generation, any proposed prompt that matched any string in the contamination set was rejected and re-synthesized.

### 3. Leakage Results

| Partition | Total Samples | Benchmark Collisions | Overlap Rate | Status |
|---|---|---|---|---|
| `arm_a_replay00_train.json` | 4,000 | 0 | 0.00% | **CLEAN** |
| `arm_b_replay25_train.json` | 4,000 | 0 | 0.00% | **CLEAN** |
| `arm_c_replay50_train.json` | 4,000 | 0 | 0.00% | **CLEAN** |
| `arm_d_replay75_train.json` | 4,000 | 0 | 0.00% | **CLEAN** |
| `cr16d_multitask_val.json` | 600 | 0 | 0.00% | **CLEAN** |
| `cr16d_compositional_benchmark_1000.json` | 1,000 | 0 | 0.00% | **CLEAN** |

### 4. Zero-Shot Split Verification
- **G1 (Novel Combinations):** Keys and values are drawn from the seen pools, but the pairings `(key, value)` strictly never appeared together in any training sample.
- **G2 (Unseen Values):** The target values (e.g., `Cotton`, `Mercury`, `Diamond`, `Spring`) were strictly quarantined from the training vocabulary of retrieval examples.
- **G3 (Unseen Keys):** The query keys (e.g., `season`, `planet`, `gem`, `weapon`) were strictly quarantined from the training pool of retrieval keys.
- **G4 (Unseen Both):** Neither the key nor the value appeared in retrieval training.

