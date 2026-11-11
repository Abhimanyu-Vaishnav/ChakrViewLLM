# CR-16D Dataset Specification & Split Formulation

### 1. Multi-Task Consolidation Design
CR-EXP-16D required training mixtures that interleave sequence-copying tasks and key-value retrieval tasks in controlled proportions, with total dataset size fixed at $N = 4,000$ examples across all training arms.

```
Total Samples per Arm: 4,000
Batch Size: 32
Total Steps: 400 (3.2 epochs)
Optimizer: AdamW (lr=2e-4, weight_decay=0.01)
Loss Function: Target-Masked Cross-Entropy (prompt tokens masked with -100)
```

### 2. Experimental Training Arms
The four training arms are parameterized by the **Replay Ratio** of sequence copying:

1. **Arm A (Replay 00%):**
   - Copying Samples: 0
   - Retrieval Samples: 4,000
   - File: `arm_a_replay00_train.json` (SHA-256: `702d4ae7...`)
2. **Arm B (Replay 25%):**
   - Copying Samples: 1,000
   - Retrieval Samples: 3,000
   - File: `arm_b_replay25_train.json` (SHA-256: `85a1128b...`)
3. **Arm C (Replay 50%):**
   - Copying Samples: 2,000
   - Retrieval Samples: 2,000
   - File: `arm_c_replay50_train.json` (SHA-256: `7ae1cf2c...`)
4. **Arm D (Replay 75%):**
   - Copying Samples: 3,000
   - Retrieval Samples: 1,000
   - File: `arm_d_replay75_train.json` (SHA-256: `0b9bd001...`)

### 3. Validation Set Composition
- **File:** `cr16d_multitask_val.json` (SHA-256: `77f8c054...`)
- **Total Samples:** 600
  - 300 held-out sequence-copying examples
  - 300 held-out key-value retrieval examples
- Zero prompt overlap with any training partition or benchmark.

### 4. Benchmark Suites & Factorial Design
Three benchmarks evaluate each checkpoint:
1. **CR-16 Sequence Copying Benchmark (100 probes):**
   - Evaluates exact copy retention of alphanumeric sequences (length 3 to 12).
2. **CR-16C Key-Value Retrieval Benchmark (1,500 probes):**
   - Evaluates in-distribution direct retrieval (CAT1), reassignment (CAT2), multi-pair distractors (CAT6, CAT8), and abstention (CAT9, CAT10).
3. **CR-16D Compositional Generalization Benchmark (1,000 probes):**
   - File: `cr16d_compositional_benchmark_1000.json` (SHA-256: `ed19e0d0...`)
   - Evaluates systematic generalizability across a $2 \times 2$ factorial grid:
     - **G1 (Seen Keys & Seen Values in Novel Combinations):** 250 probes.
     - **G2 (Seen Keys & Unseen Values):** 250 probes.
     - **G3 (Unseen Keys & Seen Values):** 250 probes.
     - **G4 (Unseen Keys & Unseen Values):** 250 probes.

