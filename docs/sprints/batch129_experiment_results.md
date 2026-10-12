# Batch 129 — Controlled Pair-Density Experiment Results

## Executive Summary

Batch 129 executed a controlled experimental comparison on two-digit calculator argument binding under free-running greedy decoding.

Two conditions were trained under strictly identical optimization settings from the verified Stage 1 copy-task pre-trained checkpoint (`copy_task_sft_step150.pt`, SHA-256 `e84b2ea9b633b50181fb120ea2444a4a3c84c008b28905dd33c24fedba2b1bc4`):
1. **Arm 1 (Baseline Replication Arm)**: 100 training records (70 tool examples across 60 pairs + 30 negative controls).
   - Checkpoint: `binding_arm1_baseline_density_step200.pt` (SHA-256: `c1e39c9b8c69a7dfc7665a4a508fc36b649d597f18157d36ca9186f25101609a`)
2. **Arm 2 (Expanded Pair-Density Arm)**: 260 training records (200 tool examples across 160 pairs + 60 negative controls).
   - Checkpoint: `binding_arm2_expanded_density_step200.pt` (SHA-256: `f73f7fd328a96785ea77885fa81fd54d9781bdc0be21f0bbef8c42ad4243b75e`)

Both models were evaluated on the identical 45-example benchmark with zero pair or reversed-pair leakage.

---

## 1. Primary Evaluation Metrics ($N = 45$)

| Evaluation Metric | Arm 1 (Baseline Density, 70 Tools) | Arm 2 (Expanded Density, 200 Tools) |
| :--- | :---: | :---: |
| **Tool Examples Evaluated ($N$)** | 35 | 35 |
| **Tool Parse Rate** | 74.3% (26/35) | 74.3% (26/35) |
| **Tool Schema Validity Rate** | 74.3% (26/35) | 74.3% (26/35) |
| **Tool Name Accuracy (`calculator`)** | 74.3% (26/35) | 74.3% (26/35) |
| **Operation Accuracy (`add`)** | 74.3% (26/35) | 74.3% (26/35) |
| **Operand A Accuracy** | **34.3% (12/35)** | 11.4% (4/35) |
| **Operand B Accuracy** | **37.1% (13/35)** | 28.6% (10/35) |
| **Exact-Call Accuracy** | **22.9% (8/35)** | 5.7% (2/35) |
| **Allowlisted Execution Success** | 74.3% (26/35) | 74.3% (26/35) |
| **End-to-End Correctness** | **22.9% (8/35)** | 5.7% (2/35) |
| **Swapped Arguments ($a \leftrightarrow b$)** | **0 / 26 (0.0%)** | **0 / 26 (0.0%)** |
| **Negative Controls Evaluated ($N$)** | 10 | 10 |
| **Total False-Positive Rate** | **0.0% (0/10)** | **0.0% (0/10)** |
| **Clean Refusal Rate** | **100.0% (10/10)** | **100.0% (10/10)** |

---

## 2. Partition-Level Breakdown

### Partition 1: Familiar Pairs with Held-Out Templates ($N = 10$)
- **Arm 1 (Baseline Density)**: Parse rate 50.0% (5/10), Exact-call accuracy = **50.0% (5/10)**.
  - On `"Execute calculator addition: first={a}, second={b}."`, Arm 1 achieved **100% exact-call accuracy (5/5)**.
- **Arm 2 (Expanded Density)**: Parse rate 50.0% (5/10), Exact-call accuracy = **0.0% (0/10)**.
  - Arm 2 suffered from training pair interference: spreading 200 optimization steps over 160 pairs meant each pair was seen only $\approx 1.25$ times, diluting the reinforcement of specific pair patterns without mastering generalized copy mechanics.

### Partition 2: Unseen Pairs with Familiar Templates ($N = 15$)
- **Arm 1 (Baseline Density)**: Parse rate 100.0% (15/15), Exact-call accuracy = **20.0% (3/15)**, A acc = 33.3%, B acc = 40.0%.
- **Arm 2 (Expanded Density)**: Parse rate 100.0% (15/15), Exact-call accuracy = **0.0% (0/15)**, A acc = 13.3%, B acc = 26.7%.

### Partition 3: Unseen Pairs with Held-Out Templates ($N = 10$)
- **Arm 1 (Baseline Density)**: Parse rate 60.0% (6/10), Exact-call accuracy = **0.0% (0/10)**.
- **Arm 2 (Expanded Density)**: Parse rate 60.0% (6/10), Exact-call accuracy = **20.0% (2/10)**.
  - Arm 2 achieved 2/10 exact calls on double-generalization examples (`[29, 88]` -> `a: 29, b: 88` and `[98, 60]` -> `a: 98, b: 60`).

### Partition 4: Negative Controls ($N = 10$)
- **Arm 1**: 0.0% false positives (0/10), 100.0% clean refusals (10/10).
- **Arm 2**: 0.0% false positives (0/10), 100.0% clean refusals (10/10).
