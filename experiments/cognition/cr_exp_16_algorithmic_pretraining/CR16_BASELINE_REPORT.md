# CR-EXP-16 Baseline Report
## Evaluation of Canonical Stage B Checkpoint on Sealed Algorithmic Benchmark

**Mission Identifier:** CR-EXP-16  
**Checkpoint Evaluated:** `experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt`  
**Checkpoint SHA-256:** `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee`  
**Benchmark Suite:** Sealed Synthetic Algorithmic Benchmark (1,000 Probes, SHA-256: `0b61e73d57d903f34dc789736ecd9957c5633c1018fc8c8264bd1e1cb73cac97`)  

---

## 1. Baseline Performance Matrix

| Capability Category / Sub-Task | Probes ($N$) | Correct | Strict Accuracy | Value Match Count | Subject Match Count |
|---|---|---|---|---|---|
| **A. Exact Sequence Copying** | 100 | 0 | **0.00%** (0/100) | 0 | N/A |
| **B. Single-Token Copying (Dev, Roy...)**| 100 | 0 | **0.00%** (0/100) | 0 | 0 |
| **C. Multi-Token Copying** | 100 | 0 | **0.00%** (0/100) | 0 | 0 |
| **D. Key-Value Retrieval** | 100 | 0 | **0.00%** (0/100) | 0 | 0 |
| **E. Subject Identification** | 100 | 0 | **0.00%** (0/100) | 0 | 0 |
| **F. Static Subject-Value Binding** | 100 | 0 | **0.00%** (0/100) | 0 | 0 |
| **G. Single-Variable Reassignment** | 100 | 0 | **0.00%** (0/100) | 0 | 0 |
| **H. Multiple-Variable Reassignment** | 100 | 0 | **0.00%** (0/100) | 0 | 0 |
| **I. Distractor Resistance** | 100 | 0 | **0.00%** (0/100) | 0 | 0 |
| **J. Multi-Turn State Tracking** | 100 | 0 | **0.00%** (0/100) | 0 | 0 |
| **Overall Algorithmic Baseline** | **1,000** | **0** | **0.00%** (0/1,000) | **0** | **0** |

---

## 2. Qualitative Observations on Stage B Output

Inspection of raw completions from the canonical Stage B pretraining checkpoint reveals:
1. **Pretraining Domain Dominance:** The model outputs encyclopedic / book-like completions (`'The first of the world, the world, which is the world of the'`, `'The first of the city of the city of the city of'`).
2. **Total Absence of In-Context Induction:** When presented with `Echo sequence: delta beta omicron zeta\nOutput: `, the model has zero induction head activity that copies prior tokens into the output stream.
3. **Established Zero-Point Baseline:** The Stage B checkpoint establishes a true, uncompromised $0.00\%$ baseline across all 10 algorithmic capability tasks, ensuring any capability acquisition during CR-16 training can be measured unambiguously.

