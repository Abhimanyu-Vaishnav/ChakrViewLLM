# CR-EXP-16C Retrieval Evaluation Report

## 1. Overall Performance Across Evaluated Models
Evaluated strictly against the sealed 1,500-probe suite (`cr16c_sealed_benchmark_1500.json`, SHA-256: `033f807ba518c168...`):

| Model Evaluated | Objective / Seed | Exact Match (1500) | Answer Coverage | Stale-Value Errors | Distractor Captures | Abstention Recall |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Stage B Baseline** | Parent Step 10k | 0 / 1500 (0.00%) | 0.0% | 0 | 0 | 0.0% (0/100) |
| **CR-16A Arm D** | Sequence Copying ckpt | 4 / 1500 (0.27%) | 0.3% | 0 | 0 | 0.0% (0/100) |
| **Retrieval Arm A** | Target-Masked (Seed 42) | **885 / 1500 (59.00%)** | **59.0%** | **1 / 350 (0.3%)** | **0 / 150 (0.0%)** | **58.0% (58/100)** |
| **Retrieval Arm A** | Target-Masked (Seed 101) | **868 / 1500 (57.87%)** | **57.9%** | **0 / 350 (0.0%)** | **0 / 150 (0.0%)** | **48.0% (48/100)** |
| **Retrieval Arm B** | Full-Loss (Seed 42) | 730 / 1500 (48.67%) | 48.7% | 12 / 350 (3.4%) | 0 / 150 (0.0%) | 64.0% (64/100) |
| **Retrieval Arm B** | Full-Loss (Seed 101) | 772 / 1500 (51.47%) | 51.5% | 16 / 350 (4.6%) | 0 / 150 (0.0%) | 63.0% (63/100) |

## 2. In-Distribution vs Out-of-Distribution Breakdown
- **In-Distribution Contexts (CAT1, CAT2, CAT3, CAT4, CAT8 - 850 probes):**
  - Retrieval Arm A achieved **807 / 850 (94.94%) exact match**.
  - Dynamic reassignment reached **99.5%**.
  - Multi-fact extraction reached **97.3%**.
  - Distractor filtering reached **100.0%**.
- **Out-of-Distribution Contexts (CAT5, CAT6, CAT7, CAT10 - 550 probes):**
  - Unseen templates (CAT10): **20.0%**.
  - Unseen keys / values (CAT5, CAT6, CAT7): **0.0%**.

