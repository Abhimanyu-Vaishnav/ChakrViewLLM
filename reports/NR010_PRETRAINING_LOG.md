# CV-GENESIS-NR-010 17M Pretraining Log

## 1. Executive Summary

- **Run ID:** `genesis_17m_pretraining_010`
- **Model:** `ChakrGenesis-17M` (17,063,808 parameters)
- **Dataset:** Stage-C Corpus (27 Train Shards, 6,651,093 tokens)
- **Hardware:** NVIDIA RTX A1000 Laptop GPU (6.0 GB VRAM)
- **Duration:** 283.3 seconds (~4.7 minutes)

---

## 2. Loss & Convergence Trajectory

| Training Step | Global Step | Learning Rate | Train Loss | Validation Loss | Validation Perplexity |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **0 (Initialization)** | 0 | — | — | **8.3911** | 4,407.53 |
| **Step 200** | 200 | $3.95 \times 10^{-4}$ | 5.0869 | — | — |
| **Step 400** | 400 | $3.63 \times 10^{-4}$ | 4.0750 | — | — |
| **Step 600** | 600 | $3.09 \times 10^{-4}$ | 2.9216 | — | — |
| **Step 800** | 800 | $2.39 \times 10^{-4}$ | 3.1266 | — | — |
| **Step 1000** | 1000 | $1.67 \times 10^{-4}$ | 3.0396 | — | — |
| **Step 1200** | 1200 | $1.03 \times 10^{-4}$ | 2.7017 | — | — |
| **Step 1400** | 1400 | $5.84 \times 10^{-5}$ | 2.4519 | — | — |
| **Step 1600** | 1600 | $4.02 \times 10^{-5}$ | 2.6853 | — | — |
| **Epoch 1 Evaluation** | 1624 | $4.00 \times 10^{-5}$ | 2.8779 | **2.8425** | **17.16** |

**Total Validation Loss Reduction:** From $8.3911$ to **$2.8425$** (Perplexity dropped by **$99.6\%$** from $4,407$ to **$17.16$**).
