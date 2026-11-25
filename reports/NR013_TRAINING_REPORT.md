# CHAKRVIEW RESEARCH REPORT: NR-013 TRAINING REPORT
**Mission ID:** CV-GENESIS-NR-013  
**Date:** 2026-11-25  
**Candidate Architecture:** ChakrGenesis-17M (17,063,808 parameters)  
**Initialization Checkpoint:** `experiments/genesis_stage_d_011/checkpoints/genesis_17m_staged_best.pt` (`2c21632e`)  
**Training Dataset:** `data/stage_e_corpus_013/stage_e_train.jsonl` (5,238 balanced samples)  
**Validation Dataset:** `data/stage_e_corpus_013/stage_e_val.jsonl` (559 validation samples)  

---

## 1. Executive Summary
Under Phase 5 of Mission CV-GENESIS-NR-013, the ChakrGenesis-17M neural foundation was fine-tuned on the balanced Stage-E corpus, combining verified Python coding datasets with the Stage-D multilingual conversational base.

---

## 2. Training Hyperparameters & Execution Ledger
- **Total Optimizer Steps:** 1,308 steps (2 full epochs)
- **Batch Size:** 8 sequences of length 256
- **Peak Learning Rate:** $8.0 \times 10^{-5}$ (AdamW $\beta_1=0.9, \beta_2=0.95$, weight decay 0.01)
- **Loss Masking:** Label masking ($-100$) on prompt tokens; backpropagation restricted exclusively to response tokens.
- **Hardware Profile:** NVIDIA RTX A1000 Laptop GPU (CUDA 12.8 compute capability)
- **Training Throughput:** ~11,300 tokens/second average
- **Total Tokens Processed:** 2,678,784 tokens

---

## 3. Convergence Progression

| Training Step | Training Step Loss | Validation Loss | Validation Perplexity | Best Checkpoint Event |
| :--- | :--- | :--- | :--- | :--- |
| **0 (Init)** | — | 8.6437 | 5674.52 | Baseline initialization |
| **Step 75** | 0.5405 | 1.1577 | 3.18 | Saved `genesis_17m_stage_e_best.pt` |
| **Step 150** | 0.6818 | 0.5719 | 1.77 | Saved new best |
| **Step 300** | 0.1329 | 0.2881 | 1.33 | Saved new best |
| **Step 450** | 0.0325 | 0.1834 | 1.20 | Saved new best |
| **Step 675** | 0.1159 | 0.1260 | 1.13 | Saved new best |
| **Step 900** | 0.1181 | 0.1094 | 1.12 | Saved new best |
| **Step 1125** | 0.0355 | 0.0909 | 1.10 | Saved new best |
| **Step 1308 (Final)**| 0.0013 | **0.0774** | **1.08** | **New Project Record Best** |

---

## 4. Checkpoint Hashes & Preservation
- `genesis_17m_stage_e_best.pt`: `00c15185085ab8ab2d729a5fb5ed81434412eaf57e5e6b8633fc864edda9f02c` (68,292,561 bytes)
- `genesis_17m_stage_e_final.pt`: `fbddb8c75b3c754e1cf2f27622e92e5525b84f104110db048947bb89f2d2d36b` (68,292,659 bytes)
