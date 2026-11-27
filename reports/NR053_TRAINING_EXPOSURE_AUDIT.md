# NR-053: Training Exposure Reconstruction Audit Report

**Mission:** CHAKRVIEW NR-053 — Generalization, Data Leakage & Overfitting Forensic Validation  
**Date:** November 27, 2026  
**Auditor:** ChakrGenesis Independent Forensic Review Team  
**Artifact Path:** [training_exposure_reconstruction.json](file:///D:/Project/ChakrView/experiments/forensics_nr053/training_exposure_reconstruction.json)

---

## 1. Executive Summary

This audit reconstructs the exact training token exposure across **NR-051** and **NR-052** campaigns by auditing data loaders, chunking mechanisms, step counts, batch sizes, and optimizer updates.

### Key Forensic Findings
1. **Nominal Corpus Size vs Optimizer Token Exposure:**
   - In **NR-051 Pretraining**, the nominal corpus size was 1,026,382 tokens. However, the optimizer executed only 529 steps at effective batch size 32 (max length 256), consuming **540,336 active tokens** (~0.53 effective epochs). Half of the nominal dataset was never processed by the optimizer.
   - In **NR-052 Pretraining**, the nominal training split was 592,824 tokens. The optimizer executed 358 steps over 2 full epochs, consuming **1,181,206 active tokens** (1.99 effective epochs).
2. **Conversational SFT Exposure:**
   - Both NR-051 and NR-052 conversational SFT phases used `data/corpus_v3/corpus_v3_train.jsonl` (4,075 samples) with batch size 16 across 2 epochs (510 total steps).
   - In both campaigns, prompt tokens and padding positions were strictly masked with `ignore_index=-100`. Only intended response tokens contributed to gradient updates.

---

## 2. Reconstructed Model Exposure Ledger

| Campaign / Model | Stage | Unique Docs | Nominal Tokens | Consumed Tokens | Steps | Effective Passes | Batch Size | Learning Rate Schedule |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **NR-051 17M Base** | Pretraining | 8,450 | 1,026,382 | 540,336 | 529 | 0.53x | 32 | Cosine decay ($5 \times 10^{-4} \to 5 \times 10^{-5}$) |
| **NR-051 38M Base** | Pretraining | 8,450 | 1,026,382 | 540,336 | 529 | 0.53x | 32 | Cosine decay ($5 \times 10^{-4} \to 5 \times 10^{-5}$) |
| **NR-051 17M SFT** | SFT (Resp Mask) | 4,075 | 301,488 | ~450,000 | 510 | 2.00x | 16 | Cosine decay ($1 \times 10^{-4} \to 1 \times 10^{-5}$) |
| **NR-051 38M SFT** | SFT (Resp Mask) | 4,075 | 301,488 | ~450,000 | 510 | 2.00x | 16 | Cosine decay ($1 \times 10^{-4} \to 1 \times 10^{-5}$) |
| **NR-052 17M Base** | Pretraining | 5,828 | 592,824 | 1,181,206 | 358 | 1.99x | 32 | Cosine decay ($5 \times 10^{-4} \to 5 \times 10^{-5}$) |
| **NR-052 38M Base** | Pretraining | 5,828 | 592,824 | 1,181,206 | 358 | 1.99x | 32 | Cosine decay ($5 \times 10^{-4} \to 5 \times 10^{-5}$) |
| **NR-052 17M SFT** | SFT (Resp Mask) | 4,075 | 301,488 | ~450,000 | 510 | 2.00x | 16 | Cosine decay ($1 \times 10^{-4} \to 1 \times 10^{-5}$) |
| **NR-052 38M SFT** | SFT (Resp Mask) | 4,075 | 301,488 | ~450,000 | 510 | 2.00x | 16 | Cosine decay ($1 \times 10^{-4} \to 1 \times 10^{-5}$) |

---

## 3. Checkpoint Selection & Early Stopping

Across both campaigns, checkpoints were saved only when validation loss improved over active tokens. In pretraining, validation occurred every 75–100 steps. In SFT, validation occurred every 50 steps. Neither campaign used validation splits for gradient updates.
