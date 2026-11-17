# UCIP-09R: Recovery Training & Optimization Report

**Project:** ChakrView  
**Experiment Branch:** `exp/cbr-01-conversational-breakthrough`  
**Run Name:** `ucip09r_natural_recovery`  
**Base Architecture:** ChakrMedium (12,932,736 parameters; 7 layers, 384 dim, 6 heads, 896 ffn)  
**Parent Weights:** `releases/ucip_09_candidates/chakrview_ucip09_medium_model.pt`  
**Date:** November 17, 2026  

---

## 1. Training Setup & Hyperparameters

- **Hardware Execution:** Local CPU-First Execution (Intel x86_64, Windows 11, Float32)
- **Optimizer:** `torch.optim.AdamW`
- **Learning Rate:** $2.5 \times 10^{-5}$
- **Weight Decay:** $0.01$
- **Batch Size:** 4
- **Gradient Clipping:** Max norm $1.0$
- **Loss Function:** Label-Smoothed Cross-Entropy ($\epsilon = 0.03$) with causal next-token shift
- **Dataset Size:** 885 training samples, 98 validation samples (natural prompt pairs, 0% benchmark overlap)
- **Total Steps:** 160 steps

---

## 2. Loss & Progress Trajectory

| Step | Train Loss | Validation Loss | Wall-Clock Time | Checkpoint Status |
|---|---|---|---|---|
| 20 | 2.4194 | 1.7976 | 4.0s | Saved (`candidate_best.pt`) |
| 40 | 1.6897 | 1.4413 | 8.8s | Saved (`candidate_best.pt`) |
| 60 | 1.5553 | 1.4920 | 13.8s | |
| 80 | 1.7052 | 1.4292 | 18.6s | Saved (`candidate_best.pt`) |
| 100 | 1.4098 | 1.4289 | 24.0s | Saved (`candidate_best.pt`) |
| 120 | 1.4510 | 1.3612 | 28.9s | Saved (`candidate_best.pt`) |
| 140 | 1.4186 | 1.3978 | 34.3s | |
| 160 | 1.2866 | **1.2461** | 39.8s | Saved (`candidate_best.pt`) |

*Final Best Validation Loss:* **1.2461** (down from 1.5893).  
*Candidate Checkpoint:* `releases/ucip_09r_candidates/chakrview_ucip09r_medium_model.pt`  
*SHA-256 Digest:* `e1b943222e70fce2bac1b198e656bfea294209de1291f0042a2ebd677ccef9f0`
