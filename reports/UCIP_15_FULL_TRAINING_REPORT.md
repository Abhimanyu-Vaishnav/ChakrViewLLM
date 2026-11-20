# UCIP-15 Full SFT Training Report

**Mission ID:** UCIP-15  
**Project:** ChakrView  
**Repository:** `D:\Project\ChakrView`  
**Branch:** `exp/cbr-01-conversational-breakthrough`  
**Execution Agent:** Antigravity  
**Execution Timestamp:** 2026-11-20T14:30:30+05:30  
**Training Status:** COMPLETED & AUDITED  

---

## 1. Training Configuration & Lineage

Full Supervised Fine-Tuning was executed following the passing of all 6 mandatory pilot acceptance gates:
- **Starting Checkpoint:** UCIP-13 Step 800 (`checkpoint_step_800.pt`, SHA-256: `8759ccb4...`).
- **Optimization Strategy:** Token-normalized joint mixture tuning (80% SFT on 12k records + 20% continuous Stage C pretraining replay).
- **Optimizer:** AdamW ($\beta_1=0.9, \beta_2=0.95, \text{weight\_decay}=0.01$).
- **Learning Rate:** $2.5 \times 10^{-5}$ peak with 50-step linear warmup and cosine decay to $2.5 \times 10^{-6}$.
- **Execution Target:** 100% CPU execution on Windows 11.
- **Total Optimizer Steps:** 600 steps (duration: 274.4 seconds).
- **Target Directories:** Stored in `experiments/ucip15/full_training/`.

---

## 2. Loss & Optimization Trajectory

| Step Interval | SFT Train Loss | Replay Train Loss | Total Joint Loss | SFT Val Loss | Stage C Retention Val Loss | Learning Rate | Grad Norm |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Step 000 (Init)** | — | — | — | **5.5442** | **3.9544** | $0.00$ | — |
| **Step 100** | 4.9096 | 4.2969 | 4.4684 | 4.7607 | 3.7956 | $2.45 \times 10^{-5}$ | 2.089 |
| **Step 200** | 4.6800 | 4.2675 | 4.3829 | 4.5639 | 3.7399 | $2.11 \times 10^{-5}$ | 2.220 |
| **Step 300** | 4.8266 | 4.5662 | 4.6258 | 4.2346 | 3.7193 | $1.54 \times 10^{-5}$ | 3.714 |
| **Step 400** | 3.7977 | 4.1541 | 4.0684 | 4.0312 | 3.6991 | $9.08 \times 10^{-6}$ | 2.680 |
| **Step 500** | 3.6140 | 3.9985 | 3.8901 | 4.1074 | 3.6919 | $4.29 \times 10^{-6}$ | 6.518 |
| **Step 550 (Best)** | 3.6218 | 3.4903 | 3.5251 | **3.8426** | **3.6887** | $2.96 \times 10^{-6}$ | 1.880 |
| **Step 600 (Final)** | 4.3377 | 4.2068 | 4.2329 | 3.9938 | 3.6889 | $2.50 \times 10^{-6}$ | 2.225 |

- **SFT Validation Loss:** Decreased from **5.5442** to **3.8426** ($-1.7016$ points, **30.7% relative improvement**).
- **Language Retention:** Maintained at **3.6887** (improved by $-0.2657$ points over the 3.9544 baseline). Zero catastrophic forgetting.
- **Selected Best Checkpoint:** `experiments/ucip15/full_training/checkpoint_best.pt` (Step 550, SHA-256: `004ca38f3e796c4590bcfd73a029a3493173637f22bcf0dcad41e98c218511e7`).

---

## 3. Stability & Invariant Summary
- Zero NaN/Inf loss or gradient incidents throughout all 600 optimization steps.
- Model architecture parameters, special tokens (BOS=0, EOS=1, PAD=2), and tied weights remained intact.
- Sealed benchmark quarantined throughout training with zero contamination.
