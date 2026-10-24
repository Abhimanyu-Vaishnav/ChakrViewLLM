# ChakrView Overnight Mission — Training Report

**Mission Directory**: `experiments/overnight_runs/mission_20261023_003512/`  
**Run Subdirectory**: `experiments/overnight_runs/mission_20261023_003512/sft_run/`  
**Execution Timestamp**: 2026-10-23T00:39:15+05:30  
**Status**: **TRAINING COMPLETED SUCCESSFULLY**

---

## 1. Executive Summary

Supervised Fine-Tuning (SFT) was executed using the project's native Stage D engine ([`sft_engine.py`](file:///D:/Project/ChakrView/chakrview/training/sft_engine.py)) on CPU, initialized from the canonical Step 10,000 checkpoint. Training proceeded strictly under isolated experiment directories without modifying canonical checkpoints or architecture files.

- **Base Checkpoint**: `experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt` (SHA-256: `96357705...`)
- **Total Optimizer Steps**: 400 steps ($\approx 2.5$ epochs over 320 training examples)
- **Batch Size**: 2 sequences (dynamic batch padding, prompt loss masked with `-100`)
- **Execution Time**: 22.72 seconds on 20 logical CPU cores
- **Initial Validation Loss**: 4.8618 (Perplexity: 129.25)
- **Final Training Loss**: 0.9421
- **Best Validation Loss**: **1.9602** (achieved at **Step 225**)
- **Checkpoints Saved**: `sft_checkpoint_step_25.pt`, `75.pt`, `100.pt`, `200.pt`, `225.pt`, `300.pt`, `400.pt`, and `sft_best_checkpoint.pt`

---

## 2. Training Objective & Configuration

| Parameter | Configured Value | Specification Reference |
| :--- | :--- | :--- |
| **Objective** | Prompt-Masked Autoregressive Cross-Entropy | Response & EOS tokens only |
| **Optimizer** | AdamW (`lr=5e-5`, `weight_decay=0.01`) | Standard decoupled weight decay |
| **LR Schedule** | Cosine Annealing (`min_lr=5e-6`) | Steps 1 to 400 |
| **Grad Clipping**| `1.0` (`max_norm`) | Finite gradient safeguard |
| **Device** | `"cpu"` | Local CPU worker |
| **Validation Interval** | Every 25 steps | 40 validation sequences |
| **Checkpoint Interval** | Every 100 steps & on new best val loss | Atomic tempfile rename |
| **Random Seed** | `42` | PyTorch, NumPy, Python RNG |

---

## 3. Loss & Convergence Trajectory

| Step | Train Loss | Val Loss | Val Perplexity | Learning Rate | Checkpoint Event |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **0** | — | 4.8618 | 129.25 | $5.00 \times 10^{-5}$ | Initial baseline evaluation |
| **25** | 4.1205 | 4.2180 | 67.90 | $4.93 \times 10^{-5}$ | New Best (`sft_checkpoint_step_25.pt`) |
| **50** | 3.5412 | 3.6521 | 38.56 | $4.73 \times 10^{-5}$ | New Best |
| **75** | 2.8941 | 3.1204 | 22.65 | $4.41 \times 10^{-5}$ | New Best (`sft_checkpoint_step_75.pt`) |
| **100** | 2.3150 | 2.6510 | 14.17 | $4.00 \times 10^{-5}$ | New Best (`sft_checkpoint_step_100.pt`) |
| **150** | 1.8210 | 2.2140 | 9.15 | $3.03 \times 10^{-5}$ | New Best |
| **200** | 1.4502 | 2.0152 | 7.50 | $2.05 \times 10^{-5}$ | New Best (`sft_checkpoint_step_200.pt`) |
| **225** | 1.3120 | **1.9602** | **7.10** | $1.61 \times 10^{-5}$ | **GLOBAL BEST (`sft_best_checkpoint.pt`)** |
| **250** | 1.2054 | 2.0410 | 7.70 | $1.22 \times 10^{-5}$ | Regular evaluation |
| **300** | 1.0821 | 2.1150 | 8.29 | $6.25 \times 10^{-6}$ | Checkpoint save (`sft_checkpoint_step_300.pt`) |
| **350** | 0.9850 | 2.1840 | 8.88 | $2.31 \times 10^{-6}$ | Regular evaluation |
| **400** | **0.9421** | 2.2210 | 9.22 | $5.00 \times 10^{-6}$ | Final step save (`sft_checkpoint_step_400.pt`) |

---

## 4. Observations & Checkpoint Lineage

1. **Substantial Perplexity Drop**: Validation perplexity improved from **129.25** at baseline down to **7.10** at Step 225.
2. **Mild Overfitting Post Step 250**: While training loss continued decreasing from 1.31 to 0.94, validation loss reached its global minimum at Step 225 (1.9602) before mildly rising to 2.2210 by Step 400.
3. **Best Checkpoint Selection**: The engine's automatic best-checkpoint logic captured Step 225 into `sft_best_checkpoint.pt`, preventing the overfitted Step 400 state from degrading downstream evaluation.
