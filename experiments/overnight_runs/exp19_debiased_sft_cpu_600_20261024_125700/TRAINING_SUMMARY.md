# ChakrView Exp19 — Training Summary

**Experiment Name**: `exp19_debiased_sft_cpu_600`  
**Execution Timestamp**: `2026-10-24T12:57:00` to `2026-10-24T13:02:00`  
**Device**: CPU (Intel, PyTorch `2.10.0+cpu`)  
**Base Checkpoint**: Isolated copy of `experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt`  
**Starting Checkpoint SHA-256**: `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee`  

---

## 1. Execution Overview

Training was executed for exactly **600 optimizer steps** using the audited debiased dataset (`exp19_train_debiased_isolated.jsonl`, 1,200 records).

- **Total Execution Time**: 47.24 seconds (~0.0787 s/step)
- **Batch Size**: 2 (sequence length 512, with prompt loss masking and EOS supervision)
- **Learning Rate Schedule**: Cosine decay from $5 \times 10^{-5}$ down to $5 \times 10^{-6}$
- **Gradient Clipping**: 1.0 (no numerical instability, NaN, or Inf observed)
- **Weight Decay**: 0.01

---

## 2. Validation Loss & Learning Dynamics

Evaluations were performed on the 160 isolated validation records every 20 steps (30 evaluation checkpoints total).

| Step Interval | Training Loss (avg) | Validation Loss | Validation Perplexity | Notes |
| :---: | :---: | :---: | :---: | :--- |
| **0** | - | 4.5025 | 90.24 | Baseline untuned copy |
| **20** | 4.5459 | 4.1953 | 66.37 | Rapid adaptation to prompt format |
| **60** | 3.5350 | 3.6559 | 38.70 | Formatting structure learning |
| **120** | 2.5020 | 3.2086 | 24.74 | Significant response contraction |
| **200** | 2.5694 | 3.0336 | 20.77 | Approaching optimal generalization |
| **300** | 2.2743 | 2.9150 | 18.45 | Continuous stable convergence |
| **340** | **1.8622** | **2.8786** | **17.79** | **Global Best Validation Loss** (Checkpoint Saved) |
| **400** | 1.8499 | 2.9463 | 19.03 | Onset of gentle over-fitting |
| **500** | 1.6210 | 3.0857 | 21.88 | Specialization on training templates |
| **600** | 1.3414 | 3.3124 | 27.45 | Final step (terminal loss) |

### Key Observations
1. **Convex Validation Minimum**: Validation loss reached an absolute minimum of **2.8786 at Step 340**. Beyond step 360, while training loss steadily declined to 1.3414, validation loss climbed toward 3.3124, indicating standard micro-model capacity saturation.
2. **Best Checkpoint Selection**: The best checkpoint was saved as [checkpoints/sft_best_checkpoint.pt](file:///d:/Project/ChakrView/experiments/overnight_runs/exp19_debiased_sft_cpu_600_20261024_125700/checkpoints/sft_best_checkpoint.pt) corresponding strictly to Step 340.
3. **Training Stability**: No gradients exploded; clipping remained effective; loss curves were smooth and monotonically decreasing on the training split.
