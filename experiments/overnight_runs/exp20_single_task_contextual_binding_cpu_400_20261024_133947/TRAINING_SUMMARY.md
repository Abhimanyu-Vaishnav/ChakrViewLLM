# ChakrView Exp20 — Training Summary

**Experiment Name**: `exp20_single_task_contextual_binding_cpu_400`  
**Task Family**: Single-Task `context_extraction`  
**Execution Timestamp**: `2026-10-24T13:43:34` to `2026-10-24T13:44:02`  
**Device**: CPU (Intel, PyTorch `2.10.0+cpu`)  
**Base Checkpoint**: Isolated copy of `experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt`  
**Starting Checkpoint SHA-256**: `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee`  

---

## 1. Execution Overview

Training was executed for exactly **400 optimizer steps** using the isolated single-task dataset (`exp20_train_context_extraction.jsonl`, 400 records).

- **Total Execution Time**: 27.32 seconds (~0.0683 s/step)
- **Batch Size**: 2 (sequence length 512, prompt loss masking, EOS supervised)
- **Learning Rate Schedule**: Cosine decay from $5 \times 10^{-5}$ down to $5 \times 10^{-6}$
- **Gradient Clipping**: 1.0 (no numerical instability, NaN, or Inf observed)
- **Weight Decay**: 0.01

---

## 2. Validation Loss & Learning Dynamics

Evaluations were performed on the 50 isolated validation records every 20 steps (20 evaluation intervals).

| Step Interval | Training Loss (avg) | Validation Loss | Validation Perplexity | Notes |
| :---: | :---: | :---: | :---: | :--- |
| **0** | - | 4.4120 | 82.43 | Baseline untuned starting copy |
| **20** | 3.8214 | 3.4152 | 30.42 | Rapid absorption of single-task format |
| **60** | 2.5104 | 2.5510 | 12.82 | Syntactic frame convergence (`X is assigned as the Y`) |
| **120** | 1.8210 | 2.0145 | 7.50 | Strong syntactic stabilization |
| **180** | 1.3540 | 1.8320 | 6.25 | Approaching validation minimum |
| **240** | **1.0215** | **1.7447** | **5.72** | **Global Best Validation Loss** (Checkpoint Saved) |
| **280** | 0.8920 | 1.7820 | 5.94 | Validation loss plateau |
| **340** | 0.6510 | 1.9540 | 7.06 | Over-fitting on 50 training entity names |
| **400** | 0.4120 | 2.3087 | 10.06 | Final step (memorization of training names) |

### Key Observations
1. **Dramatically Lower Validation Loss**: Single-task isolation reduced validation loss to **1.7447** (Perplexity 5.72), compared to 2.8786 in multi-task Exp19. The model achieved near-perfect syntax mastery of the single task.
2. **Best Checkpoint Selection**: The best validation checkpoint was saved as [checkpoints/sft_best_checkpoint.pt](file:///d:/Project/ChakrView/experiments/overnight_runs/exp20_single_task_contextual_binding_cpu_400_20261024_133947/checkpoints/sft_best_checkpoint.pt) corresponding to **Step 240**.
3. **Training Loss Trajectory**: Training loss collapsed to 0.4120, showing complete memorization of the training entity set.
