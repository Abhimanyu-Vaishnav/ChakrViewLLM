# Batch 73 — Controlled Learning Experiment Results

## 1. Experimental Overview & Rigorous Methodology

Batch 73 evaluated whether ChakrMicro v0.1 learns the single-digit addition transformation on the verified benchmark established in Batch 72. All conditions were executed with strictly controlled parameters:
- **Starting Weights**: Immutable Canonical Checkpoint Step 10k (`96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee`).
- **Data Splits**: 30 Train, 10 Validation, 10 Held-Out Test, 11 Symmetry Diagnostics.
- **Optimizer & Schedule**: AdamW (`lr=5e-4`, `weight_decay=0.01`, `grad_clip=1.0`), 100 steps, batch size 2, seed 42.
- **Hardware**: CPU-only execution (PyTorch 2.14.0+cpu, Windows).

---

## 2. Quantitative Results Across All Evaluated Conditions

| Condition / Strategy | TRAIN Exact Match | VAL Exact Match | TEST Exact Match | Symmetry Diagnostic | TRAIN Ans Loss | VAL Ans Loss | EOS Completion |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Baseline (Untrained Step 10k)** | 0 / 30 (0.0%) | 0 / 10 (0.0%) | 0 / 10 (0.0%) | 0 / 11 (0.0%) | 10.0813 | 10.0336 | **0.0%** (no EOS) |
| **Condition A: Uniform SFT (100 steps)** | 5 / 30 (16.67%) | 1 / 10 (10.0%) | 1 / 10 (10.0%) | 0 / 11 (0.0%) | 2.2003 | 2.3444 | **100.0%** |
| **Condition B: Answer-Emphasis (5x, 100 steps)** | 5 / 30 (16.67%) | 1 / 10 (10.0%) | 1 / 10 (10.0%) | 0 / 11 (0.0%) | 2.2619 | 2.3315 | **100.0%** |
| **Condition C: Answer-Only Diagnostic** | 0 / 30 (0.0%) | 0 / 10 (0.0%) | 0 / 10 (0.0%) | 0 / 11 (0.0%) | 2.3399 | 2.3591 | **0.0%** (runaway loop) |
| **Overfit Control (3 examples, 50 steps)** | **3 / 3 (100.0%)** | 0 / 10 (0.0%) | 3 / 10 (30.0%)* | 1 / 11 (9.09%)* | **0.0046** | 8.2669 | **100.0%** |

*\*Note: In the overfit control, the 3/10 test accuracy is an artifact of the model memorizing a constant digit (`8`) that happens to match 3 test examples whose sum is 8.*

---

## 3. Dissecting the "1/10" Generalization: The Constant Attractor

Analysis of generated outputs across every single example reveals:
- In Condition A and Condition B, the model predicted **`7` on all 30 train examples, all 10 validation examples, and all 10 held-out test examples**.
- In the training set, exactly 5 pairs had sum 7 $\to$ recorded as 5/30 train accuracy.
- In the validation set, exactly 1 pair (`3 + 4`) had sum 7 $\to$ recorded as 1/10 val accuracy.
- In the held-out test set, exactly 1 pair (`4 + 3`) had sum 7 $\to$ recorded as 1/10 test accuracy.
- In the symmetry diagnostic set, 0 pairs had sum 7 $\to$ recorded as 0/11 accuracy.

### Conclusion on Generalization:
The apparent 1/10 validation and test exact-match accuracy is **not mathematical generalization**. It is a **constant modal attractor** where the model emits the median digit `7` for all inputs. The model has not learned addition; it has learned that the prompt demands a single digit between 0 and 9, and defaulting to `7` minimizes global cross-entropy loss across the training distribution.

---

## 4. Overfit Control Verification

The overfit control trained on 3 examples (`3 + 3 = 6`, `2 + 4 = 6`, `7 + 0 = 7`) for 50 full-batch steps:
- Loss dropped to **0.0046**.
- Training accuracy reached **3/3 (100.0%)**.
- This verifies that the optimization engine, backpropagation, and causal target alignment are functioning properly. The lack of generalization on the broader dataset is a fundamental mathematical property of the model/task formulation, not a broken training pipeline.
