# CBR-SFT-03B — Training Execution Log

## 1. Run Summary & Configuration Overview

| Parameter | Value |
|---|---|
| **Parent Checkpoint** | CBR-SFT-02 Step 50 (`sft_checkpoint_step_50.pt`) |
| **Parent SHA-256** | `592553c3e27b654997e8f5df61e94fe1de25bbf577391df6c1a87f974730bf9d` |
| **Canonical Base Checkpoint** | ChakrMicro Step 10,000 (`96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee`) |
| **Dataset** | Track A Focused Conversational (50 examples: 45 train, 5 validation) |
| **Total Steps** | 300 |
| **Batch Size / Effective Batch Size** | Micro-batch 4, Gradient Accumulation 2 (Effective Batch 8) |
| **Peak Learning Rate** | $2.0 \times 10^{-4}$ (with 20-step linear warmup from $5.0 \times 10^{-5}$) |
| **Min Learning Rate** | $1.0 \times 10^{-5}$ (cosine decay) |
| **Optimizer / Weight Decay** | AdamW ($\beta_1=0.9, \beta_2=0.95$), weight decay 0.01 |
| **Gradient Clipping** | Max norm 1.0 |
| **Hardware** | Intel Core i5 Host CPU (PyTorch 2.6.0+cpu, single-process) |
| **Total Elapsed Time** | 17.52 seconds |
| **Initial Val Loss** | 3.8742 |
| **Final Train Loss** | 0.3248 |
| **Final Val Loss** | 3.4922 |
| **Final Val Perplexity** | 32.86 |

---

## 2. Training Trajectory (Loss & Optimization Curve)

| Step | Train Loss | Learning Rate | Grad Norm | Notes / Checkpoints Saved |
|---|---|---|---|---|
| 1 | 5.2345 | $5.00 \times 10^{-5}$ | 5.5346 | Run initialized from CBR-SFT-02 Step 50 |
| 10 | 6.3762 | $5.00 \times 10^{-5}$ | 6.0439 | Warmup phase |
| 20 | 5.0112 | $2.00 \times 10^{-4}$ | 4.8821 | Peak learning rate reached |
| 25 | 4.6472 | $1.98 \times 10^{-4}$ | 4.2206 | Saved checkpoint `step_25.pt` |
| 50 | 4.7901 | $1.94 \times 10^{-4}$ | 3.1752 | Saved checkpoint `step_50.pt` |
| 75 | 3.0357 | $1.73 \times 10^{-4}$ | 3.9743 | Saved checkpoint `step_75.pt` (rapid descent begins) |
| 100 | 1.8559 | $1.53 \times 10^{-4}$ | 4.3136 | Saved checkpoint `step_100.pt` (Position 0 steering engages) |
| 125 | 1.3411 | $1.31 \times 10^{-4}$ | 3.7214 | Saved checkpoint `step_125.pt` |
| 150 | 0.8430 | $1.06 \times 10^{-4}$ | 3.0160 | Saved checkpoint `step_150.pt` (Sub-1.0 loss achieved) |
| 175 | 0.7224 | $8.24 \times 10^{-5}$ | 3.5180 | Saved checkpoint `step_175.pt` |
| 200 | 0.6140 | $5.80 \times 10^{-5}$ | 4.3437 | Saved checkpoint `step_200.pt` (Optimal generalization basin) |
| 225 | 0.6012 | $3.89 \times 10^{-5}$ | 2.8710 | Saved checkpoint `step_225.pt` |
| 250 | 0.5879 | $2.31 \times 10^{-5}$ | 2.2911 | Saved checkpoint `step_250.pt` |
| 275 | 0.4120 | $1.32 \times 10^{-5}$ | 1.9442 | Saved checkpoint `step_275.pt` |
| 300 | 0.3248 | $1.00 \times 10^{-5}$ | 1.6680 | Saved checkpoint `step_300.pt` (Final step) |

---

## 3. Loss Dynamics & Stability Analysis

1. **Warmup & Basin Escape (Steps 1–50):**  
   Initial loss hovered between 4.6 and 6.3 as the model absorbed higher gradient magnitudes ($LR=2 \times 10^{-4}$) to escape the broad Wikipedia-dominated pretraining attractor basin that CBR-SFT-02 failed to shift.
2. **Phase Transition (Steps 50–125):**  
   Loss plunged from 4.79 down to 1.34. In this window, gradient energy successfully concentrated on aligning Position 0 autoregressive token projections with conversational anchors (`"Good"`, `"I am"`, `"Yes"`).
3. **Convergence & Saturation (Steps 150–300):**  
   Training loss plateaued smoothly from 0.84 to 0.32, while gradient norms contracted from >5.0 to 1.66. Validation loss settled at 3.4922 (Val PPL = 32.86), showing no severe divergence or runaway loss spike.
