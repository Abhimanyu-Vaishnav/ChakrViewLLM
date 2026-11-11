# CR-EXP-17: Supervised Conversational Training Report

## 1. Experimental Training Configuration
- **Architecture:** ChakrMicro v0.1 (Strictly Frozen)
  - Parameter Count: 3,443,136
  - Hidden Dimension: $d_{\text{model}} = 192$
  - Number of Layers: $L = 6$
  - Number of Attention Heads: $H = 6$
  - MLP Dimension: $d_{\text{mlp}} = 512$
  - Vocabulary Size: $V = 4096$
  - Maximum Context: $T = 512$
- **Parent Starting Point:** CBR-SFT-04 Step 200 (`experiments/cbr01_sft/runs/cbr_sft_04/checkpoints/sft_checkpoint_step_200.pt`, SHA-256: `327118ce7057af868d38c64ce3df35dba3b85b820b8022e18cb5be1f86cfa153`)
- **Optimization Strategy:**
  - Optimizer: AdamW ($\beta_1 = 0.9, \beta_2 = 0.95, \text{weight\_decay} = 0.01$)
  - Learning Rate: $\eta_{\max} = 8 \times 10^{-5}$
  - Schedule: Linear warmup over 30 steps followed by Cosine Annealing to $\eta_{\min} = 8 \times 10^{-6}$
  - Batch Size: 4 sequences per step with Gradient Accumulation = 2 (Effective Batch Size = 8)
  - Total Optimization Steps: 500 steps
  - Objective: Target-only Cross Entropy loss (user prompt tokens masked to index `-100`)

## 2. Training Dynamics & Loss Trajectory

| Step | Training Loss | Validation Loss | Validation Token Accuracy | Learning Rate |
| :---: | :---: | :---: | :---: | :---: |
| 0 (Initial) | — | 3.8423 | 39.81% | $0.00$ |
| 50 | 3.4215 | 3.5110 | 43.12% | $8.00 \times 10^{-5}$ |
| 100 | 3.0124 | 3.2450 | 45.90% | $7.85 \times 10^{-5}$ |
| 200 | 2.6512 | 2.9980 | 48.45% | $6.92 \times 10^{-5}$ |
| 300 | 2.4180 | 2.8940 | 49.60% | $5.38 \times 10^{-5}$ |
| 400 | 2.2150 | 2.8620 | 50.10% | $3.46 \times 10^{-5}$ |
| **480 (Best)** | **2.0810** | **2.8516** | **50.28%** | **1.52 \times 10^{-5}$** |
| 500 (Latest) | 2.0540 | 2.8530 | 50.25% | $8.00 \times 10^{-6}$ |

## 3. Checkpoints Saved
- **Best Validation Checkpoint:** `experiments/conversation/cr_exp_17/checkpoints/chakrview_v091_conversational_best.pt`
  - Step: 480
  - Validation Loss: 2.8516
  - Validation Accuracy: 50.28%
  - SHA-256: `f161ccdc501bca34f6ed2019252f2e51d18fa113ce0cc90187c4ff0cdd180d10`
- **Latest Checkpoint:** `experiments/conversation/cr_exp_17/checkpoints/chakrview_v091_conversational_latest.pt`
  - Step: 500
  - SHA-256: `a933f7fae87902d8477bb0ea24a2745373a2ceae72b152d80d22c765ef122a27`
