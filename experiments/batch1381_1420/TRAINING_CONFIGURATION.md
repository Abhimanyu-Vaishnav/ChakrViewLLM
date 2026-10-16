# Training Configuration Specification

## Batches 1381–1420: Controlled Domain Adaptation Runner Specification

### 1. Training Parameters

| Configuration Field | Value | Rationale |
|---|---|---|
| **Base Checkpoint** | `experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt` | Canonical frozen weights (SHA-256: `96357705a6...`) |
| **Target Checkpoint** | `experiments/batch1381_1420/checkpoints/chakrmicro_domain_adapted_experimental.pt` | Dedicated mission checkpoint path |
| **Model Architecture** | `ChakrMicro` (3,443,136 tied parameters) | Standard decoder Transformer ($d=192, L=6, H=6, V=4096$) |
| **Optimizer** | `torch.optim.AdamW` | Standard adaptive gradient optimizer |
| **Peak Learning Rate** | $5.0 \times 10^{-5}$ | Conservative rate to prevent catastrophic representation collapse |
| **Minimum Learning Rate** | $5.0 \times 10^{-6}$ | Minimum learning rate floor at cosine cycle completion |
| **Betas** | $(0.9, 0.95)$ | Standard language modeling momentum parameters |
| **Weight Decay** | $0.05$ | Explicit regularization against parameter drift on small corpus |
| **Warmup Steps** | 10 steps | Linear warmup from $1.0 \times 10^{-6}$ |
| **Schedule** | Cosine Annealing | Smooth decay over 80 steps |
| **Max Steps Ceiling** | 80 steps | ~16 passes over the 20 training documents |
| **Effective Batch Size** | 4 documents per step (~510 target tokens/step) | Stable gradient estimate |
| **Gradient Clipping** | Max norm 1.0 | Safeguard against gradient explosion on sharp subwords |
| **Validation Frequency** | Every 5 steps | Frequent monitoring of dev set loss |
| **Validation Patience** | 3 evaluations (15 steps) | Early stopping if validation loss deteriorates or fails to improve |
| **Random Seed** | 42 | Strict reproducibility of document batch shuffling |
| **Compute Device** | Local CPU | Zero GPU required; execution < 15 seconds |

---

### 2. Validation & Checkpoint Selection Protocol

- The model will be evaluated every 5 steps on the **5 validation documents** (637 target tokens).
- The candidate checkpoint will be saved **only when a new lowest validation loss is achieved**.
- If the run finishes or early stops, the checkpoint corresponding to the **minimum validation loss** is retained for final held-out evaluation against the 27 test documents.
