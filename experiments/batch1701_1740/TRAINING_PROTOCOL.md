# TRAINING PROTOCOL & HYPERPARAMETER SPECIFICATION
## Batches 1701–1740 Optimization Protocols, Schedulers & Stopping Conditions

### 1. Training Protocol Standards

All training runs in Batches 1701–1740 adhere to strict local CPU reproducibility guidelines:

- **Host Device:** CPU (Intel Core i5-12450HX)
- **Base Checkpoint:** `experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt` (SHA-256: `96357705...`)
- **Random Seed:** `42` (Fixed across PyTorch, NumPy, and Python standard random)
- **Loss Function:** Multi-Class Cross-Entropy with class balance weighting:
  $$\mathcal{L}_{cls} = - \sum_{c=1}^{6} y_c \log \hat{y}_c$$
- **Gradient Clipping:** Max norm $1.0$
- **Checkpoint Isolation:** Checkpoints saved exclusively under `experiments/batch1701_1740/checkpoints/`. Reference canonical weights remain read-only.

---

### 2. Multi-Variant Optimization Hyperparameters

| Hyperparameter | Variant A (Frozen Backbone) | Variant B (Fine-Tuned Backbone) | Data Scaling Protocol |
| :--- | :--- | :--- | :--- |
| **Optimizer** | AdamW | AdamW | AdamW |
| **Backbone Learning Rate** | $0.0$ (Frozen) | $3 \times 10^{-5}$ | $3 \times 10^{-5}$ |
| **Head Learning Rate** | $1 \times 10^{-3}$ | $1 \times 10^{-4}$ | $1 \times 10^{-4}$ |
| **Weight Decay** | $0.01$ | $0.01$ | $0.01$ |
| **Batch Size** | 8 | 4 | 4 |
| **Max Steps** | 100 optimizer steps | 100 optimizer steps | 100 steps (per budget: 25%, 50%, 100%) |
| **Evaluation Interval** | Every 10 steps on `val.jsonl` | Every 10 steps on `val.jsonl` | Every 10 steps on `val.jsonl` |
| **Model Selection Rule** | Lowest validation cross-entropy | Lowest validation cross-entropy | Lowest validation cross-entropy |

---

### 3. Safety and Model Preservation Rules
1. Never evaluate or select models based on held-out test performance.
2. Checkpoint hashes must be re-verified pre- and post-flight.
3. Every evaluated prediction must record the model's uncalibrated softmax probability for confidence calibration audits.
