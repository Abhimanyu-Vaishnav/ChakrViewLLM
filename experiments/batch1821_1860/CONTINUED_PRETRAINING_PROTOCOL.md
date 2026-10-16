# CONTINUED PRETRAINING PROTOCOL
## Batches 1821–1860: Controlled Experimental Protocol & Hyperparameter Specifications

### 1. Predeclared Protocol & Design Safeguards

To prevent post-hoc rationalization or data leakage, the continued pre-training protocol was predeclared and frozen prior to model execution:

- **Base Checkpoint**: Canonical ChakrMicro Step 10,000 (`training_checkpoint_step_10000.pt`, SHA-256: `96357705...`).
- **Training Objective**: Standard Autoregressive Causal Next-Token Prediction with Cross-Entropy Loss:
  $$L(\theta) = -\sum_{t=1}^T \log P_\theta(x_t \mid x_{<t})$$
- **Training Corpus**: `experiments/batch1821_1860/data/train_corpus.json` (30 documents across 6 balanced domains).
- **Validation Corpus**: `experiments/batch1821_1860/data/val_corpus.json` (12 documents).
- **Held-Out Test Corpus**: `experiments/batch1821_1860/data/test_held_out_corpus.json` (18 documents).
- **Data-Matched Control (Candidate C)**: Document-order shuffled version of the exact training corpus to isolate sequencing effects.

---

### 2. Hyperparameter Configuration

| Hyperparameter | Value | Rationale |
| :--- | :---: | :--- |
| **Optimizer** | AdamW | Standard decoupled weight-decay optimizer. |
| **Learning Rate** | $2 \times 10^{-5}$ | Conservative learning rate to preserve foundational representations and prevent catastrophic forgetting. |
| **Weight Decay** | $0.01$ | Standard L2 regularization on non-norm weights. |
| **Batch Size** | 4 sequences | Balanced mini-batch suited for CPU memory and cache locality. |
| **Max Sequence Crop** | 128 tokens | Covers full document sentences while matching typical training spans. |
| **Total Optimization Steps** | 120 steps | Approximately 1.5 epochs over the curated multi-domain corpus. |
| **Gradient Accumulation** | 1 | Direct step updates. |
| **Random Seeds** | 42, 123, 999 | Standard 3-seed protocol for empirical variance estimation. |
| **Hardware Platform** | Intel x86_64 CPU | Zero external API or GPU acceleration dependencies. |

---

### 3. Predeclared Success Criteria

For continued pre-training to be judged a scientifically valid improvement:
1. **Mean Cross-Entropy Loss Reduction**: $\Delta \text{Loss} \le -0.10$ vs. Canonical Baseline ($5.3578$).
2. **Document Consistency**: Loss must improve across at least 4 of the 6 evaluated domains (not concentrated in an outlier document).
3. **Multi-Seed Stability**: Standard deviation across the 3 seeds must not exceed $\pm 0.05$.
4. **Zero Checkpoint Corruption**: Checkpoints must load with identical strict parameter bindings.
