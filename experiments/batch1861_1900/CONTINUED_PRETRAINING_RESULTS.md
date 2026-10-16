# CONTINUED PRETRAINING RESULTS
## Batches 1861–1900: Data Scaling & Multi-Seed Validation Across 12,479 Held-Out Tokens

### 1. Scaling Benchmark Verification

In Batches 1861–1900, the evaluation set was scaled from 708 tokens to **12,479 valid held-out prediction tokens** across 48 documents.

Continued pretraining was executed across **three independent random seeds (42, 123, 999)** under Condition B (150 steps, AdamW lr=2e-5, wd=0.01) on CPU.

#### Multi-Seed Replication Table:

| Run Identifier | Random Seed | Training Steps | Training Wall Time | Held-Out Loss | Held-Out Perplexity | Top-1 Accuracy | Top-5 Accuracy | Net Delta vs. Canonical |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Canonical Baseline** | — | 0 | 0.0s | **5.7817** | **324.30** | **10.22%** | **23.41%** | Reference |
| **Condition B Run 1** | 42 | 150 | 14.89s | 4.0426 | 56.97 | 30.30% | 46.78% | -1.7391 |
| **Condition B Run 2** | 123 | 150 | 14.52s | 3.9906 | 54.09 | 31.00% | 46.78% | -1.7911 |
| **Condition B Run 3** | 999 | 150 | 14.65s | 3.9869 | 53.89 | 31.00% | 46.78% | -1.7948 |
| **Condition B Summary** | — | **150** | **14.69s** | **4.0067 ± 0.0254** | **54.98** | **30.77% ± 0.40%** | **46.78%** | **-1.7750** |

---

### 2. Checkpoint Serialization & Integrity

Checkpoints generated during this experiment were saved to `experiments/batch1861_1900/checkpoints/` and hashed:

| Checkpoint Name | Step Count | Optimizer LR | Parameter Count | Checkpoint Path |
| :--- | :---: | :---: | :---: | :--- |
| `cpt_balanced_seed_42.pt` | 150 | 2e-5 | 3,443,136 | `experiments/batch1861_1900/checkpoints/cpt_balanced_seed_42.pt` |
| `cpt_balanced_seed_123.pt` | 150 | 2e-5 | 3,443,136 | `experiments/batch1861_1900/checkpoints/cpt_balanced_seed_123.pt` |
| `cpt_balanced_seed_999.pt` | 150 | 2e-5 | 3,443,136 | `experiments/batch1861_1900/checkpoints/cpt_balanced_seed_999.pt` |

All checkpoints load strictly into the canonical architecture without mismatch warnings.
