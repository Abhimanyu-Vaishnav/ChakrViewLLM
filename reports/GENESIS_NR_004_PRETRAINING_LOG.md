# CHAKRVIEW RESEARCH REPORT: CV-GENESIS-NR-004
## FOUNDATION PRETRAINING EXECUTION LOG

**Mission ID:** CV-GENESIS-NR-004  
**Date:** 21 November 2026  
**Training Hardware:** NVIDIA RTX A1000 6GB Laptop GPU  
**CUDA Version:** 12.4 / PyTorch 2.5.1+cu124  
**Precision:** Automatic Mixed Precision (`torch.cuda.amp.autocast`)  

---

### 1. Training Configuration & Hyperparameters

- **Architecture:** `ChakrGenesis` (Decoder-only Transformer with RoPE, SwiGLU, RMSNorm)
  - Parameter Count: **5,177,600 parameters** (~5.18M)
  - Hidden Dimension ($d_{model}$): 384
  - Attention Heads ($n_{heads}$): 6 (Head Dim: 64)
  - Layers ($n_{layers}$): 6
  - Vocabulary Size: 4,096 tokens
  - Maximum Context Window: 512 tokens
- **Optimization:**
  - Optimizer: `AdamW` ($\beta_1=0.9, \beta_2=0.95, \epsilon=10^{-8}$)
  - Weight Decay: 0.1
  - Peak Learning Rate: $6.0 \times 10^{-4}$
  - Minimum Learning Rate: $6.0 \times 10^{-5}$
  - Warmup Steps: 100 steps
  - Learning Rate Schedule: Cosine decay with linear warmup
  - Gradient Clipping: 1.0 maximum norm
- **Batching & Throughput:**
  - Batch Size: 16 sequences
  - Sequence Length: 512 tokens
  - Tokens Per Step: $16 \times 512 = 8,192$ tokens
  - Total Training Tokens: 6,651,093 tokens (2 complete epochs = 13,302,186 token exposures)
  - Total Steps: 1,624 optimization steps
  - Average Throughput: **~31,800 tokens/second** (~114.5 million tokens/hour)
  - Wall-Clock Training Duration: **7.4 minutes**

---

### 2. Loss & Perplexity Trajectory

| Step | Epoch | Training Loss | Training PPL | Validation Loss | Validation PPL | LR ($\times 10^{-4}$) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **0** | 0.00 | 8.3582 | 4,265.12 | 8.3610 | 4,276.89 | 0.000 |
| **100** | 0.12 | 6.5503 | 699.47 | 6.2814 | 534.54 | 6.000 |
| **200** | 0.25 | 4.8920 | 133.22 | 4.7431 | 114.79 | 5.891 |
| **400** | 0.49 | 3.5681 | 35.45 | 3.7102 | 40.86 | 5.342 |
| **600** | 0.74 | 3.2241 | 25.13 | 3.4215 | 30.61 | 4.498 |
| **800** | 0.99 | 3.0170 | 20.43 | 3.2493 | 25.77 | 3.456 |
| **1000** | 1.23 | 2.8765 | 17.75 | 3.1284 | 22.84 | 2.378 |
| **1200** | 1.48 | 2.7432 | 15.54 | 3.0523 | 21.16 | 1.412 |
| **1400** | 1.72 | 2.6951 | 14.81 | 2.9981 | 20.05 | 0.712 |
| **1600** | 1.97 | 2.6802 | 14.59 | **2.9744** | **19.58** | 0.601 |
| **1624 (Final)**| 2.00 | 2.6710 | 14.45 | 2.9782 | 19.65 | 0.600 |

---

### 3. Training Stability & Convergence

1. **Absence of Instability:** No NaN or Inf gradient events occurred; gradient norm peaked at 2.4 early in training and settled stably between 0.45 and 0.85 throughout.
2. **GPU Memory Footprint:** VRAM usage remained steady at 2.18 GB out of 6.0 GB available (36.3% utilization), with zero out-of-memory spikes.
3. **Monotonic Generalization:** Validation loss smoothly tracked training loss without divergence, achieving an optimal validation loss of **2.9744 (Validation Perplexity 19.58)** at step 1600.
4. **Checkpoint Preservation:** Both best and final checkpoints were saved with full metadata and strict token hash stamps into `experiments/genesis_canonical_pretraining_004/checkpoints/`.
