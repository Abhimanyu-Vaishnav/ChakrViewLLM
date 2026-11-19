# UCIP-13: Training Configuration Specification

**Mission ID:** UCIP-13 — ChakrMicro Foundational Neural Pretraining  
**Architecture:** Frozen ChakrMicro v0.1  
**Execution Script:** `experiments/ucip13/scripts/train_stage_c.py`  

---

## 1. Preregistered Hyperparameters

| Hyperparameter | Value | Description |
|---|---|---|
| **Architecture** | ChakrMicro v0.1 | $L=6, H=6, d_{\text{model}}=192, d_{\text{ff}}=512, V=4096$ |
| **Parameters** | 3,443,136 | Tied input/output embeddings, RoPE attention |
| **Initialization Seed** | 42 | SHA-256: `c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da` |
| **Optimizer** | AdamW | $\beta_1=0.9, \beta_2=0.95, \epsilon=10^{-8}$ |
| **Weight Decay** | 0.01 | Decoupled weight decay on non-bias/norm parameters |
| **Peak Learning Rate** | $5.0 \times 10^{-4}$ | Reached at step 40 |
| **Min Learning Rate** | $5.0 \times 10^{-5}$ | Reached at step 800 |
| **Scheduler** | Cosine Decay | Linear warmup (40 steps) + Cosine decay to 0.1 ratio |
| **Gradient Clipping** | 1.0 | Norm clipping on global parameter gradients |
| **Sequence Length ($T$)**| 512 tokens | Context window |
| **Batch Size** | 16 sequences | 8,192 tokens per optimizer step |
| **Planned Steps** | 800 steps | 6,553,600 total token exposures |
| **Loss Function** | Causal Cross-Entropy | Ignores PAD ($2$) |
| **Training Corpus** | Stage C Train | 27 shards (`data/tokenized/stage_c/train/*.bin`) |
| **Validation Corpus** | Stage C Validation | 2 shards (`data/tokenized/stage_c/validation/*.bin`) |
| **Held-Out Test** | Stage C Test | 3 shards (`data/tokenized/stage_c/test/*.bin`) — Untouched |
