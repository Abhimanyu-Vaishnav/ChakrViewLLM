# ChakrView Step 4.1: Exact Programmatic Parameter Accounting

**Document Version**: 1.0.0  
**Phase**: Step 4.1 — Exact Parameter Count  
**Model Target**: Chakr-Micro v0.1  
**Status**: VERIFIED BIT-FOR-BIT  

---

## 1. Parameter Breakdown

Extracted programmatically from the model tensors:

| Component | Dimensions / Formula | Per-Layer Count | Total Count (6 Layers) | Trainable |
| :--- | :--- | :---: | :---: | :---: |
| **1. Token Embedding ($E$)** | $V \times d_{\text{model}} = 4096 \times 192$ | — | **$786,432$** | Yes |
| **2. Attention Sub-layers** | $W_Q, W_K, W_V, W_O \in \mathbb{R}^{192 \times 192}$ | $147,456$ | **$884,736$** | Yes |
| ├── $W_Q$ Projection | $192 \times 192$ | $36,864$ | $221,184$ | Yes |
| ├── $W_K$ Projection | $192 \times 192$ | $36,864$ | $221,184$ | Yes |
| ├── $W_V$ Projection | $192 \times 192$ | $36,864$ | $221,184$ | Yes |
| └── $W_O$ Out Projection | $192 \times 192$ | $36,864$ | $221,184$ | Yes |
| **3. Feed-Forward Sub-layers (SwiGLU)** | $W_{\text{gate}}, W_{\text{up}} \in \mathbb{R}^{192 \times 512}, W_{\text{down}} \in \mathbb{R}^{512 \times 192}$ | $294,912$ | **$1,769,472$** | Yes |
| ├── $W_{\text{gate}}$ Projection | $192 \times 512$ | $98,304$ | $589,824$ | Yes |
| ├── $W_{\text{up}}$ Projection | $192 \times 512$ | $98,304$ | $589,824$ | Yes |
| └── $W_{\text{down}}$ Projection | $512 \times 192$ | $98,304$ | $589,824$ | Yes |
| **4. RMSNorm Layer Scales** | $\text{Norm}_1, \text{Norm}_2 \in \mathbb{R}^{192}$ | $384$ | **$2,304$** | Yes |
| **5. Final Normalization Scale** | $\text{Norm}_{\text{final}} \in \mathbb{R}^{192}$ | — | **$192$** | Yes |
| **6. Output-Head Parameters** | Tied with $E$ ($W_{\text{out}} \equiv E^T$) | — | **$0$** (Unique) | Shared |
| **7. TOTAL UNIQUE PARAMETERS** | $786,432 + 884,736 + 1,769,472 + 2,304 + 192$ | — | **$3,443,136$** | — |
| **8. Trainable Parameters** | Parameters with `requires_grad=True` | — | **$3,443,136$** | — |
| **9. Non-Trainable Parameters** | Buffers (`cos_cache`, `sin_cache`, `causal_mask`) | — | **$0$** | — |

---

## 2. Weight Tying Physical & Object Identity

- **Physical Storage Equality**:
  ```python
  assert model.lm_head.weight.data_ptr() == model.embedding.weight.data_ptr()
  ```
- **Object Identity Equality**:
  ```python
  assert model.lm_head.weight is model.embedding.weight
  ```
- **Storage Deduplication**:
  The output projection reuses the exact underlying transposed embedding matrix $E^T$ in `F.linear(x, self.weight)`, preventing the allocation of an independent $786,432$-parameter duplicate tensor.

---
*End of Exact Parameter Accounting Document*
