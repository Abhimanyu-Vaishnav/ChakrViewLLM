# ChakrView v0.9.0 — Neural Architecture Specification

## 1. Dimensional Hyperparameters

The foundational architecture is **ChakrMicro v0.1**:

```json
{
  "vocab_size": 4096,
  "d_model": 192,
  "n_layers": 6,
  "n_heads": 6,
  "head_dim": 32,
  "hidden_dim": 512,
  "max_seq_len": 512,
  "rms_norm_eps": 1e-05,
  "rope_theta": 10000.0,
  "tie_embeddings": true,
  "use_bias": false,
  "dropout": 0.0,
  "initializer_range": 0.02
}
```

---

## 2. Parameter Accounting

| Component | Dimensions / Shape | Parameter Count |
| :--- | :--- | :---: |
| **Token Embedding** | `[4096, 192]` | 786,432 |
| **Transformer Blocks (6 layers)** | | |
| — Input RMSNorm | `[192]` $\times 6$ | 1,152 |
| — Multi-Head Attention (Q, K, V, Out) | $4 \times [192, 192] \times 6$ | 884,736 |
| — Post-Attention RMSNorm | `[192]` $\times 6$ | 1,152 |
| — SwiGLU MLP ($W_{gate}, W_{up}, W_{down}$)| $(2 \times [192, 512] + [512, 192]) \times 6$ | 1,769,472 |
| **Final RMSNorm** | `[192]` | 192 |
| **LM Head** | Shared with Token Embedding ($W_{out} = E^T$) | 0 *(Tied)* |
| **Total Unique Parameters** | | **3,443,136** |

---

## 3. Structural Innovations & Constraints
- **Pairwise RoPE Rotations:** Even head dimension ($d_{head} = 32$) ensures exact pairwise 2D rotation for relative sequence distance representation up to context length 512.
- **Strict Weight Tying:** Token embedding weights are directly reused as the projection matrix of the LM head, preventing parameter explosion while regularizing token representation space.
- **SwiGLU Activation:** Gated linear units ($Swish(xW_{gate}) \otimes (xW_{up})) W_{down}$ provide expressive non-linear representation capacity within a compact hidden dimension.
