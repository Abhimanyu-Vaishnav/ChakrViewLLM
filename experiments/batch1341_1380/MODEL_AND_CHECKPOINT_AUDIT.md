# Model and Checkpoint Audit

## Batches 1341–1380: ChakrMicro Forward Path, Architecture Configuration and Checkpoint Lineage

### 1. Model Identity and Source Path

- **Model Class:** `ChakrMicro`
- **Source File:** [chakrview/brain/model.py](file:///D:/Project/ChakrView/chakrview/brain/model.py)
- **Configuration Class:** `ModelConfig` ([chakrview/brain/config.py](file:///D:/Project/ChakrView/chakrview/brain/config.py))
- **Model Type:** Foundational indigenous decoder-only causal Transformer.

---

### 2. Architecture Specifications & Hyperparameters

| Hyperparameter | Value | Description |
|---|---|---|
| **Vocabulary Size (`vocab_size`)** | 4,096 | Discrete token vocabulary |
| **Hidden Dimension (`d_model`)** | 192 | Representation dimension across layers |
| **Transformer Layers (`n_layers`)** | 6 | Number of stacked decoder blocks |
| **Attention Heads (`n_heads`)** | 6 | Multi-head self-attention query heads |
| **Head Dimension (`head_dim`)** | 32 | $192 / 6 = 32$ (even for RoPE) |
| **Intermediate FFN Dim (`hidden_dim`)**| 512 | SwiGLU projection dimension ($(8/3) \times 192$) |
| **Context Window (`max_seq_len`)** | 512 | Maximum sequence length supported |
| **Position Embeddings** | Rotary (RoPE) | Base frequency $\theta = 10,000.0$ |
| **Normalization** | RMSNorm | Pre-normalization with $\epsilon = 10^{-5}$ |
| **Linear Layer Bias** | False | No additive biases on linear projections |
| **Embedding Weight Tying** | True | `lm_head.weight` is tied to `embedding.weight` |
| **Precision / Dtype** | `float32` | Standard 32-bit floating point execution |

#### Parameter Counts:
- **Total Registered Parameters in State Dict:** 4,229,568 parameters.
- **Unique Parameters with Weight Tying:** 3,443,136 parameters.
  - Non-embedding weights: $3,443,136 - (4096 \times 192) = 2,656,512$ parameters.
  - Embedding / LM head weight: $4096 \times 192 = 786,432$ parameters.

---

### 3. Checkpoint Provenance and Verification

- **Checkpoint File:** `experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt`
- **Canonical SHA-256:** `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee`
- **Status:** **VERIFIED INTACT**
- **Checkpoint Metadata:**
  - `step`: 10,000
  - `seed`: 42
  - `config`: `ModelConfig(vocab_size=4096, d_model=192, n_layers=6, n_heads=6, max_seq_len=512, hidden_dim=512, tie_embeddings=True)`
  - `state_dict` keys: 57 tensor keys. Strict load succeeded with 0 missing keys and 0 unexpected keys.

---

### 4. Forward Pass and Causal Verification

The model's forward path was audited and verified through unit tests in [tests/test_batch1341_1380_lm_evaluation.py](file:///D:/Project/ChakrView/tests/test_batch1341_1380_lm_evaluation.py):
1. **Tensor Input Shape:** `[B, T]` with integer token IDs in `[0, 4095]`.
2. **Tensor Output Shape:** `[B, T, 4096]` producing unnormalized next-token logits.
3. **Causal Masking:** Transformer blocks implement lower-triangular causal attention masking with RoPE rotation. Confirmed that altering token $x_T$ produces zero perturbation in logits at positions $t < T$ (maximum difference = 0.0).
4. **Context Window Boundary:** Validated context sequences up to 512 tokens. Sequences exceeding 512 raise `ValueError`.
