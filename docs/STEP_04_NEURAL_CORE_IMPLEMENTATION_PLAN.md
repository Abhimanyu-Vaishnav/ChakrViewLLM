# ChakrView Step 4: Neural Core Implementation Plan

**Document Version**: 1.0.0  
**Phase**: Step 4 — Neural Core Prototype Implementation  
**Model Target**: Chakr-Micro v0.1  
**Status**: APPROVED IMPLEMENTATION PLAN  

---

## 1. Objective & Architectural Scope

The goal of this phase is to construct the **first working neural core prototype** for ChakrView:
- An indigenous, from-scratch decoder-only causal language model (`ChakrMicro`).
- Zero reliance on external pretrained weights, HuggingFace model abstractions, or wrapper libraries.
- Strict alignment with all frozen Step 1, Step 2, and Step 3 contracts:
  - $V = 4096$ (provisional ratified vocabulary)
  - $d_{\text{model}} = 192$
  - $N = 6$ layers
  - $H = 6$ heads ($d_{\text{head}} = 32$)
  - $d_{\text{ff}} = 512$ (SwiGLU intermediate dimension)
  - $T_{\text{max}} = 512$
  - Weight tying: $W_{\text{out}} = E^T$
  - Pre-RMSNorm ($\epsilon = 10^{-5}$)
  - Rotary Position Embeddings (RoPE, $\Theta = 10000.0, d_{\text{rot}} = 32$)
  - Strictly bias-free linear projections ($b = 0$)
  - Simple Multi-Head Attention (MHA) for v0.1

---

## 2. Module Boundaries & Package Architecture

The neural core prototype is structured cleanly inside `chakrview/brain/`:

```
chakrview/brain/
├── __init__.py           # Package exports for brain primitives, configs, and models
├── config.py             # ModelConfig dataclass with structural validation
├── embeddings.py         # TokenEmbedding ([B, T] -> [B, T, d_model])
├── normalization.py      # RMSNorm (Pre-RMSNorm with scale gamma, bias-free)
├── rotary.py             # RotaryEmbedding (RoPE frequency cache and pairwise rotation)
├── masking.py            # CausalMask (Strict causal lower-triangular additive mask)
├── attention.py          # MultiHeadAttention (Bias-free QKV/OutProj, MHA, RoPE on Q/K)
├── feedforward.py        # SwiGLU (Gate, Up, Swish activation, Down projection)
├── block.py              # TransformerBlock (Pre-RMSNorm -> MHA -> Res -> Pre-RMSNorm -> FFN -> Res)
├── output.py             # LMHead (Tied projection sharing embedding tensor)
├── model.py              # ChakrMicro (Full end-to-end model pipeline)
└── initialization.py     # Deterministic mathematically defensible weight initializers
```

---

## 3. Mathematical Equations & Forward-Pass Sequence

### 3.1 Mathematical Data Flow
For batch size $B$ and sequence length $T \le 512$:

1. **Token Embedding**:
   $$\mathbf{x}_0 = \mathbf{E}[\mathbf{t}], \quad \mathbf{t} \in \{0, \dots, V-1\}^{B \times T}, \quad \mathbf{E} \in \mathbb{R}^{V \times d_{\text{model}}}$$
   $$\mathbf{x}_0 \in \mathbb{R}^{B \times T \times 192}$$

2. **Transformer Block $l \in \{1, \dots, 6\}$**:
   - **Pre-Attention Normalization**:
     $$\mathbf{u}_l = \text{RMSNorm}_1(\mathbf{x}_{l-1}) = \frac{\mathbf{x}_{l-1}}{\sqrt{\frac{1}{d} \sum_{i=1}^d x_{l-1, i}^2 + \epsilon}} \odot \boldsymbol{\gamma}_{1, l}$$
   - **Projections**:
     $$Q_{\text{flat}} = \mathbf{u}_l W_Q, \quad K_{\text{flat}} = \mathbf{u}_l W_K, \quad V_{\text{flat}} = \mathbf{u}_l W_V$$
     where $W_Q, W_K, W_V \in \mathbb{R}^{192 \times 192}$ (bias-free).
   - **Head Partitioning & RoPE**:
     $$Q, K, V \in \mathbb{R}^{B \times 6 \times T \times 32}$$
     $$\tilde{Q} = \text{RoPE}(Q), \quad \tilde{K} = \text{RoPE}(K), \quad V \text{ unrotated}$$
   - **Scaled Causal Dot-Product Attention**:
     $$\mathbf{S} = \frac{\tilde{Q} \tilde{K}^T}{\sqrt{32}} + M_{\text{causal}}, \quad M_{\text{causal}}(i, j) = \begin{cases} 0 & j \le i \\ -\infty & j > i \end{cases}$$
     $$\mathbf{P} = \text{Softmax}(\mathbf{S}, \text{dim}=-1) \in \mathbb{R}^{B \times 6 \times T \times T}$$
     $$\mathbf{O}_{\text{heads}} = \mathbf{P} V \in \mathbb{R}^{B \times 6 \times T \times 32}$$
   - **Output Projection & First Residual**:
     $$\mathbf{a}_l = \text{Reshape}(\mathbf{O}_{\text{heads}}, (B, T, 192)) W_O, \quad W_O \in \mathbb{R}^{192 \times 192}$$
     $$\mathbf{h}_l = \mathbf{x}_{l-1} + \mathbf{a}_l$$
   - **Pre-FFN Normalization**:
     $$\mathbf{v}_l = \text{RMSNorm}_2(\mathbf{h}_l) \in \mathbb{R}^{B \times T \times 192}$$
   - **SwiGLU Transformation**:
     $$\mathbf{f}_l = \left( \text{Swish}(\mathbf{v}_l W_{\text{gate}}) \odot (\mathbf{v}_l W_{\text{up}}) \right) W_{\text{down}}$$
     where $W_{\text{gate}}, W_{\text{up}} \in \mathbb{R}^{192 \times 512}$ and $W_{\text{down}} \in \mathbb{R}^{512 \times 192}$ (bias-free).
   - **Second Residual**:
     $$\mathbf{x}_l = \mathbf{h}_l + \mathbf{f}_l \in \mathbb{R}^{B \times T \times 192}$$

3. **Final Normalization & Tied LM Head**:
   $$\mathbf{x}_{\text{norm}} = \text{RMSNorm}_{\text{final}}(\mathbf{x}_6) \in \mathbb{R}^{B \times T \times 192}$$
   $$\mathbf{z} = \mathbf{x}_{\text{norm}} \mathbf{E}^T \in \mathbb{R}^{B \times T \times 4096}$$

---

## 4. Parameter Count Formulas

| Component | Dimensions / Formula | Parameters |
| :--- | :--- | :---: |
| **Token Embedding ($E$)** | $V \times d_{\text{model}} = 4096 \times 192$ | $786,432$ |
| **Attention $W_Q, W_K, W_V, W_O$ (per layer)** | $4 \times (192 \times 192)$ | $147,456$ |
| **RMSNorm 1 (per layer)** | $192$ | $192$ |
| **SwiGLU $W_{\text{gate}}, W_{\text{up}}, W_{\text{down}}$ (per layer)** | $3 \times (192 \times 512)$ | $294,912$ |
| **RMSNorm 2 (per layer)** | $192$ | $192$ |
| **Subtotal Per Layer** | $147,456 + 192 + 294,912 + 192$ | **$442,752$** |
| **All $N = 6$ Layers** | $6 \times 442,752$ | **$2,656,512$** |
| **Final RMSNorm** | $192$ | $192$ |
| **Tied LM Head** | $0$ (shares $E$) | $0$ |
| **Total Model Parameters** | $786,432 + 2,656,512 + 192$ | **$3,443,136$** |

---

## 5. Weight Tying Implementation Policy

Weight tying is an absolute structural invariant:
$$\mathbf{W}_{\text{head}} \equiv \mathbf{E}^T$$
- In the implementation, `LMHead` does NOT allocate an independent parameter tensor.
- `LMHead` accepts a reference to `TokenEmbedding.weight` or executes `F.linear(x, embedding.weight)`.
- A dedicated assertion must verify:
  `assert model.lm_head.weight.data_ptr() == model.embedding.weight.data_ptr()`

---

## 6. Initialization Strategy

To ensure zero gradient explosion during early optimization:
1. **Embedding**: $\mathcal{N}(0, 0.02)$ or $\mathcal{N}(0, 1/\sqrt{d_{\text{model}}} = 0.072)$.
2. **Standard Projections ($W_Q, W_K, W_V, W_{\text{gate}}, W_{\text{up}}$)**: Truncated normal with standard deviation $\sigma = 0.02$.
3. **Residual Projections ($W_O, W_{\text{down}}$)**: Scaled by depth:
   $$\sigma_{\text{residual}} = \frac{0.02}{\sqrt{2N}} = \frac{0.02}{\sqrt{12}} \approx 0.00577$$
4. **RMSNorm Scales ($\boldsymbol{\gamma}$)**: Initialized identically to $\mathbf{1.0}$.

---

## 7. Data Type & Numerical Precision Policy

- **Default Tensor Dtype**: `torch.float32` for all model weights and activations.
- **Trigonometric Evaluation**: RoPE angular frequencies and rotation matrices evaluated strictly in `float32`.
- **Masking Value**: Additive $-\infty$ represented as $-1e9$ (float32 safe) or `torch.finfo(torch.float32).min` to prevent underflow artifacts in softmax.

---

## 8. Error-Handling & Bounds Checking Policy

The neural core must fail fast on invalid inputs:
1. **Token ID Bounds**: Token IDs outside $[0, V-1] = [0, 4095]$ must raise a `ValueError` or `IndexError`.
2. **Context Length Bounds**: Sequences with $T > 512$ must be rejected with a descriptive `ValueError`.
3. **Dimension Consistency**: Hidden dimension and head count must strictly adhere to $d_{\text{model}} \pmod H = 0$.

---

## 9. Comprehensive Testing Strategy

1. **Primitive Unit Tests**:
   - `test_token_embedding_shape_and_bounds`
   - `test_rmsnorm_mathematical_precision_and_invariance`
   - `test_rope_rotation_properties_and_norm_preservation`
   - `test_causal_mask_triangular_property`
   - `test_multi_head_attention_shape_and_residuals`
   - `test_swiglu_activation_and_shape`
   - `test_transformer_block_forward`
2. **Complete Model Verification**:
   - `test_chakr_micro_forward_shape` (`[2, 16] -> [2, 16, 4096]`)
   - `test_weight_tying_memory_sharing`
   - `test_parameter_count_matches_accounting` ($3,443,136$)
   - `test_causality_temporal_leakage` (modifying $t+1$ produces identical outputs at $t$)
   - `test_determinism_eval_mode`
   - `test_gradient_flow_and_smoke_training` (loss decreases on synthetic sequence)
   - `test_context_boundary_exceeded_rejected`
3. **Regression Tests**:
   - All 151 existing tokenizer and configuration tests must remain 100% green.

---
*End of Neural Core Implementation Plan*
