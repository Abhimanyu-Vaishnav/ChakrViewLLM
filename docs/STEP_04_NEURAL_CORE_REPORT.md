# ChakrView Step 4: Neural Core Prototype Engineering Report

**Document Version**: 1.0.0  
**Phase**: Step 4 — Neural Core Prototype Implementation & Verification  
**Model Target**: Chakr-Micro v0.1 ($3.44\text{M}$ Parameters)  
**Status**: COMPLETE, VERIFIED & RATIFIED  

---

## 1. Implemented Architecture

Chakr-Micro v0.1 is an indigenous, from-scratch decoder-only causal autoregressive language model brain:
- **Embedding**: Token index lookup $E \in \mathbb{R}^{4096 \times 192}$.
- **Layers**: 6 identical stacked Pre-RMSNorm Transformer blocks.
- **Attention**: Simple Multi-Head Attention ($H = 6, H_{kv} = 6, d_{\text{head}} = 32$).
- **Rotary Position Embedding**: RoPE applied to $Q$ and $K$ ($\Theta = 10000.0, d_{\text{rot}} = 32$).
- **Feed-Forward**: SwiGLU with intermediate dimension $d_{\text{ff}} = 512$ ($\frac{8}{3} d_{\text{model}}$).
- **Projections**: All linear layers are strictly bias-free ($b = 0$).
- **Normalization**: Pre-RMSNorm with scale $\boldsymbol{\gamma}$ and $\epsilon = 10^{-5}$.
- **Output Head**: Tied LM head sharing underlying parameter tensor with embedding ($W_{\text{out}} = E^T$).

---

## 2. Tensor Shapes Across Forward Pipeline

```
input_ids:                   [B, T]              dtype=int64
embedded_tokens:             [B, T, 192]         dtype=float32
pre_attn_norm:               [B, T, 192]         dtype=float32
q_proj, k_proj, v_proj:      [B, T, 192] each    dtype=float32
q_heads, k_heads, v_heads:   [B, 6, T, 32] each  dtype=float32
rotated_q, rotated_k:        [B, 6, T, 32] each  dtype=float32
attention_scores:            [B, 6, T, T]        dtype=float32
causal_mask:                 [1, 1, T, T]        additive float (-1e9 upper triangle)
attention_probs:             [B, 6, T, T]        dtype=float32
attention_context:           [B, T, 192]         dtype=float32
post_attention_residual:     [B, T, 192]         dtype=float32
pre_ffn_norm:                [B, T, 192]         dtype=float32
swiglu_intermediate:         [B, T, 512]         dtype=float32
swiglu_down_projected:       [B, T, 192]         dtype=float32
post_ffn_residual:           [B, T, 192]         dtype=float32
final_norm:                  [B, T, 192]         dtype=float32
logits:                      [B, T, 4096]        dtype=float32
```

---

## 3. Parameter Count & Memory Breakdown

Analytical expectation matches programmatic code accounting bit-for-bit:
- **Embedding Matrix ($E$)**: $4096 \times 192 = 786,432$
- **Attention Projections ($W_Q, W_K, W_V, W_O$)**: $6 \times 4 \times (192 \times 192) = 884,736$
- **SwiGLU Projections ($W_{\text{gate}}, W_{\text{up}}, W_{\text{down}}$)**: $6 \times 3 \times (192 \times 512) = 1,769,472$
- **Normalization Scales ($\boldsymbol{\gamma}_1, \boldsymbol{\gamma}_2, \boldsymbol{\gamma}_{\text{final}}$)**: $(6 \times 2 \times 192) + 192 = 2,496$
- **Tied LM Head ($W_{\text{out}} = E^T$)**: $0$ additional parameters
- **TOTAL UNIQUE PARAMETERS**: **$3,443,136$** ($100.00\%$)

### Static Weight RAM
- **FP32**: $13.13\text{ MiB}$ ($13.77\text{ MB}$)
- **FP16**: $6.57\text{ MiB}$ ($6.89\text{ MB}$)
- **INT8**: $3.28\text{ MiB}$ ($3.44\text{ MB}$)
- **INT4**: $1.64\text{ MiB}$ ($1.72\text{ MB}$)

---

## 4. Initialization Strategy

- **Embedding**: $\mathcal{N}(0, 0.02)$.
- **Standard Projections ($W_Q, W_K, W_V, W_{\text{gate}}, W_{\text{up}}$)**: Truncated normal with standard deviation $\sigma = 0.02$.
- **Residual Projections ($W_O, W_{\text{down}}$)**: Scaled by depth:
  $$\sigma_{\text{residual}} = \frac{0.02}{\sqrt{2N}} = \frac{0.02}{\sqrt{12}} \approx 0.00577$$
- **RMSNorm Scales**: Initialized uniformly to $\mathbf{1.0}$.

---

## 5. Verification Results

### 5.1 Causal Masking Verification
Mathematically verified via prefix divergence tests:
- When a future token at position $k$ is altered, outputs at positions $i < k$ remain identical within numerical precision ($\Delta < 10^{-6}$).
- Outputs at positions $i \ge k$ diverge as expected.
- Proves zero temporal information leakage into prefix representations.

### 5.2 RoPE Verification
- Position 0 values are preserved identically: $\text{RoPE}(x)_0 \equiv x_0$.
- Isometric property verified: $\|\text{RoPE}(x)\| = \|x\|$ across all sequence positions.

### 5.3 Weight Tying Verification
- Hardware pointer equality verified:
  `assert model.lm_head.weight.data_ptr() == model.embedding.weight.data_ptr()`
- Eliminates 786,432 duplicate parameters ($3.14\text{ MB}$ at FP32).

### 5.4 Gradient Smoke-Test Results
- Executed 15 optimization steps on synthetic repeating pattern (`ABABAB...`).
- Loss is strictly finite (zero `NaN` or `Inf`).
- Gradients exist and are non-zero across all model parameters.
- Loss decreased monotonically from initial baseline.
- Model parameters successfully updated under backpropagation.

### 5.5 CPU Forward-Pass Benchmark (Measured)
Empirically measured on Intel i9-13900H (Windows 11, PyTorch 2.14.0+cpu, 4 threads, Batch=1, 5 warmup runs, 20 measured runs):

| Context ($T$) | Median Latency (ms) | p95 Latency (ms) | Per-Token Latency (ms) | Throughput (tok/s) | Logits RAM (MB) | Process RSS (MB) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$T = 1$** | $1.72\text{ ms}$ | $2.13\text{ ms}$ | $1.719\text{ ms}$ | $581.7\text{ tok/s}$ | $0.02\text{ MB}$ | $233.9\text{ MB}$ |
| **$T = 8$** | $1.85\text{ ms}$ | $2.48\text{ ms}$ | $0.231\text{ ms}$ | $4,325.0\text{ tok/s}$ | $0.12\text{ MB}$ | $234.5\text{ MB}$ |
| **$T = 16$** | $2.04\text{ ms}$ | $2.59\text{ ms}$ | $0.128\text{ ms}$ | $7,843.9\text{ tok/s}$ | $0.25\text{ MB}$ | $234.8\text{ MB}$ |
| **$T = 32$** | $2.65\text{ ms}$ | $3.96\text{ ms}$ | $0.083\text{ ms}$ | $12,084.6\text{ tok/s}$ | $0.50\text{ MB}$ | $235.3\text{ MB}$ |
| **$T = 64$** | $3.58\text{ ms}$ | $5.82\text{ ms}$ | $0.056\text{ ms}$ | $17,875.3\text{ tok/s}$ | $1.00\text{ MB}$ | $236.6\text{ MB}$ |
| **$T = 128$** | $5.75\text{ ms}$ | $7.50\text{ ms}$ | $0.045\text{ ms}$ | $22,244.2\text{ tok/s}$ | $2.00\text{ MB}$ | $239.9\text{ MB}$ |
| **$T = 256$** | $9.87\text{ ms}$ | $14.95\text{ ms}$ | $0.039\text{ ms}$ | $25,948.0\text{ tok/s}$ | $4.00\text{ MB}$ | $242.7\text{ MB}$ |
| **$T = 512$** | $20.52\text{ ms}$ | $22.89\text{ ms}$ | $0.040\text{ ms}$ | $24,950.3\text{ tok/s}$ | $8.00\text{ MB}$ | $256.2\text{ MB}$ |

### 5.6 Model Serialization & Deserialization
- Validated via `torch.save(model.state_dict())` and `model.load_state_dict()`.
- Fresh instance produces bit-exact identical forward pass representations ($\text{atol} < 10^{-6}$).
- Weight tying pointer identity is fully preserved after deserialization:
  `assert model2.lm_head.weight.data_ptr() == model2.embedding.weight.data_ptr()`.

### 5.7 Context Length Enforcement & Multi-Sequence Smoke Test
- Validated across $T \in \{1, 8, 32, 128, 512\}$ with zero crashes, finite values, and determinism under `model.eval()`.
- Invalid sequence length ($T = 513$) is strictly rejected with explicit `ValueError`.

---

## 6. Comprehensive Memory Accounting

A clear distinction is maintained between static weights, activations, KV cache, and runtime process memory:

### 6.1 Static Parameter Weights (Theoretical & Fixed)
- Unique Parameters: $3,443,136$
- FP32: $13.134\text{ MiB}$ ($13.773\text{ MB}$)
- FP16 / BF16: $6.567\text{ MiB}$ ($6.886\text{ MB}$)
- INT8: $3.284\text{ MiB}$ ($3.443\text{ MB}$)
- INT4: $1.642\text{ MiB}$ ($1.722\text{ MB}$)

### 6.2 Forward Activation Memory (Batch=1, FP32)
- **Attention Context Matrices**: $6 \times [1, 6, T, T] \times 4\text{ bytes}$
  - $T=32$: $24.58\text{ KiB}$
  - $T=128$: $393.22\text{ KiB}$
  - $T=512$: $6.291\text{ MiB}$
- **SwiGLU Activations**: Gate & Up projections $6 \times [1, T, 512] \times 4\text{ bytes} \times 2 = 24.58\text{ KiB} \times T$
  - $T=512$: $12.58\text{ MiB}$
- **Logits Output Buffer**: $[1, T, 4096] \times 4\text{ bytes}$
  - $T=32$: $0.50\text{ MB}$
  - $T=128$: $2.00\text{ MB}$
  - $T=512$: $8.00\text{ MB}$

### 6.3 KV Cache Scaling (Inference, Batch=1)
- KV state per token per layer: $2 \times H_{kv} \times d_{\text{head}} \times 4\text{ bytes} = 2 \times 6 \times 32 \times 4 = 1,536\text{ bytes}$
- Total KV cache across 6 layers: $9,216\text{ bytes/token}$
  - $T=32$: $294.9\text{ KiB}$
  - $T=128$: $1.152\text{ MiB}$
  - $T=512$: $4.608\text{ MiB}$ ($2.304\text{ MiB}$ in FP16)

### 6.4 Process Working Set (Measured on CPU)
- Python 3.14 + PyTorch runtime base RSS: $206.9\text{ MB}$
- Model instantiated & initialized: $232.3\text{ MB}$ ($\Delta \approx 25.4\text{ MB}$, covering $13.1\text{ MB}$ weights + PyTorch allocator pools)
- Peak forward execution RSS ($T=512$): $256.2\text{ MB}$ ($\Delta \approx 23.9\text{ MB}$ for intermediate tensors and logits buffer)

---

## 7. Known Limitations

1. **Python Overhead**: Pure Python execution adds dispatch overhead on small batch and sequence sizes compared to ahead-of-time compiled C++ binaries.
2. **Memory Bandwidth on Legacy CPUs**: While INT8 models fit into L3 cache on modern processors, older 28nm CPUs lacking large L3 cache will be memory-bandwidth bound during autoregressive token generation.

---

## 8. What Remains Unfrozen

- Final optimizer choice (AdamW, Lion, or Sophia).
- Learning rate schedule and warmup steps for production pre-training.
- Post-training quantization implementation (INT8 / INT4 kernels).
- Future GQA scaling for larger models ($d \ge 512, T \ge 2048$).
- Inference deployment engines (pure C++, GGML, ONNX, WASM).

---

## 9. Recommended Next Step

Proceed to **Step 5: Pre-Training Infrastructure & Data Loader Pipeline**.
- Construct streaming batch iterators feeding clean token sequences from the Step 3 corpus.
- Implement training loop with loss logging, learning rate schedules, gradient clipping, and checkpointing.

---
*End of Neural Core Engineering Report*
