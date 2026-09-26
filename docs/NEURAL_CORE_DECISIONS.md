# Neural Core Decision Records (ADR) — ChakrView Step 4

**Project**: ChakrView  
**Phase**: Step 4 — Neural Core Architecture Specification & Tensor Contract  
**Status**: ACTIVE, RATIFIED & FROZEN  

---

## Overview

This document formally records the architectural decisions ratified during **Step 4** for the **ChakrView Neural Core (Chakr-Micro v0.1)**. These records build upon the preceding architecture and tokenizer decisions and freeze the exact mathematical, structural, and dimensional contracts required before neural implementation.

---

## ADR 17: Ratification of Chakr-Micro v0.1 Sizing ($N=6, d_{\text{model}}=192$)

### Context
A concrete, unambiguous model sizing is required for the first trainable indigenous neural core of ChakrView. Candidate depths $N \in \{4, 6, 8, 10, 12\}$ and head counts $H \in \{3, 4, 6, 8\}$ were evaluated on parameter count, FLOPs, memory footprint, cache alignment, and implementation complexity on low-resource CPU targets.

### Decision
Ratify and freeze **Chakr-Micro v0.1**:
- $N = 6$ stacked transformer layers
- $d_{\text{model}} = 192$
- $H = 6$ query heads, $H_{kv} = 6$ key-value heads
- $d_{\text{head}} = 32$
- $d_{\text{ff}} = 512$
- $T_{\text{max}} = 512$
- $V = 4096$
- Weight tying enabled ($W_{\text{out}} = E^T$)
- Total parameters: **$3,443,136$**

### Reasoning & Trade-offs
- **Exact Integer Cache Alignment**: In Chakr-Micro, $d_{\text{ff}} = \frac{8}{3} \times 192 = 512$ is an exact integer and a pure power of 2 ($2^9$), aligning with CPU cache line boundaries ($512 \times 4\text{ bytes} = 2,048\text{ bytes} = 32$ cache lines).
- **SIMD Register Mapping**: Head dimension $d_{\text{head}} = 32$ maps cleanly to four 256-bit AVX2 registers, two 512-bit AVX-512 registers, or eight 128-bit ARM NEON registers with zero unaligned loads.
- **Hierarchical Representational Depth**: $N = 6$ provides sufficient depth for compositional feature extraction while bounding backward-pass memory ($114.75\text{ MiB}$ at FP32 for $T=512$) on low-spec CPUs.

---

## ADR 18: Retention of Simple Multi-Head Attention (MHA) for v0.1

### Context
Grouped-Query Attention (GQA) reduces KV cache memory in autoregressive generation by sharing key/value heads across query head groups. We evaluated whether GQA should be introduced in v0.1 or deferred.

### Decision
Retain **Simple Multi-Head Attention (MHA)** with $H_{kv} = H = 6$ for v0.1. Document GQA strictly as an architectural extension point for larger future models ($d_{\text{model}} \ge 512, T \ge 2048$).

### Reasoning & Trade-offs
- **Trivial KV Cache Footprint**: For Chakr-Micro ($T_{\text{max}} = 512, B = 1$), the FP16 KV cache across all 6 layers is only $2.25\text{ MiB}$ ($2.36\text{ MB}$).
- **Negligible Parameter Savings**: GQA with $H_{kv} = 2$ would only save $147,456$ parameters ($4.2\%$ of model weights).
- **CPU Kernel Simplicity**: On low-spec CPUs and edge devices, standard MHA uses simple contiguous matrix multiplications without head-broadcasting or strided index remapping.

---

## ADR 19: Pre-RMSNorm Formulation and Numerical Epsilon Policy

### Context
Normalization must provide numerical stability across deep residual networks while minimizing CPU compute overhead.

### Decision
Adopt **Pre-RMSNorm** across all transformer blocks with $\epsilon = 10^{-5}$ and learnable scale $\boldsymbol{\gamma} \in \mathbb{R}^{d_{\text{model}}}$, omitting additive bias $\boldsymbol{\beta}$ ($b=0$):
$$\text{RMSNorm}(\mathbf{u}) = \frac{\mathbf{u}}{\sqrt{\frac{1}{d_{\text{model}}} \sum_{i=1}^{d_{\text{model}}} u_i^2 + \epsilon}} \odot \boldsymbol{\gamma}$$

### Reasoning & Trade-offs
- **Compute Efficiency**: Dispensing with mean-centering saves $7\text{--}12\%$ of normalizer runtime on CPU architectures compared to standard LayerNorm.
- **Epsilon Selection**: $\epsilon = 10^{-5}$ provides robust headroom against floating-point underflow/overflow across FP32, FP16, and BF16.
- **Residual Highway**: Pre-normalization ensures an unattenuated identity path along the residual stream, preventing vanishing or exploding gradients.

---

## ADR 20: Rotary Position Embedding (RoPE) Integration Standard

### Context
Autoregressive attention requires position information. The mechanism must support relative distance awareness without adding trainable parameters or introducing non-standard kernel dependencies.

### Decision
Ratify **Rotary Position Embedding (RoPE)** with base frequency $\Theta = 10000.0$ and full rotary dimension $d_{\text{rot}} = d_{\text{head}} = 32$, applied pairwise to Query ($Q$) and Key ($K$) head representations inside the attention sub-layer:
$$\begin{pmatrix} \tilde{v}_{2k} \\ \tilde{v}_{2k+1} \end{pmatrix} = \begin{pmatrix} \cos(m \theta_k) & -\sin(m \theta_k) \\ \sin(m \theta_k) & \cos(m \theta_k) \end{pmatrix} \begin{pmatrix} v_{2k} \\ v_{2k+1} \end{pmatrix}, \quad \theta_k = 10000^{-k / 16}, \quad k \in \{0, \dots, 15\}$$

### Reasoning & Trade-offs
- **Zero Trainable Parameters**: RoPE adds zero weights to the parameter budget, keeping the model compact.
- **Strict Relative Displacement**: The inner product $\langle \text{RoPE}(\mathbf{q}_m), \text{RoPE}(\mathbf{k}_n) \rangle$ is a function strictly of $(m - n)$, matching natural language token displacement.
- **Values and Residual Stream Untouched**: RoPE is excluded from $V$ and the residual stream, preserving semantic content invariance.

---

## ADR 21: SwiGLU Activation & Feed-Forward Dimensional Sizing

### Context
The feed-forward sub-layer provides non-linear expressive capacity. Traditional transformers utilize 2-matrix MLPs with ReLU or GELU.

### Decision
Adopt **SwiGLU** with dimension $d_{\text{ff}} = 512$:
$$\text{SwiGLU}(\hat{h}) = \left( \text{Swish}(\hat{h} W_{\text{gate}}) \odot (\hat{h} W_{\text{up}}) \right) W_{\text{down}}$$
where $W_{\text{gate}}, W_{\text{up}} \in \mathbb{R}^{192 \times 512}$ and $W_{\text{down}} \in \mathbb{R}^{512 \times 192}$ are bias-free linear projections.

### Reasoning & Trade-offs
- **High Expressive Density**: Gated bilinear activations consistently outperform standard ReLU/GELU MLPs in perplexity per parameter and FLOP.
- **Exact Compute Parity**: Setting $d_{\text{ff}} = \frac{8}{3} d_{\text{model}} = 512$ provides exact FLOP and parameter parity with a conventional $4d_{\text{model}}$ 2-matrix FFN ($3 \times 192 \times 512 = 294,912$ params vs $2 \times 192 \times 768 = 294,912$ params).
- **Cache-Line Alignment**: $512$ elements in FP32 correspond to exactly $2,048$ bytes ($32$ standard 64-byte CPU cache lines).

---

## ADR 22: Bias-Free Linear Projections and Weight Tying

### Context
Dense projections may include additive biases, and output heads may use independent matrices.

### Decision
1. **Omit additive biases** ($b = 0$) across all projections ($W_Q, W_K, W_V, W_O, W_{\text{gate}}, W_{\text{up}}, W_{\text{down}}$).
2. **Enable Weight Tying**: $W_{\text{out}} = E^T$.

### Reasoning & Trade-offs
- **Parameter Savings**: Weight tying saves $786,432$ parameters ($22.84\%$ of the total parameter budget).
- **Cache Locality**: Reusing the embedding matrix in memory reduces working set size during inference.
- **Simplified GEMMs**: Bias-free linear layers eliminate trailing vector additions and simplify quantization kernels.

---

## ADR 23: Hardware-Centric Memory and FLOP Accounting Policy

### Context
Claims regarding model size, cache fitting, and memory consumption must adhere to strict scientific boundaries.

### Decision
Enforce rigorous boundaries between static weight storage, dynamic runtime buffers, and hardware claims:
1. **Static Parameter Footprint**:
   - Total parameters: $3,443,136$
   - FP32: $13.13\text{ MiB}$ ($13.77\text{ MB}$)
   - FP16: $6.57\text{ MiB}$ ($6.89\text{ MB}$)
   - INT8: $3.28\text{ MiB}$ ($3.44\text{ MB}$)
   - INT4: $1.64\text{ MiB}$ ($1.72\text{ MB}$)
2. **Explicit Exclusions from Static Figures**:
   - Dynamic activations ($15.19\text{ MiB}$ to $114.75\text{ MiB}$ during training)
   - KV cache ($1.125\text{ MiB}$ to $4.50\text{ MiB}$ depending on context length and precision)
   - Runtime allocator and engine overhead
3. **CPU Cache Claim Rule**:
   - The assertion that Chakr-Micro fits in CPU L3 cache is strictly classified as an **empirical engineering hypothesis** to be benchmarked on target hardware, not an architectural guarantee.

---
*End of Neural Core Decision Records*
