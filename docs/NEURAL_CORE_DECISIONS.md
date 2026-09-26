# Neural Core Decision Records (ADR) — ChakrView Step 3.1

**Project**: ChakrView  
**Phase**: Step 3.1 — Neural Core Architecture & Mathematical Specification  
**Status**: ACTIVE & RATIFIED  

---

## Overview

This document records the architectural decisions ratified during **Step 3.1** for the **ChakrView Neural Core (Chakr-Micro v0.1)**. These records build upon ADR 01–ADR 10 from previous steps and freeze the exact mathematical, structural, and dimensional contracts required before neural implementation.

---

## ADR 11: Freezing Chakr-Micro Core Hyperparameters over Structural Alternatives

### Context
A concrete, non-ambiguous sizing was required for the first trainable indigenous neural core of ChakrView. Three candidate configurations were evaluated based on parameter count, FLOPs, memory footprint, cache alignment, and implementation complexity:
- Candidate A (Deeper / Narrower): $L=8, d_{\text{model}}=160, H=5, d_{\text{head}}=32, d_{\text{ff}}=416$ ($3.07\text{M}$ params)
- Candidate B (Chakr-Micro — Balanced): $L=6, d_{\text{model}}=192, H=6, d_{\text{head}}=32, d_{\text{ff}}=512$ ($3.44\text{M}$ params)
- Candidate C (Shallower / Wider): $L=4, d_{\text{model}}=256, H=8, d_{\text{head}}=32, d_{\text{ff}}=672$ ($4.16\text{M}$ params)

### Decision
Ratify and freeze **Candidate B (Chakr-Micro)** as the definitive architectural configuration for v0.1:
- $L = 6$ layers
- $d_{\text{model}} = 192$
- $H = 6$ query heads, $H_{kv} = 6$ key-value heads (Simple MHA)
- $d_{\text{head}} = 32$
- $d_{\text{ff}} = 512$
- $T_{\text{max}} = 512$
- Provisional $V = 4,096$
- Weight tying enabled ($W_{\text{out}} = E^T$)
- Total parameters: **$3,443,136$**

### Reasoning & Trade-offs
- **Symmetric Subspace Partitioning**: Candidate A has $H = 5$, an odd number of attention heads that cannot be divided evenly into SIMD pairs, complicates future GQA experiments, and prevents balanced multi-core threading. Candidate B ($H = 6$) permits clean grouping by 1, 2, 3, or 6.
- **Exact Integer Cache Alignment**: In Candidate B, $d_{\text{ff}} = \frac{8}{3} \times 192 = 512$ is an exact integer and a pure power of 2 ($2^9$), aligning with CPU cache line boundaries ($512 \times 4\text{ bytes} = 2,048\text{ bytes} = 32$ cache lines). Candidates A and C required fractional rounding.
- **Hierarchical Representational Depth**: Candidate C ($L = 4$) restricts compositional feature depth, while Candidate A ($L = 8$) increases forward-pass memory latency on memory-bandwidth-bound CPUs. Candidate B provides an optimal balance for low-spec CPU research.

---

## ADR 12: Pre-RMSNorm Formulation & Numerical Epsilon Standard

### Context
Normalization must provide numerical stability across deep residual networks while minimizing CPU compute overhead.

### Decision
Adopt **Pre-RMSNorm** across all transformer blocks with $\epsilon = 10^{-5}$ and learnable scale $\boldsymbol{\gamma} \in \mathbb{R}^{d_{\text{model}}}$, omitting additive bias $\boldsymbol{\beta}$ ($b=0$):
$$\text{RMSNorm}(\mathbf{u}) = \frac{\mathbf{u}}{\sqrt{\frac{1}{d_{\text{model}}} \sum_{i=1}^{d_{\text{model}}} u_i^2 + \epsilon}} \odot \boldsymbol{\gamma}$$

### Reasoning & Trade-offs
- **Compute Efficiency**: By eliminating mean computation ($\mu = \frac{1}{d} \sum x_i$), RMSNorm saves $7\text{--}12\%$ of normalization runtime on CPU architectures compared to standard LayerNorm.
- **Epsilon Selection**: $\epsilon = 10^{-5}$ provides robust headroom against floating-point underflow/overflow across both FP32 and FP16/BF16 without distorting feature variance.
- **Identity Gradient Highway**: Applying normalization strictly before sub-layer operations (pre-norm) ensures an unattenuated identity path along the residual stream, preventing vanishing or exploding gradients.

---

## ADR 13: Rotary Position Embedding (RoPE) Integration Standard

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

## ADR 14: SwiGLU Activation & Feed-Forward Dimensional Sizing

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

## ADR 15: Memory & FLOP Accounting Boundary Policy

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
   - Dynamic activations ($45\text{--}80\text{ MB}$ during training)
   - Optimizer state ($27.55\text{ MB}$ for AdamW moments)
   - KV cache ($1.125\text{ MB}$ to $4.50\text{ MB}$ depending on context length and precision)
   - Runtime allocator and engine overhead
3. **CPU Cache Claim Rule**:
   - The assertion that Chakr-Micro fits in CPU L3 cache is strictly classified as an **empirical engineering hypothesis** to be benchmarked on target hardware, not an architectural guarantee.

---
*End of Neural Core Decision Records*
