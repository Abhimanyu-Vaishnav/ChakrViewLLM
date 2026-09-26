# ChakrView Step 4: Neural Core Architecture Specification & Tensor Contract

**Document Version**: 1.0.0  
**Phase**: Step 4 — Neural Core Architecture Specification & Tensor Contract  
**Model Target**: Chakr-Micro v0.1  
**Status**: ACTIVE, RATIFIED & FROZEN  
**Scope**: Strictly Architecture, Mathematics, Dimensions & Tensor Contracts (Zero Implementation Code, Zero Model Weights, Zero Training Loops)

---

## 1. Executive Summary & Continuity Audit

Step 4 establishes the formal, frozen architectural and mathematical specification for the **Chakr-Micro v0.1** neural core. This specification bridges the experimentally ratified Byte-Level BPE tokenizer from Step 3 ($V = 4096$, Raw Byte BPE, Normal BPE numerics) to the downstream neural computation stages.

### 1.1 Continuity from Preceding Steps
- **Step 1 (`docs/STEP_01_NEURAL_CORE_SPEC.md`)**: Established foundational decoder-only transformer principles, Pre-RMSNorm, RoPE, SwiGLU, and bias-free projections.
- **Step 2 & 2.4 (`docs/STEP_02_4_TOKENIZER_INTERFACE.md`)**: Defined the tensor handoff interface: `input_ids [B, T]` and `attention_mask [B, T]` producing `logits [B, T, 4096]` with weight tying $W_{\text{out}} = E^T$.
- **Step 3 (`docs/STEP_03_DECISIONS.md`)**: Empirically ratified vocabulary size $V = 4096$, Raw Byte BPE (Variant A), and frequency-based numeric tokenization (Candidate C), with $139/139$ passing tests.

### 1.2 Core Design Tenet
ChakrView is an indigenous, from-scratch tiny language model designed to run efficiently on CPU and edge environments, including older low-end processors (e.g. 28nm era chips). We do NOT build wrappers or use external model weights. Every mathematical operation, memory buffer, parameter formula, and dimensional alignment is planned for hardware efficiency, deterministic execution, and future low-precision quantization.

---

## 2. Step 4A — Parameter Budget Analysis

The model dimension is frozen at $d_{\text{model}} = 192$ and vocabulary at $V = 4096$. We evaluate layer depth $N \in \{4, 6, 8, 10, 12\}$ and attention head counts $H \in \{3, 4, 6, 8\}$.

### 2.1 Mathematical Divisibility and Head Constraints
The head dimension is defined as:
$$d_{\text{head}} = \frac{d_{\text{model}}}{H}$$
For pairwise 2D rotation in RoPE, $d_{\text{head}}$ must be an even integer ($d_{\text{head}} \pmod 2 = 0$).

| Head Count ($H$) | $d_{\text{head}}$ | Divisible? | RoPE Parity? | SIMD & Cache Line Suitability | Assessment |
| :---: | :---: | :---: | :---: | :--- | :--- |
| **$H = 3$** | $64$ | Yes ($192/3$) | Even (32 pairs) | 4 AVX2 / 2 AVX-512 regs; 4 cache lines | **Rejected**: Odd head count prevents symmetric multi-core parallelization; incompatible with GQA partitioning (cannot divide 3 heads into 2 KV groups). |
| **$H = 4$** | $48$ | Yes ($192/4$) | Even (24 pairs) | 3 cache lines (192 bytes) | **Rejected**: Fractional cache line alignment; non-power-of-2 head dimension complicates SIMD vector loads. |
| **$H = 6$** | **$32$** | **Yes ($192/6$)** | **Even (16 pairs)** | **2 cache lines (128 bytes); exactly 4 AVX2 / 8 NEON regs** | **Selected**: Optimal cache alignment; power of 2 ($2^5$); divides evenly into 1, 2, 3, or 6 threads; seamless future GQA scaling (e.g. 2 or 3 KV heads). |
| **$H = 8$** | $24$ | Yes ($192/8$) | Even (12 pairs) | 96 bytes (1.5 cache lines) | **Rejected**: Cache line tearing; non-power-of-2 requires register unaligned loads. |
| **$H = 5$** | $38.4$ | No | Invalid | Fractional | **Rejected**: Mathematically invalid. |

### 2.2 Parameter Formulation per Subcomponent
For a given layer depth $N$, intermediate FFN width $d_{\text{ff}} = 512$, and vocabulary $V = 4096$:

1. **Input Embedding Matrix ($E$)**:
   $$\text{Params}_{\text{emb}} = V \times d_{\text{model}} = 4096 \times 192 = 786,432$$
2. **Attention Sub-layer ($W_Q, W_K, W_V, W_O$)**:
   All linear projections are bias-free ($b = 0$):
   $$\text{Params}_{\text{attn}} = 4 \times (d_{\text{model}} \times d_{\text{model}}) = 4 \times (192 \times 192) = 4 \times 36,864 = 147,456$$
3. **Attention Pre-RMSNorm ($\boldsymbol{\gamma}_1$)**:
   Bias-free learnable scale vector:
   $$\text{Params}_{\text{norm1}} = d_{\text{model}} = 192$$
4. **SwiGLU FFN Sub-layer ($W_{\text{gate}}, W_{\text{up}}, W_{\text{down}}$)**:
   Three bias-free matrices where $d_{\text{ff}} = 512$:
   $$\text{Params}_{\text{ffn}} = 3 \times (d_{\text{model}} \times d_{\text{ff}}) = 3 \times (192 \times 512) = 3 \times 98,304 = 294,912$$
5. **FFN Pre-RMSNorm ($\boldsymbol{\gamma}_2$)**:
   $$\text{Params}_{\text{norm2}} = d_{\text{model}} = 192$$
6. **Per-Layer Total ($\text{Params}_{\text{layer}}$)**:
   $$\text{Params}_{\text{layer}} = 147,456 + 192 + 294,912 + 192 = 442,752$$
7. **Final RMSNorm ($\boldsymbol{\gamma}_{\text{final}}$)**:
   $$\text{Params}_{\text{final}} = d_{\text{model}} = 192$$
8. **Tied LM Head ($W_{\text{out}} = E^T$)**:
   $$\text{Params}_{\text{head}} = 0 \quad (\text{tied to } E)$$
9. **Total Model Parameters**:
   $$\text{Params}_{\text{total}} = \text{Params}_{\text{emb}} + N \times \text{Params}_{\text{layer}} + \text{Params}_{\text{final}} = 786,432 + N \times 442,752 + 192$$

### 2.3 Layer Candidates Evaluation ($N \in \{4, 6, 8, 10, 12\}$)

| Candidate | Total Parameters | Params/Layer | Embedding % | Layers % | FP32 Weight RAM | FP16/BF16 Weight RAM | INT8 Weight RAM | INT4 Weight RAM |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$N = 4$** | $2,557,632$ | $442,752$ | $30.75\%$ | $69.24\%$ | $9.76\text{ MiB}$ | $4.88\text{ MiB}$ | $2.44\text{ MiB}$ | $1.22\text{ MiB}$ |
| **$N = 6$ (Selected)** | **$3,443,136$** | **$442,752$** | **$22.84\%$** | **$77.15\%$** | **$13.13\text{ MiB}$** | **$6.57\text{ MiB}$** | **$3.28\text{ MiB}$** | **$1.64\text{ MiB}$** |
| **$N = 8$** | $4,328,640$ | $442,752$ | $18.17\%$ | $81.82\%$ | $16.51\text{ MiB}$ | $8.26\text{ MiB}$ | $4.13\text{ MiB}$ | $2.06\text{ MiB}$ |
| **$N = 10$** | $5,214,144$ | $442,752$ | $15.08\%$ | $84.91\%$ | $19.89\text{ MiB}$ | $9.95\text{ MiB}$ | $4.97\text{ MiB}$ | $2.49\text{ MiB}$ |
| **$N = 12$** | $6,099,648$ | $442,752$ | $12.89\%$ | $87.10\%$ | $23.27\text{ MiB}$ | $11.63\text{ MiB}$ | $5.82\text{ MiB}$ | $2.91\text{ MiB}$ |

> **IMPORTANT SEPARATION**: Static weight memory reflects only parameter storage ($3,443,136$ elements). It strictly excludes dynamic training activations, optimizer states, KV caches, memory alignment padding, and runtime framework heaps.

---

## 3. Step 4B — Attention Design: Simple Multi-Head Attention (MHA)

### 3.1 Subspace Structure
Chakr-Micro v0.1 implements strictly **Simple Multi-Head Attention (MHA)** with $H = 6$ query heads and $H_{kv} = 6$ key-value heads.
- Query representation: $Q \in \mathbb{R}^{B \times H \times T \times d_{\text{head}}}$
- Key representation: $K \in \mathbb{R}^{B \times H \times T \times d_{\text{head}}}$
- Value representation: $V \in \mathbb{R}^{B \times H \times T \times d_{\text{head}}}$
- Output projection: $W_O \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}$ ($192 \times 192$, bias-free)

### 3.2 Evaluation of MHA vs GQA for v0.1
Grouped-Query Attention (GQA) reduces KV cache memory by sharing key/value heads across query head groups (e.g. $H_{kv} = 2$ for $H = 6$).
- **KV Cache Footprint in v0.1**: For $T_{\text{max}} = 512$ at batch size $B = 1$, the complete FP16 KV cache for MHA across all 6 layers is only $2.25\text{ MiB}$ ($2.36\text{ MB}$).
- **Parameter Impact**: GQA with $H_{kv} = 2$ would only save $2 \times (192 \times 64) \times 6 = 147,456$ parameters ($4.2\%$ of model weights).
- **Implementation & Kernel Simplicity**: On low-spec CPUs and edge devices, standard MHA uses simple contiguous matrix multiplications without head-broadcasting or strided index remapping.
- **Architectural Decision**: **MHA remains frozen for v0.1**. GQA is preserved as a documented extension point for larger versions ($d_{\text{model}} \ge 512, T \ge 2048$).

---

## 4. Step 4C — Rotary Position Embeddings (RoPE)

RoPE encodes token position directly into query and key representations via orthogonal 2D Givens rotations, preserving relative token displacement without trainable parameters.

### 4.1 Mathematical Formulation
For head vector $\mathbf{x} \in \mathbb{R}^{d_{\text{head}}}$ at sequence index $m \in [0, T-1]$, elements are grouped into $d_{\text{head}} / 2 = 16$ pairs:
$$\begin{pmatrix} \tilde{x}_{2k} \\ \tilde{x}_{2k+1} \end{pmatrix} = \begin{pmatrix} \cos(m \theta_k) & -\sin(m \theta_k) \\ \sin(m \theta_k) & \cos(m \theta_k) \end{pmatrix} \begin{pmatrix} x_{2k} \\ x_{2k+1} \end{pmatrix}, \quad k \in \{0, 1, \dots, 15\}$$

Where the rotary frequencies $\theta_k$ are geometrically spaced:
$$\theta_k = \Theta^{-2k / d_{\text{head}}} = 10000.0^{-2k / 32} = 10000.0^{-k / 16}$$

### 4.2 Application Bounds & Rules
1. **Target Tensors**: Applied strictly to $Q$ and $K$ heads.
2. **Untouched Tensors**: Value tensor $V$ and residual stream $x$ receive **zero** positional encoding, preserving semantic content invariance.
3. **Head Sharing**: All $H = 6$ heads use identical frequency schedules $\theta_k$.
4. **Numerical Precision**: Angular arguments $m \theta_k$ and trigonometric tables $\cos, \sin$ must be evaluated in float32 to prevent phase-drift at long contexts.

---

## 5. Step 4D — Root Mean Square Normalization (Pre-RMSNorm)

### 5.1 Formulation
RMSNorm regularizes input activations by their root mean square without subtracting the mean:
$$\text{RMS}(\mathbf{u}) = \sqrt{\frac{1}{d_{\text{model}}} \sum_{i=1}^{d_{\text{model}}} u_i^2 + \epsilon}$$
$$\text{RMSNorm}(\mathbf{u}) = \frac{\mathbf{u}}{\text{RMS}(\mathbf{u})} \odot \boldsymbol{\gamma}$$

### 5.2 Specifications & Rationale
- **Numerical Stability Epsilon**: $\epsilon = 10^{-5}$.
- **Learnable Parameters**: Scale vector $\boldsymbol{\gamma} \in \mathbb{R}^{192}$, initialized to $\mathbf{1.0}$.
- **Bias Policy**: Strictly bias-free ($b = 0$).
- **Placement**: Applied strictly before sub-layer operations (**Pre-RMSNorm**):
  - $\text{RMSNorm}_1$ before Attention.
  - $\text{RMSNorm}_2$ before SwiGLU FFN.
  - $\text{RMSNorm}_{\text{final}}$ before Tied LM Head.
- **CPU Advantage**: Dispensing with mean calculation ($\mu = \frac{1}{d} \sum x_i$) reduces memory passes and eliminates 2 vector reductions per normalization, yielding a $7\text{--}12\%$ speedup on CPU vector units.

---

## 6. Step 4E — SwiGLU Feed-Forward Network

### 6.1 Formulation
The feed-forward block utilizes a Gated Linear Unit with Swish activation ($\text{Swish}(z) = z \cdot \sigma(z)$):
$$\text{SwiGLU}(\mathbf{x}) = \left( \text{Swish}(\mathbf{x} W_{\text{gate}}) \odot (\mathbf{x} W_{\text{up}}) \right) W_{\text{down}}$$
where:
- $W_{\text{gate}} \in \mathbb{R}^{192 \times 512}$ (bias-free)
- $W_{\text{up}} \in \mathbb{R}^{192 \times 512}$ (bias-free)
- $W_{\text{down}} \in \mathbb{R}^{512 \times 192}$ (bias-free)

### 6.2 Comparison of FFN Expansion Ratios

| Expansion Ratio | $d_{\text{ff}}$ | Layer FFN Params | Total Model Params (6L) | Cache Alignment (FP32) | Assessment |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **$2.0\times$** | $384$ | $221,184$ | $3,000,768$ | 1,536 bytes (24 cache lines) | Lower capacity; under-utilizes hidden representations. |
| **$2.5\times$** | $480$ | $276,480$ | $3,332,544$ | 1,920 bytes (30 cache lines) | Non-power-of-2; odd cache line boundary. |
| **$\frac{8}{3}\times$ (2.67$\times$)** | **$512$** | **$294,912$** | **$3,443,136$** | **2,048 bytes (32 cache lines)** | **Selected**: Exact integer power of 2 ($2^9$); perfect cache line alignment; exact compute parity with standard $4d$ 2-matrix MLPs ($3 \times 192 \times 512 = 2 \times 192 \times 768$). |
| **$3.0\times$** | $576$ | $331,776$ | $3,664,320$ | 2,304 bytes (36 cache lines) | Increases layer memory without clean power-of-2 alignment. |
| **$3.5\times$** | $672$ | $387,072$ | $3,996,096$ | 2,688 bytes (42 cache lines) | Exceeds micro-parameter budget. |
| **$4.0\times$** | $768$ | $442,368$ | $4,327,872$ | 3,072 bytes (48 cache lines) | Over-allocates parameters to FFN (+50% FFN FLOPs). |

---

## 7. Step 4F — Residual Stream & Block Equations

For layer $l \in \{1, 2, \dots, N\}$, given input tensor $\mathbf{x}_{l-1} \in \mathbb{R}^{B \times T \times 192}$:

1. **Pre-Attention Normalization**:
   $$\mathbf{u}_l = \text{RMSNorm}_1(\mathbf{x}_{l-1}) \in \mathbb{R}^{B \times T \times 192}$$
2. **Attention Projections & RoPE**:
   $$Q_l = \text{RoPE}(\mathbf{u}_l W_Q) \in \mathbb{R}^{B \times 6 \times T \times 32}$$
   $$K_l = \text{RoPE}(\mathbf{u}_l W_K) \in \mathbb{R}^{B \times 6 \times T \times 32}$$
   $$V_l = \mathbf{u}_l W_V \in \mathbb{R}^{B \times 6 \times T \times 32}$$
3. **Causal Attention Core**:
   $$\mathbf{A}_l = \text{Softmax}\left( \frac{Q_l K_l^T}{\sqrt{32}} + M_{\text{causal}} \right) V_l \in \mathbb{R}^{B \times 6 \times T \times 32}$$
   $$\mathbf{a}_l = \text{Reshape}(\mathbf{A}_l) W_O \in \mathbb{R}^{B \times T \times 192}$$
4. **First Residual Connection**:
   $$\mathbf{h}_l = \mathbf{x}_{l-1} + \mathbf{a}_l \in \mathbb{R}^{B \times T \times 192}$$
5. **Pre-FFN Normalization**:
   $$\mathbf{v}_l = \text{RMSNorm}_2(\mathbf{h}_l) \in \mathbb{R}^{B \times T \times 192}$$
6. **SwiGLU Transformation**:
   $$\mathbf{f}_l = \left( \text{Swish}(\mathbf{v}_l W_{\text{gate}}) \odot (\mathbf{v}_l W_{\text{up}}) \right) W_{\text{down}} \in \mathbb{R}^{B \times T \times 192}$$
7. **Second Residual Connection**:
   $$\mathbf{x}_l = \mathbf{h}_l + \mathbf{f}_l \in \mathbb{R}^{B \times T \times 192}$$

---

## 8. Step 4G — Causal Masking Specification

### 8.1 Causality Enforcement
Token at index $i$ must never attend to tokens at indices $j > i$.
The additive causal mask $M_{\text{causal}} \in \mathbb{R}^{1 \times 1 \times T \times T}$ is defined as:
$$M_{\text{causal}}(i, j) = \begin{cases} 0.0 & \text{if } j \le i \\ -\infty \quad (\text{represented as } -10^4 \text{ or } -10^9) & \text{if } j > i \end{cases}$$

### 8.2 Mask Formulation for Reference Implementation
- **v0.1 Reference Implementation**: Use an **explicit upper-triangular additive mask** (`torch.triu` with filled $-\infty$ or numpy boolean mask). It guarantees 100% determinism, mathematical clarity, and compatibility with standard CPU BLAS routines.
- **Future Runtime**: Document implicit block causal kernels (FlashAttention CPU / custom tiling) as future optimization points.

---

## 9. Step 4H — Forward-Pass Tensor Contract

The complete forward pipeline operates strictly according to the following tensor shapes:

```
[Input IDs]            (B, T)                  dtype=int64
      │
[Embedding Lookup]     (B, T, 192)             dtype=float32
      │
┌─────┴───────────────────────────────────────────────────────┐
│ Transformer Block l ∈ {1..6}                                │
│   ├─ [RMSNorm 1]     (B, T, 192)             dtype=float32  │
│   ├─ [Q, K, V Proj]  (B, T, 192) each        dtype=float32  │
│   ├─ [Head Reshape]  (B, 6, T, 32) each      dtype=float32  │
│   ├─ [RoPE (Q, K)]   (B, 6, T, 32) each      dtype=float32  │
│   ├─ [Scores Q*K^T]  (B, 6, T, T)            dtype=float32  │
│   ├─ [Causal Mask]   (1, 1, T, T)            additive float │
│   ├─ [Softmax Probs] (B, 6, T, T)            dtype=float32  │
│   ├─ [Context S*V]   (B, 6, T, 32)           dtype=float32  │
│   ├─ [Out Project]   (B, T, 192)             dtype=float32  │
│   ├─ [Residual 1]    (B, T, 192)             dtype=float32  │
│   ├─ [RMSNorm 2]     (B, T, 192)             dtype=float32  │
│   ├─ [SwiGLU Gate]   (B, T, 512)             dtype=float32  │
│   ├─ [SwiGLU Up]     (B, T, 512)             dtype=float32  │
│   ├─ [Swish * Up]    (B, T, 512)             dtype=float32  │
│   ├─ [SwiGLU Down]   (B, T, 192)             dtype=float32  │
│   └─ [Residual 2]    (B, T, 192)             dtype=float32  │
└─────────────────────────────────────────────────────────────┘
      │
[Final RMSNorm]        (B, T, 192)             dtype=float32
      │
[Tied LM Head]         (B, T, 4096)            dtype=float32  (W_out = E^T)
```

---

## 10. Step 4I — Memory Budget Analysis

### 10.1 Static Weight Memory
- **Total Parameters**: $3,443,136$
- **FP32 (4 bytes/param)**: $13,772,544\text{ bytes} = 13.13\text{ MiB}$ ($13.77\text{ MB}$)
- **FP16/BF16 (2 bytes/param)**: $6,886,272\text{ bytes} = 6.57\text{ MiB}$ ($6.89\text{ MB}$)
- **INT8 (1 byte/param)**: $3,443,136\text{ bytes} = 3.28\text{ MiB}$ ($3.44\text{ MB}$)
- **INT4 (0.5 bytes/param)**: $1,721,568\text{ bytes} = 1.64\text{ MiB}$ ($1.72\text{ MB}$)

### 10.2 Dynamic KV Cache Memory (Batch Size $B = 1$)
$$\text{Elements} = 2 \times B \times N \times T \times d_{\text{model}} = 2 \times 1 \times 6 \times T \times 192 = 2,304 \times T$$

| Context ($T$) | Total Elements | FP32 Memory | FP16 Memory | INT8 Memory |
| :---: | :---: | :---: | :---: | :---: |
| **$T = 128$** | $294,912$ | $1,179,648\text{ B}$ ($1.125\text{ MiB}$) | $589,824\text{ B}$ ($0.5625\text{ MiB}$) | $294,912\text{ B}$ ($0.281\text{ MiB}$) |
| **$T = 256$** | $589,824$ | $2,359,296\text{ B}$ ($2.250\text{ MiB}$) | $1,179,648\text{ B}$ ($1.125\text{ MiB}$) | $589,824\text{ B}$ ($0.562\text{ MiB}$) |
| **$T = 512$** | $1,179,648$ | $4,718,592\text{ B}$ ($4.500\text{ MiB}$) | $2,359,296\text{ B}$ ($2.250\text{ MiB}$) | $1,179,648\text{ B}$ ($1.125\text{ MiB}$) |

### 10.3 Dynamic Training Activations (Stored for Backpropagation, $B = 1$)

| Context ($T$) | Elements Stored | FP32 Activation RAM | FP16 Activation RAM |
| :---: | :---: | :---: | :---: |
| **$T = 128$** | $3,981,312$ | $15.19\text{ MiB}$ | $7.59\text{ MiB}$ |
| **$T = 256$** | $10,321,920$ | $39.38\text{ MiB}$ | $19.69\text{ MiB}$ |
| **$T = 512$** | $30,081,024$ | $114.75\text{ MiB}$ | $57.38\text{ MiB}$ |

---

## 11. Step 4J — Computational FLOPs Budget

We calculate floating-point operations using standard $2 M K N$ multiply-accumulate accounting:

| Component | Per-Token Formula | Sequence Formula ($T=512$) | FLOPs ($T=512$) | % of Total |
| :--- | :--- | :--- | :---: | :---: |
| **Attention Linear ($Q,K,V,O$)** | $4 \times (2 d^2) \times N$ | $N \times 8 T d^2$ | $905.97\text{ MFLOPs}$ | $18.98\%$ |
| **Attention Quadratic ($QK^T, SV$)** | $4 T d + 3 H T$ | $N \times (4 T^2 d + 3 H T^2)$ | $1,236.27\text{ MFLOPs}$ | $25.90\%$ |
| **SwiGLU FFN** | $(6 d \cdot d_{\text{ff}} + 4 d_{\text{ff}}) \times N$ | $N \times (6 T d \cdot d_{\text{ff}} + 4 T d_{\text{ff}})$ | $1,818.23\text{ MFLOPs}$ | $38.09\%$ |
| **Tied LM Head** | $2 d V$ | $2 T d V$ | $805.31\text{ MFLOPs}$ | $16.87\%$ |
| **RoPE, Norms, Residuals** | $\approx 22 d$ | $T \times \text{misc}$ | $7.27\text{ MFLOPs}$ | $0.15\%$ |
| **Total Sequence Compute** | — | — | **$4,773.05\text{ MFLOPs}$** | **$100.00\%$** |

### Summary across Context Lengths:
- **$T = 128$**: $961.46\text{ MFLOPs}$ total ($7.51\text{ MFLOPs/token}$)
- **$T = 256$**: $2,077.46\text{ MFLOPs}$ total ($8.12\text{ MFLOPs/token}$)
- **$T = 512$**: $4,773.05\text{ MFLOPs}$ total ($9.32\text{ MFLOPs/token}$)

---

## 12. Step 4K — Low-Spec Hardware & Edge CPU Review

All technical conclusions are explicitly categorized according to verification status:

| Analysis Domain | Observation & Technical Finding | Classification |
| :--- | :--- | :---: |
| **RAM Bandwidth Bottleneck** | At batch size 1 autoregressive generation, reading INT8 weights ($3.28\text{ MiB}$) over a $20\text{ GB/s}$ DDR3/DDR4 bus requires $\approx 0.17\text{ ms/token}$. | `[CALCULATED]` |
| **L3 Cache Residency** | On processors with $\ge 8\text{ MB}$ L3 cache, quantized INT8 weights ($3.28\text{ MiB}$) plus KV cache ($1.125\text{ MiB}$) could theoretically fit entirely into CPU cache, bypassing DRAM latency. | `[ENGINEERING HYPOTHESIS]` |
| **Older 28nm CPUs** | On older budget CPUs lacking large L3 cache (e.g. 2MB L2 only), weights must stream from RAM on every token generation. | `[ENGINEERING HYPOTHESIS]` |
| **SIMD Register Mapping** | Head dimension $d_{\text{head}} = 32$ maps cleanly to four 256-bit AVX2 registers, two 512-bit AVX-512 registers, or eight 128-bit ARM NEON registers with zero padding. | `[CALCULATED]` |
| **Multi-Core Threading** | With $H = 6$ attention heads, workload distributes evenly across 1, 2, 3, or 6 threads without fractional head splits. | `[CALCULATED]` |
| **C/C++ & WASM Portability** | Because the architecture uses standard dense GEMMs without custom CUDA kernels, it exports directly to pure C, GGML, ONNX, and WebAssembly. | `[CALCULATED]` |
| **Real Hardware Latency** | Actual tokens-per-second on older Intel/ARM CPUs can only be validated once the C++ runtime is implemented in future steps. | `[UNVERIFIED]` |

---

## 13. Step 4L — Quantization Architecture Plan

The neural core is designed for future post-training quantization and quantization-aware training:

1. **Weight Quantization Targets**:
   - **INT8 (Primary)**: Per-channel symmetric quantization for all linear projections ($W_Q, W_K, W_V, W_O, W_{\text{gate}}, W_{\text{up}}, W_{\text{down}}, W_{\text{head}}$).
   - **INT4 (Experimental)**: Group-wise quantization (group size 32 or 64) for FFN matrices.
2. **High-Precision Components (Kept in FP32)**:
   - RMSNorm scale parameters ($\boldsymbol{\gamma}$) and variance calculation.
   - RoPE trigonometric tables ($\cos, \sin$) and rotation kernel.
   - Softmax normalization in attention.
3. **Accumulator Precision**:
   - INT8 matrix multiplications must accumulate into **INT32** registers before requantization, completely avoiding overflow ($192 \times 127 \times 127 \approx 3.09 \times 10^6 \ll 2^{31}-1$).

---

## 14. Step 4M — Weight Initialization Strategy

To ensure stable training dynamics from step 0:
1. **Embedding Matrix ($E$)**:
   $$\mathbf{E}_{i, j} \sim \mathcal{N}\left(0, \frac{1}{\sqrt{d_{\text{model}}}}\right) = \mathcal{N}\left(0, \frac{1}{\sqrt{192}}\right) \approx \mathcal{N}(0, 0.0722)$$
2. **Projection Matrices ($W_Q, W_K, W_V, W_{\text{gate}}, W_{\text{up}}$)**:
   Truncated normal with standard deviation $\sigma = \sqrt{\frac{2}{d_{\text{in}} + d_{\text{out}}}}$.
3. **Residual Projections ($W_O, W_{\text{down}}$)**:
   To prevent residual stream variance from exploding with depth $N = 6$, scale down by $\frac{1}{\sqrt{2N}}$:
   $$\sigma_{\text{residual}} = \frac{1}{\sqrt{2N}} \times \sqrt{\frac{2}{d_{\text{in}} + d_{\text{out}}}} = \frac{1}{\sqrt{12}} \times \sigma \approx 0.2887 \times \sigma$$
4. **RMSNorm Scales ($\boldsymbol{\gamma}$)**:
   Initialized uniformly to $\mathbf{1.0}$.

---

## 15. Step 4N — Future Training Compatibility

The frozen neural core contract natively supports:
- **Next-Token Prediction**: Target tensor $[B, T]$ shifted by 1 relative to input IDs.
- **Cross-Entropy Loss**: Evaluated over unnormalized logits $[B, T, 4096]$ with `ignore_index = 2` (`<PAD>`).
- **Teacher Forcing**: Full sequence processed in parallel using the causal mask.
- **Mixed Precision**: Forward pass compatible with FP16/BF16 autocasting with FP32 master weights.
- **Gradient Clipping**: Norm clipping at threshold $1.0$ to stabilize early learning.

---

## 16. Step 4O — Failure Modes & Architectural Mitigations

| Failure Mode | Root Cause | Detection | Architectural Mitigation |
| :--- | :--- | :--- | :--- |
| **Gradient Explosion** | Variance accumulation across 6 residual additions | Sudden loss spikes / `NaN` gradients | Pre-RMSNorm ensures unit input variance; depth-scaled $1/\sqrt{2N}$ initialization on $W_O$ and $W_{\text{down}}$. |
| **Attention Entropy Collapse** | Dot products saturate before softmax | Softmax probabilities concentrate on a single token | Scale dot products by $1/\sqrt{d_{\text{head}}} = 1/\sqrt{32} \approx 0.1768$. |
| **Token ID Out-of-Bounds** | Tokenizer emitting IDs $\ge 4096$ | CUDA/CPU assertion error or index fault | Step 2.4 / Step 3 interface contract enforces token ID bounds check $[0, 4095]$ at handoff. |
| **Quantization Clipping** | Outlier activations in intermediate SwiGLU | High perplexity degradation under INT8 | Keep RMSNorm and RoPE in FP32; use per-channel symmetric scales on weight matrices. |

---

## 17. Step 4P — Architectural Alternatives & Tradeoff Analysis

| Component | Selected for v0.1 | Alternative Considered | Engineering Trade-off & Reason for Selection |
| :--- | :--- | :--- | :--- |
| **Attention** | **Simple MHA** ($H_{kv}=6$) | GQA ($H_{kv}=2$) | At $d_{\text{model}}=192$, MHA KV cache is already trivial ($2.25\text{ MiB}$ at FP16). MHA avoids broadcasting overhead on simple CPU loops. |
| **FFN Activation** | **SwiGLU** ($d_{\text{ff}}=512$) | Standard GELU MLP ($4d$) | SwiGLU provides superior representational capacity per parameter; $d_{\text{ff}}=512$ provides exact FLOP parity with 2-matrix $4d$ FFN. |
| **Normalization** | **Pre-RMSNorm** | Post-LayerNorm | Pre-norm eliminates vanishing gradients; RMSNorm eliminates mean computation, saving $7\text{--}12\%$ CPU normalizer latency. |
| **Positional Encoding** | **RoPE** ($\Theta=10000.0$) | Learned Absolute Embeddings | RoPE has zero trainable parameters and relative distance awareness; absolute embeddings add $512 \times 192 = 98,304$ static parameters. |

---

## 18. Step 4Q — Definitive Chakr-Micro v0.1 Configuration

The following configuration is formally ratified for Chakr-Micro v0.1:

| Structural Parameter | Frozen Value |
| :--- | :---: |
| **Architecture** | Decoder-Only Autoregressive Causal Transformer |
| **Vocabulary Size ($V$)** | **$4,096$** |
| **Hidden Dimension ($d_{\text{model}}$)** | **$192$** |
| **Transformer Layers ($N$)** | **$6$** |
| **Attention Heads ($H$)** | **$6$** |
| **Key-Value Heads ($H_{kv}$)** | **$6$** (Simple MHA) |
| **Head Dimension ($d_{\text{head}}$)** | **$32$** |
| **FFN Intermediate Dimension ($d_{\text{ff}}$)** | **$512$** |
| **Maximum Context Window ($T_{\text{max}}$)** | **$512$** |
| **Normalization** | **Pre-RMSNorm** ($\epsilon = 10^{-5}$) |
| **Positional Mechanism** | **RoPE** ($\Theta = 10000.0, d_{\text{rot}} = 32$) |
| **Feed-Forward Activation** | **SwiGLU** |
| **Linear Projection Biases** | **None** ($b = 0$, strictly bias-free) |
| **Weight Tying** | **Enabled** ($W_{\text{out}} = E^T$) |
| **Total Model Parameters** | **$3,443,136$** |

---

## 19. Step 4R — Formal Frozen vs Unfrozen Decision Matrix

| Architectural Element | Status | Modification Policy |
| :--- | :---: | :--- |
| **Model Dimension ($d_{\text{model}} = 192$)** | **FROZEN** | Non-negotiable for v0.1. |
| **Vocabulary Size ($V = 4096$)** | **FROZEN** | Ratified by Step 3 empirical data. |
| **Layer Count ($N = 6$)** | **FROZEN** | Core Chakr-Micro sizing. |
| **Attention Heads ($H = 6, H_{kv} = 6$)** | **FROZEN** | Simple MHA for v0.1. |
| **FFN Dimension ($d_{\text{ff}} = 512$)** | **FROZEN** | Exact $\frac{8}{3} d$ integer alignment. |
| **Normalization (Pre-RMSNorm, $\epsilon=10^{-5}$)** | **FROZEN** | Identity highway, bias-free. |
| **Positional Encoding (RoPE, $\Theta=10000.0$)** | **FROZEN** | Full head rotary encoding. |
| **Weight Tying ($W_{\text{out}} = E^T$)** | **FROZEN** | Saves 786,432 parameters. |
| **Context Window ($T_{\text{max}} = 512$)** | **FROZEN** | Micro-budget bound. |
| **Special Token IDs (0, 1, 2)** | **FROZEN** | Inherited from tokenizer contract. |
| *Learning Rate Schedule* | **UNFROZEN** | Subject to Step 5/6 training research. |
| *Optimizer (AdamW / Sophia / Lion)* | **UNFROZEN** | Subject to empirical training research. |
| *Weight Initialization Scale Tuning* | **UNFROZEN** | Experimental fine-tuning. |
| *Quantization Strategy (INT8 / INT4)* | **UNFROZEN** | Post-training deployment phase. |
| *Future GQA Scaling* | **UNFROZEN** | Extension point for larger models ($d \ge 512$). |

---

## 20. Step 4S — Future Test Suite Specification

The neural core implementation in subsequent steps will be verified against the following mandatory test battery:

1. **Shape Tests**: Every intermediate tensor matches [`get_tensor_forward_contracts`](file:///D:/Project/ChakrView/chakrview/config.py#L251-L290).
2. **Mathematical RMSNorm Test**: Validates mean-independence and scaling against analytical formulas.
3. **RoPE Mathematical Test**: Verifies norm-preservation ($\|\text{RoPE}(x)\| = \|x\|$) and relative shift invariance.
4. **SwiGLU Mathematical Test**: Verifies gating behavior against manual activation calculation.
5. **Causality Invariant Test**: Verifies that modifying token $t+1$ induces zero gradient or output change at token $t$.
6. **Determinism Test**: Identical inputs and identical weights produce bit-exact identical logits.
7. **Gradient & Numerical Stability**: Finite gradients with zero `NaN` or `Inf` across all layers.
8. **Parameter Count Test**: PyTorch / C++ module parameter count equals exactly $3,443,136$.
9. **Weight Tying Identity Test**: Asserts `model.lm_head.weight.data_ptr() == model.embedding.weight.data_ptr()`.
10. **Context Boundary Test**: $T = 512$ executes successfully; $T = 513$ raises dimension error.

---

## 21. Step 4T & 4U — Code Policy & Documentation Verification

- **Code Policy Adherence**: Zero model classes, zero transformer implementations, zero weights, and zero training loops have been introduced.
- **Module Created**: [`chakrview/config.py`](file:///D:/Project/ChakrView/chakrview/config.py) provides only dataclasses, parameter math, memory accounting, and tensor contracts using pure Python standard library.
- **Unit Tests Added**: [`tests/test_neural_core_spec.py`](file:///D:/Project/ChakrView/tests/test_neural_core_spec.py) with 12 unit tests verifying configuration validation, parameter formulas, memory scaling, and tensor shapes.

---

## 22. Step 4V — Final Validation Checklist

- [x] Every tensor shape verified end-to-end.
- [x] Parameter formulas verified ($3,443,136$ total params).
- [x] Memory budgets calculated for $T \in \{128, 256, 512\}$ in FP32, FP16, INT8, INT4.
- [x] FLOP budgets calculated for $T \in \{128, 256, 512\}$.
- [x] Attention head divisibility verified ($192 / 6 = 32$).
- [x] Weight tying shape compatibility confirmed ($[4096, 192]$ and $[192, 4096]$).
- [x] Context length $T_{\text{max}} = 512$ verified.
- [x] Tokenizer interface contract compatibility verified ($V = 4096$, `<BOS>`: 0, `<EOS>`: 1, `<PAD>`: 2).
- [x] Zero pretrained weights or external models introduced.
- [x] All 151 unit tests passing (100% green).
- [x] Working tree ready for clean commit.

---
*End of Neural Core Architecture Specification*
