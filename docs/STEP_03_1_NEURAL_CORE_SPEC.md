# ChakrView Neural Core Architecture & Mathematical Specification (Step 3.1)

**Document Version**: 0.1.0  
**Phase**: Step 3.1 — Neural Core Architecture & Mathematical Specification  
**Model Target**: Chakr-Micro v0.1  
**Status**: APPROVED RESEARCH & MATHEMATICAL SPECIFICATION  
**Scope**: Strictly Mathematical & Structural Specification (No Code, No Pretrained Weights, No Framework Dependencies)  

---

## 1. Audit of Existing Architecture & Continuity

Before formalizing the complete neural core specification for Step 3.1, a rigorous audit of the preceding architecture documents was conducted:
- `docs/STEP_01_NEURAL_CORE_SPEC.md` (Step 1 Foundational Core Spec)
- `docs/ARCHITECTURE_DECISIONS.md` (Architecture Decision Records ADR 01–ADR 10)
- `docs/STEP_02_4_TOKENIZER_INTERFACE.md` (Tokenizer-to-Neural-Core Contract)
- `docs/PROJECT_STATUS.md` (Project Milestones)
- `chakrview/tokenizer/interface.py` (Handoff Implementation Contract)

### 1.1 Frozen Architectural Decisions Maintained
1. **Model Paradigm**: Decoder-only autoregressive causal language model.
2. **Normalization**: Pre-RMSNorm applied before multi-head attention and before SwiGLU FFN, plus a final RMSNorm before unembedding. LayerNorm is strictly excluded.
3. **Positional Mechanism**: Rotary Position Embedding (RoPE) applied to query and key head slices. Absolute learned or sinusoidal embeddings are omitted.
4. **Feed-Forward Activation**: SwiGLU ($W_{\text{down}}(\text{Swish}(W_{\text{gate}}x) \odot W_{\text{up}}x)$).
5. **Linear Projections**: Bias-free linear layers ($b = 0$) across all projections ($W_Q, W_K, W_V, W_O, W_{\text{gate}}, W_{\text{up}}, W_{\text{down}}$).
6. **Attention Mechanism**: Simple Multi-Head Attention (MHA) with $H_{kv} = H = 6$. Grouped-Query Attention (GQA) is strictly an architectural extension point for future versions and is **not** implemented in v0.1.
7. **Weight Tying**: Output projection tied to input embedding matrix ($W_{\text{head}} = W_{\text{out}} = E^T$).
8. **Context Window**: Maximum context $T_{\text{max}} = 512$ tokens for Chakr-Micro.
9. **Zero Pretrained Weights**: 100% indigenous weights initialized from first principles.
10. **Provisional Vocabulary**: $V = 4,096$ experimentally selected from the Step 2.3 benchmark corpus, decoupled from inner layers and replaceable without structural changes.

### 1.2 Verification of Non-Contradiction
- The hidden dimension $d_{\text{model}} = 192$ and head count $H = 6$ yield an integer head dimension $d_{\text{head}} = 32$ without remainder.
- The SwiGLU intermediate dimension $d_{\text{ff}} = 512$ matches the exact theoretical $\frac{8}{3}d_{\text{model}} = \frac{8}{3} \times 192 = 512$ integer relation.
- The embedding table shape $[4096, 192]$ and tied output projection $[192, 4096]$ match the interface frozen in Step 2.4.
- Zero mathematical contradictions were detected across existing specifications.

---

## 2. Complete Architectural Parameter Specification: Chakr-Micro v0.1

The following table pins every structural hyperparameter for the Chakr-Micro v0.1 neural core:

| Architectural Hyperparameter | Notation | Value | Engineering Rationale & Justification |
| :--- | :---: | :---: | :--- |
| **Vocabulary Size** | $V$ | $4,096$ | Provisional candidate selected from Step 2.3 benchmark; balances token fertility ($2.43\text{ tok/word}$ Hindi, $1.77$ Hinglish) with low embedding parameter fraction ($22.84\%$). |
| **Hidden Model Dimension** | $d_{\text{model}}$ | $192$ | Compact representation tailored for low-spec CPU memory bandwidth; enables complete layer working set to remain in fast memory; multiple of 64 bytes. |
| **Number of Transformer Layers** | $L$ | $6$ | Sufficient depth for hierarchical feature extraction (lexical $\to$ syntactic $\to$ associative) while bounding backward pass memory and forward latency on low-spec CPUs. |
| **Number of Attention Heads** | $H$ | $6$ | Equal to layer count; allows $d_{\text{head}} = 32$; symmetric multi-head subspace decomposition. |
| **Number of Key-Value Heads** | $H_{kv}$ | $6$ | Standard Simple Multi-Head Attention (MHA) for v0.1. $H_{kv} = H$ guarantees clean, baseline causal attention. |
| **Head Dimension** | $d_{\text{head}}$ ($d_k$) | $32$ | Exactly $192 / 6 = 32$; perfectly aligns with 256-bit SIMD registers ($8 \times \text{float32}$) and 512-bit registers ($16 \times \text{float32}$); divisible by 2 for RoPE ($16$ 2D rotation planes). |
| **SwiGLU Hidden Dimension** | $d_{\text{ff}}$ | $512$ | Exact $\frac{8}{3} \times 192 = 512$; power of 2 ($2^9$); aligns with hardware cache line boundaries ($512 \times 4\text{ bytes} = 2,048\text{ bytes} = 32$ cache lines). |
| **Maximum Sequence Context** | $T_{\text{max}}$ | $512$ | Tailored for micro-models; quadratic attention footprint bounded to $512^2 \times 4\text{ bytes} = 1\text{ MB}$ per head, entirely preventing CPU out-of-memory. |
| **Normalization Type** | — | Pre-RMSNorm | Root Mean Square Normalization with zero mean-centering; eliminates $7\text{--}12\%$ normalizer compute on CPU. |
| **Normalization Epsilon** | $\epsilon$ | $10^{-5}$ | Prevents division-by-zero during RMS computation; maintains numerical headroom in FP16/BF16 arithmetic. |
| **Positional Mechanism** | — | RoPE | Rotary Position Embeddings applied to $Q$ and $K$ heads; zero trainable parameter overhead; strict relative distance awareness. |
| **RoPE Base Frequency** | $\Theta$ | $10000.0$ | Standard canonical rotary base; provides smooth relative angular decay across sequence positions $m \in [0, 511]$. |
| **RoPE Rotary Dimension** | $d_{\text{rot}}$ | $32$ | Full rotation ($d_{\text{rot}} = d_{\text{head}} = 32$); all head dimensions participate in rotary encoding. |
| **Linear Layer Biases** | $b$ | None ($0$) | All projections are strictly bias-free ($W \cdot x$), saving memory and eliminating linear bias memory-drift during long generations. |
| **Weight Tying** | — | Enabled | Output LM head shares weights with input token embeddings ($W_{\text{out}} = E^T$); saves $786,432$ parameters. |
| **Output Logits Dimension** | — | $[B, T, 4096]$ | Emits unnormalized categorical log-probabilities over the discrete vocabulary $V$. |

---

## 3. Attention Design: Simple Multi-Head Attention (MHA)

Chakr-Micro v0.1 implements strictly **Simple Multi-Head Attention (MHA)**. Grouped-Query Attention (GQA) is strictly an architectural extension point for future model iterations and will **NOT** be implemented in v0.1.

### 3.1 Input Formulation
Let the pre-normalized activation tensor entering the attention sub-layer of layer $l \in \{1, \dots, L\}$ be:
$$\hat{h} = \text{RMSNorm}_1(h_{l-1}) \in \mathbb{R}^{B \times T \times d_{\text{model}}}$$
where $B$ is batch size, $T$ is sequence length ($T \le T_{\text{max}}$), and $d_{\text{model}} = 192$.

### 3.2 Bias-Free Linear Projections
Three independent, bias-free linear transformations project the normalized state into query, key, and value representations:
$$Q_{\text{flat}} = \hat{h} W_Q \in \mathbb{R}^{B \times T \times d_{\text{model}}}, \quad W_Q \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}$$
$$K_{\text{flat}} = \hat{h} W_K \in \mathbb{R}^{B \times T \times d_{\text{model}}}, \quad W_K \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}$$
$$V_{\text{flat}} = \hat{h} W_V \in \mathbb{R}^{B \times T \times d_{\text{model}}}, \quad W_V \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}$$
where $W_Q, W_K, W_V \in \mathbb{R}^{192 \times 192}$.

### 3.3 Head Partitioning & Transposition
The projected representations are decomposed into $H = 6$ parallel attention heads of dimension $d_{\text{head}} = 32$:
$$Q = \text{Permute}(\text{Reshape}(Q_{\text{flat}}, (B, T, H, d_{\text{head}})), (0, 2, 1, 3)) \in \mathbb{R}^{B \times H \times T \times d_{\text{head}}}$$
$$K = \text{Permute}(\text{Reshape}(K_{\text{flat}}, (B, T, H, d_{\text{head}})), (0, 2, 1, 3)) \in \mathbb{R}^{B \times H \times T \times d_{\text{head}}}$$
$$V = \text{Permute}(\text{Reshape}(V_{\text{flat}}, (B, T, H, d_{\text{head}})), (0, 2, 1, 3)) \in \mathbb{R}^{B \times H \times T \times d_{\text{head}}}$$

### 3.4 Rotary Position Application
Before dot-product scoring, Rotary Position Embeddings are applied to $Q$ and $K$:
$$\tilde{Q} = \text{RoPE}(Q) \in \mathbb{R}^{B \times H \times T \times d_{\text{head}}}$$
$$\tilde{K} = \text{RoPE}(K) \in \mathbb{R}^{B \times H \times T \times d_{\text{head}}}$$
*(Value tensor $V$ receives no positional encoding).*

### 3.5 Scaled Dot-Product Causal Attention Scores
The attention affinity matrix between sequence positions is calculated via batched matrix multiplication scaled by $1 / \sqrt{d_{\text{head}}}$:
$$S_{\text{raw}} = \frac{\tilde{Q} \tilde{K}^T}{\sqrt{d_{\text{head}}}} \in \mathbb{R}^{B \times H \times T \times T}$$
where $\sqrt{d_{\text{head}}} = \sqrt{32} \approx 5.656854$.

The causal mask $M_{\text{causal}} \in \mathbb{R}^{1 \times 1 \times T \times T}$ prevents attention to future positions:
$$M_{\text{causal}}(i, j) = \begin{cases} 0.0 & \text{if } i \ge j \\ -\infty & \text{if } i < j \end{cases}$$
$$S = S_{\text{raw}} + M_{\text{causal}} \in \mathbb{R}^{B \times H \times T \times T}$$

### 3.6 Attention Probabilities & Context Aggregation
Attention probabilities are formed via numerically stable row-wise softmax along the final sequence dimension:
$$P = \text{Softmax}(S, \text{dim}=-1) \in \mathbb{R}^{B \times H \times T \times T}$$
The context vectors for all heads are aggregated:
$$O_{\text{heads}} = P V \in \mathbb{R}^{B \times H \times T \times d_{\text{head}}}$$

### 3.7 Output Projection & Head Concatenation
The multi-head context is permuted, contiguous-reshaped back to the original model dimension, and linearly projected:
$$O_{\text{flat}} = \text{Reshape}(\text{Permute}(O_{\text{heads}}, (0, 2, 1, 3)), (B, T, d_{\text{model}})) \in \mathbb{R}^{B \times T \times d_{\text{model}}}$$
$$h_{\text{attn}} = O_{\text{flat}} W_O \in \mathbb{R}^{B \times T \times d_{\text{model}}}, \quad W_O \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}} = \mathbb{R}^{192 \times 192}$$

### 3.8 Attention Residual Connection
The attention output is merged into the incoming residual stream:
$$h_{\text{mid}} = h_{l-1} + h_{\text{attn}} \in \mathbb{R}^{B \times T \times d_{\text{model}}}$$

---

## 4. Head Dimension Verification & Hardware SIMD Alignment

### 4.1 Exact Integer Division Verification
$$d_{\text{head}} = \frac{d_{\text{model}}}{H} = \frac{192}{6} = 32 \quad (\text{Exact Integer: Remainder } = 0)$$

> **CRITICAL VERIFICATION**:
> $192 = 6 \times 32$. There is zero fractional rounding.
> Every head operates on an exact 32-dimensional subspace.

### 4.2 Engineering Justification for $d_{\text{head}} = 32$
1. **SIMD Vector Lane Geometry**:
   - Modern x86 CPUs with **AVX2** feature 256-bit vector registers accommodating $8 \times 32\text{-bit float}$ values. A 32-element vector maps to exactly $4$ AVX2 vector registers ($4 \times 8 = 32$).
   - High-performance x86 CPUs with **AVX-512** feature 512-bit registers accommodating $16 \times \text{float32}$ values. A 32-element vector maps to exactly $2$ AVX-512 registers ($2 \times 16 = 32$).
   - Modern ARM CPUs with **NEON** feature 128-bit registers accommodating $4 \times \text{float32}$ values. A 32-element vector maps to exactly $8$ NEON registers ($8 \times 4 = 32$).
   - Hardware loops unroll cleanly with zero remainder or padding loops.
2. **RoPE Compatibility**:
   - RoPE operates on 2D pairs. $d_{\text{head}} = 32$ decomposes into exactly $16$ independent 2D rotation planes.
3. **Numerical Conditioning**:
   - The dot-product scaling factor $\frac{1}{\sqrt{32}} \approx 0.176777$ maintains unit variance of attention logits under Gaussian inputs, preventing early saturation of the softmax function.
4. **Subspace Expressivity**:
   - 6 heads with dimension 32 offer superior diversity over fewer larger heads (e.g., 2 heads of 96) for tracking local syntax, n-grams, and long-range dependencies simultaneously.

---

## 5. Rotary Position Embeddings (RoPE) Mathematical Specification

### 5.1 Placement in Computational Graph
RoPE is applied exclusively to the Query ($Q$) and Key ($K$) representations **inside the attention block**, immediately after head reshaping and immediately **before** computing the scaled dot products $Q K^T$. RoPE is **not** applied to Values ($V$) and is **not** applied to the residual stream.

```
Linear Projection (Q, K) -> Reshape [B, H, T, 32] -> RoPE(Q, K) -> Q K^T -> Softmax
```

### 5.2 Frequency Spectrum Calculation
For rotary dimension $d_{\text{rot}} = d_{\text{head}} = 32$, the frequency vector $\Theta \in \mathbb{R}^{16}$ is defined by:
$$\theta_k = b^{-2k / d_{\text{head}}} = 10000^{-2k / 32} = 10000^{-k / 16}, \quad k \in \{0, 1, \dots, 15\}$$

Numerical values for the 16 base frequencies ($k = 0 \dots 15$):
- $\theta_0 = 10000^{0} = 1.000000$ (fastest rotating plane, wavelength $2\pi \approx 6.28$ tokens)
- $\theta_1 = 10000^{-1/16} \approx 0.562341$
- $\theta_2 = 10000^{-2/16} \approx 0.316228$
- $\dots$
- $\theta_{15} = 10000^{-15/16} \approx 0.000178$ (slowest rotating plane, wavelength $\approx 35,340$ tokens)

### 5.3 Pairwise 2D Rotation Formulation
Let $\mathbf{v} \in \mathbb{R}^{d_{\text{head}}}$ be a head vector (from $Q$ or $K$) at sequence index $m \in \{0, 1, \dots, T-1\}$.  
The vector is partitioned into $16$ consecutive 2D coordinates:
$$\mathbf{v}^{(k)} = \begin{pmatrix} v_{2k} \\ v_{2k+1} \end{pmatrix} \in \mathbb{R}^2, \quad k \in \{0, 1, \dots, 15\}$$

Each coordinate pair is multiplied by an orthogonal 2D Givens rotation matrix:
$$\mathcal{R}_m^{(k)} \mathbf{v}^{(k)} = \begin{pmatrix} \cos(m \theta_k) & -\sin(m \theta_k) \\ \sin(m \theta_k) & \cos(m \theta_k) \end{pmatrix} \begin{pmatrix} v_{2k} \\ v_{2k+1} \end{pmatrix} = \begin{pmatrix} v_{2k} \cos(m \theta_k) - v_{2k+1} \sin(m \theta_k) \\ v_{2k} \sin(m \theta_k) + v_{2k+1} \cos(m \theta_k) \end{pmatrix}$$

### 5.4 Relative Displacement Invariant
For token positions $m$ (query) and $n$ (key), the inner product across all rotation planes satisfies:
$$\langle \text{RoPE}(\mathbf{q}, m), \text{RoPE}(\mathbf{k}, n) \rangle = \sum_{k=0}^{15} \left( \mathcal{R}_m^{(k)} \mathbf{q}^{(k)} \right)^T \left( \mathcal{R}_n^{(k)} \mathbf{k}^{(k)} \right) = \sum_{k=0}^{15} \left( \mathbf{q}^{(k)} \right)^T \mathcal{R}_{m - n}^{(k)} \mathbf{k}^{(k)}$$
The dot product depends strictly on relative offset $(m - n)$, preserving shift invariance across the context window.

### 5.5 Numerical Precision Considerations
- Precomputed $\sin$ and $\cos$ tables of shape $[1, 1, T_{\text{max}}, d_{\text{head}}]$ must be generated and stored in `float32`.
- When operating in lower precisions (FP16/BF16), table lookup values are cast to match activation precision at the point of multiplication.

---

## 6. Pre-RMSNorm Specification

Chakr-Micro v0.1 enforces strict **Pre-RMSNorm**. LayerNorm (with mean subtraction) is completely omitted.

### 6.1 Mathematical Formulation
For an input vector $\mathbf{u} \in \mathbb{R}^{d_{\text{model}}}$ ($d_{\text{model}} = 192$):
$$\text{RMSNorm}(\mathbf{u}) = \frac{\mathbf{u}}{\text{RMS}(\mathbf{u}) + \epsilon} \odot \boldsymbol{\gamma}$$
where:
$$\text{RMS}(\mathbf{u}) = \sqrt{\frac{1}{d_{\text{model}}} \sum_{j=1}^{d_{\text{model}}} u_j^2} = \sqrt{\frac{1}{192} \sum_{j=1}^{192} u_j^2}$$
- $\boldsymbol{\gamma} \in \mathbb{R}^{d_{\text{model}}}$ is a learnable scaling parameter vector, initialized to $\boldsymbol{\gamma} = \mathbf{1}_{192}$.
- $\epsilon = 10^{-5}$ is the numerical stabilization constant.
- Additive bias $\boldsymbol{\beta}$ is **omitted** ($\boldsymbol{\beta} = \mathbf{0}$).

### 6.2 Pre-Norm Structural Placement
1. **$\text{RMSNorm}_{1, l}$**: Applied to the residual stream $h_{l-1}$ immediately prior to the attention QKV projections.
2. **$\text{RMSNorm}_{2, l}$**: Applied to the mid-block state $h_{\text{mid}}$ immediately prior to the SwiGLU FFN projections.
3. **$\text{RMSNorm}_{\text{final}}$**: Applied to the final transformer output $h_L$ immediately prior to the tied LM unembedding head.

Zero normalizations are applied inside residual addition branches, preserving an unimpeded gradient identity highway:
$$\frac{\partial h_L}{\partial h_0} = \mathbf{I} + \sum_{l=1}^L \frac{\partial \Delta_l}{\partial h_{l-1}}$$

---

## 7. SwiGLU Feed-Forward Network (FFN) Specification

### 7.1 Mathematical Formulation
The feed-forward sub-layer is formulated as a three-matrix Gated Linear Unit with Swish ($\text{SiLU}$) non-linearity:
$$\hat{h}_{\text{mid}} = \text{RMSNorm}_{2, l}(h_{\text{mid}}) \in \mathbb{R}^{B \times T \times d_{\text{model}}}$$
$$\text{gate} = \hat{h}_{\text{mid}} W_{\text{gate}} \in \mathbb{R}^{B \times T \times d_{\text{ff}}}, \quad W_{\text{gate}} \in \mathbb{R}^{d_{\text{model}} \times d_{\text{ff}}}$$
$$\text{up} = \hat{h}_{\text{mid}} W_{\text{up}} \in \mathbb{R}^{B \times T \times d_{\text{ff}}}, \quad W_{\text{up}} \in \mathbb{R}^{d_{\text{model}} \times d_{\text{ff}}}$$
$$\text{activated} = \text{Swish}(\text{gate}) \odot \text{up} \in \mathbb{R}^{B \times T \times d_{\text{ff}}}$$
$$h_{\text{ffn}} = \text{activated} W_{\text{down}} \in \mathbb{R}^{B \times T \times d_{\text{model}}}, \quad W_{\text{down}} \in \mathbb{R}^{d_{\text{ff}} \times d_{\text{model}}}$$
where:
$$\text{Swish}(z) = z \cdot \sigma(z) = \frac{z}{1 + e^{-z}}$$
$\odot$ represents element-wise (Hadamard) multiplication. All projection matrices ($W_{\text{gate}}, W_{\text{up}}, W_{\text{down}}$) are bias-free.

### 7.2 Dimensional Transitions
$$d_{\text{model}} (192) \xrightarrow{W_{\text{gate}}, W_{\text{up}}} d_{\text{ff}} (512) \xrightarrow{\text{Swish} \odot} d_{\text{ff}} (512) \xrightarrow{W_{\text{down}}} d_{\text{model}} (192)$$

### 7.3 Exact Derivation of $d_{\text{ff}} = 512$
Standard practice in modern LLM architecture (PaLM, LLaMA) specifies that a 3-matrix SwiGLU layer should possess parameter and FLOP equivalence to a standard 2-matrix FFN with expansion factor $4d_{\text{model}}$.
In a 2-matrix FFN: $\text{Parameters} = 2 \times d_{\text{model}} \times (4 d_{\text{model}}) = 8 d_{\text{model}}^2$.  
In a 3-matrix SwiGLU: $\text{Parameters} = 3 \times d_{\text{model}} \times d_{\text{ff}}$.  
Equating the parameters:
$$3 d_{\text{model}} d_{\text{ff}} \approx 8 d_{\text{model}}^2 \implies d_{\text{ff}} \approx \frac{8}{3} d_{\text{model}}$$
For $d_{\text{model}} = 192$:
$$d_{\text{ff}} = \frac{8}{3} \times 192 = 8 \times 64 = 512 \quad (\text{Exact Integer!})$$

$d_{\text{ff}} = 512$ is an exact integer multiple of 64 and a pure power of 2 ($2^9$), ensuring optimal cache-line memory alignment and GEMM kernel efficiency on CPU hardware.

---

## 8. Residual Stream Architecture & Highway Dynamics

Each of the $L = 6$ stacked Transformer blocks executes two pre-norm residual stages:

```
        h_{l-1} (Incoming Residual State: [B, T, 192])
          │
          ├───► RMSNorm_1 ──► MHA (Q, K, V, RoPE, Attn, W_O) ──┐
          │                                                      ▼
         (+) ◄───────────────────────────────────────────────────┘ (Residual 1)
          │
        h_{mid} (Mid-Block State: [B, T, 192])
          │
          ├───► RMSNorm_2 ──► SwiGLU (Gate, Up, Act, W_down) ───┐
          │                                                      ▼
         (+) ◄───────────────────────────────────────────────────┘ (Residual 2)
          │
         h_l (Outgoing Residual State: [B, T, 192])
```

### Mathematical Definitions
1. **Attention Sub-Block**:
   $$h_{\text{mid}} = h_{l-1} + \text{Attention}(\text{RMSNorm}_{1, l}(h_{l-1}))$$
2. **Feed-Forward Sub-Block**:
   $$h_l = h_{\text{mid}} + \text{SwiGLU}(\text{RMSNorm}_{2, l}(h_{\text{mid}}))$$

### Residual Invariants
- The residual stream dimension is invariant: $\dim(h_l) = d_{\text{model}} = 192$ for all $l \in \{0, \dots, L\}$.
- Projections $W_O$ and $W_{\text{down}}$ operate inside their respective sub-layers before adding to the residual highway.
- No scale, multiplier, or normalization is placed on the direct residual connection, preventing gradient attenuation across depth.

---

## 9. Complete End-to-End Forward Pass & Tensor Shapes

Let the input sequence of token IDs be $X \in \mathbb{N}^{B \times T}$ where $X_{b, t} \in [0, 4095]$.

| Step | Operation Description | Formula | Output Variable | Exact Tensor Shape | Data Type |
| :---: | :--- | :--- | :---: | :---: | :---: |
| **1** | Token Embedding Lookup | $E[X]$ | $h_0$ | `(B, T, 192)` | `float32` |
| **2a**| Block $l$ Attn Pre-Norm | $\text{RMSNorm}_{1, l}(h_{l-1})$ | $\hat{h}_1$ | `(B, T, 192)` | `float32` |
| **2b**| Q Projection | $\hat{h}_1 W_{Q, l}$ | $Q_{\text{flat}}$ | `(B, T, 192)` | `float32` |
| **2c**| K Projection | $\hat{h}_1 W_{K, l}$ | $K_{\text{flat}}$ | `(B, T, 192)` | `float32` |
| **2d**| V Projection | $\hat{h}_1 W_{V, l}$ | $V_{\text{flat}}$ | `(B, T, 192)` | `float32` |
| **2e**| Head Decomposition | Permute/Reshape | $Q, K, V$ | `(B, 6, T, 32)` | `float32` |
| **2f**| RoPE Rotary Encoding | $\text{RoPE}(Q), \text{RoPE}(K)$ | $\tilde{Q}, \tilde{K}$ | `(B, 6, T, 32)` | `float32` |
| **2g**| Causal Dot-Product Scoring | $(\tilde{Q} \tilde{K}^T)/\sqrt{32} + M_{\text{causal}}$ | $S$ | `(B, 6, T, T)` | `float32` |
| **2h**| Attention Softmax | $\text{Softmax}(S, \text{dim}=-1)$ | $P$ | `(B, 6, T, T)` | `float32` |
| **2i**| Context Aggregation | $P V$ | $O_{\text{heads}}$ | `(B, 6, T, 32)` | `float32` |
| **2j**| Head Concatenation | Reshape/Permute | $O_{\text{flat}}$ | `(B, T, 192)` | `float32` |
| **2k**| Attn Output Projection | $O_{\text{flat}} W_{O, l}$ | $h_{\text{attn}}$ | `(B, T, 192)` | `float32` |
| **2l**| Residual 1 Merge | $h_{l-1} + h_{\text{attn}}$ | $h_{\text{mid}}$ | `(B, T, 192)` | `float32` |
| **2m**| Block $l$ FFN Pre-Norm | $\text{RMSNorm}_{2, l}(h_{\text{mid}})$ | $\hat{h}_2$ | `(B, T, 192)` | `float32` |
| **2n**| Gate Linear Projection | $\hat{h}_2 W_{\text{gate}, l}$ | $\text{gate}$ | `(B, T, 512)` | `float32` |
| **2o**| Up Linear Projection | $\hat{h}_2 W_{\text{up}, l}$ | $\text{up}$ | `(B, T, 512)` | `float32` |
| **2p**| SwiGLU Activation | $(\text{gate} \cdot \sigma(\text{gate})) \odot \text{up}$ | $\text{act}$ | `(B, T, 512)` | `float32` |
| **2q**| Down Linear Projection | $\text{act} W_{\text{down}, l}$ | $h_{\text{ffn}}$ | `(B, T, 192)` | `float32` |
| **2r**| Residual 2 Merge | $h_{\text{mid}} + h_{\text{ffn}}$ | $h_l$ | `(B, T, 192)` | `float32` |
| **3** | Final Normalization | $\text{RMSNorm}_{\text{final}}(h_L)$ | $h_{\text{final}}$ | `(B, T, 192)` | `float32` |
| **4** | Tied Logits Projection | $h_{\text{final}} E^T$ | $\mathbf{z}$ | `(B, T, 4096)` | `float32` |

---

## 10. Causal Attention Masking Specification

To preserve causal autoregressive generation, token position $i$ must not attend to any future position $j$ where $j > i$.

### 10.1 Explicit Additive Attention Mask Matrix
In training and parallel sequence evaluation, an explicit upper-triangular mask $M \in \mathbb{R}^{1 \times 1 \times T \times T}$ is added to raw dot products:
$$M_{i, j} = \begin{cases} 0.0 & \text{if } i \ge j \\ -\infty & \text{if } i < j \end{cases}$$
In floating-point implementations, $-\infty$ is represented by the lower limit of the data type (e.g., $-10^{9}$ or $-10^{4}$ in FP16, or `-torch.finfo(dtype).max`). When fed into $\text{Softmax}$, $\exp(-\infty) = 0.0$, strictly nullifying attention weights to future tokens.

### 10.2 Generation-Time Implicit Triangular Computation
During autoregressive inference ($B=1$, generating token at step $T$), the query vector length is strictly $1$ ($Q \in \mathbb{R}^{1 \times H \times 1 \times d_{\text{head}}}$), while the Key tensor spans the past context length $K \in \mathbb{R}^{1 \times H \times T_{\text{past}} \times d_{\text{head}}}$.  
Because all historical keys satisfy $j \le i$, no future tokens exist in the KV cache:
$$S = \frac{\tilde{Q} \tilde{K}^T}{\sqrt{d_{\text{head}}}} \in \mathbb{R}^{1 \times H \times 1 \times T_{\text{past}}}$$
At inference, no $T \times T$ mask matrix is instantiated. Softmax is evaluated directly over the $1 \times T_{\text{past}}$ row vector, reducing memory and computation from $O(T^2)$ to $O(T)$.

---

## 11. Language Model Head & Weight Tying Specification

### 11.1 Weight Tying Definition
The unembedding output projection $W_{\text{head}} \in \mathbb{R}^{d_{\text{model}} \times V}$ is algebraically tied to the input token embedding matrix $E \in \mathbb{R}^{V \times d_{\text{model}}}$:
$$W_{\text{head}} = E^T \in \mathbb{R}^{192 \times 4096}$$
The forward logits computation is:
$$\mathbf{z} = h_{\text{final}} E^T \in \mathbb{R}^{B \times T \times 4096}$$

### 11.2 Parameter & Dimensional Verification
- Embedding Matrix $E$: $[4096, 192]$ $\to 4,096 \times 192 = 786,432$ parameters.
- Unembedding Matrix $W_{\text{head}}$: Transpose of $E$. **$0$ additional unique parameters**.
- Final Hidden State: $[B, T, 192]$.
- Output Logits Matrix $\mathbf{z}$: $[B, T, 4096]$.

### 11.3 Engineering Justification for Small-Model Weight Tying
1. **Parameter Economy**: For a micro-model, an independent linear head would consume an additional $786,432$ parameters, inflating the non-embedding parameter budget by nearly $30\%$. Weight tying eliminates this footprint completely.
2. **Memory Footprint**: On memory-constrained devices, keeping a single physical copy of the $4096 \times 192$ weight table avoids duplicating 3 MB of FP32 RAM or 1.5 MB of FP16 RAM.
3. **Semantic Coherence**: Enforcing $W_{\text{head}} = E^T$ forces the input token semantic space to align geometrically with the output prediction space, acting as a strong regularizer that accelerates convergence on small corpora.

---

## 12. Analytical Parameter Accounting: Chakr-Micro v0.1

This section provides exact, non-rounded parameter counts for all components of Chakr-Micro v0.1.

### 12.1 Analytical Formulas
- **Input Embeddings**: $P_{\text{embed}} = V \cdot d_{\text{model}}$
- **Attention Sub-Layer** (per layer):
  - Pre-RMSNorm: $d_{\text{model}}$
  - Query Projection: $d_{\text{model}} \cdot (H \cdot d_{\text{head}}) = d_{\text{model}}^2$
  - Key Projection: $d_{\text{model}} \cdot (H \cdot d_{\text{head}}) = d_{\text{model}}^2$
  - Value Projection: $d_{\text{model}} \cdot (H \cdot d_{\text{head}}) = d_{\text{model}}^2$
  - Output Projection: $(H \cdot d_{\text{head}}) \cdot d_{\text{model}} = d_{\text{model}}^2$
  - Total Attention Parameters per layer: $d_{\text{model}} + 4 d_{\text{model}}^2$
- **SwiGLU Sub-Layer** (per layer):
  - Pre-RMSNorm: $d_{\text{model}}$
  - Gate Projection: $d_{\text{model}} \cdot d_{\text{ff}}$
  - Up Projection: $d_{\text{model}} \cdot d_{\text{ff}}$
  - Down Projection: $d_{\text{ff}} \cdot d_{\text{model}}$
  - Total SwiGLU Parameters per layer: $d_{\text{model}} + 3 (d_{\text{model}} \cdot d_{\text{ff}})$
- **Total Single Transformer Block**:
  $$P_{\text{block}} = 2 d_{\text{model}} + 4 d_{\text{model}}^2 + 3 (d_{\text{model}} \cdot d_{\text{ff}})$$
- **Final Layer Normalization**: $P_{\text{final}} = d_{\text{model}}$
- **Unembedding Head** (Tied): $P_{\text{head}} = 0$
- **Total Model Parameters**:
  $$P_{\text{total}} = V \cdot d_{\text{model}} + L \cdot P_{\text{block}} + d_{\text{model}}$$

### 12.2 Per-Layer Exact Numerical Breakdown ($d_{\text{model}} = 192, d_{\text{ff}} = 512, H = 6, d_{\text{head}} = 32$)

| Sub-Component | Shape | Calculation Formula | Exact Parameter Count |
| :--- | :---: | :---: | :---: |
| **RMSNorm 1 (Pre-Attention)** | `[192]` | $192$ | $192$ |
| **$W_Q$ Projection** | `[192, 192]` | $192 \times 192$ | $36,864$ |
| **$W_K$ Projection** | `[192, 192]` | $192 \times 192$ | $36,864$ |
| **$W_V$ Projection** | `[192, 192]` | $192 \times 192$ | $36,864$ |
| **$W_O$ Output Projection** | `[192, 192]` | $192 \times 192$ | $36,864$ |
| **Total Attention Sub-Layer** | — | $192 + 4 \times 36,864$ | **$147,648$** |
| **RMSNorm 2 (Pre-FFN)** | `[192]` | $192$ | $192$ |
| **$W_{\text{gate}}$ Projection** | `[192, 512]` | $192 \times 512$ | $98,304$ |
| **$W_{\text{up}}$ Projection** | `[192, 512]` | $192 \times 512$ | $98,304$ |
| **$W_{\text{down}}$ Projection** | `[512, 192]` | $512 \times 192$ | $98,304$ |
| **Total SwiGLU Sub-Layer** | — | $192 + 3 \times 98,304$ | **$295,104$** |
| **Total Single Transformer Block ($P_{\text{block}}$)** | — | $147,648 + 295,104$ | **$442,752$** |

### 12.3 Full Model Parameter Total ($L = 6, V = 4096$)

| Section | Parameter Calculation | Exact Parameter Count | Fraction of Model |
| :--- | :--- | :---: | :---: |
| **Token Embeddings ($E$)** | $4096 \times 192$ | $786,432$ | $22.84\%$ |
| **Transformer Blocks ($L = 6$)** | $6 \times 442,752$ | $2,656,512$ | $77.15\%$ |
| **Final RMSNorm ($\boldsymbol{\gamma}_{\text{final}}$)** | $192$ | $192$ | $0.01\%$ |
| **Tied LM Head ($W_{\text{head}} = E^T$)** | Shared with $E$ | $0$ | $0.00\%$ |
| **Grand Total Model Parameters** | $786,432 + 2,656,512 + 192$ | **$3,443,136$** | **$100.00\%$** |

---

## 13. Memory Accounting: Static Weight Storage vs. Runtime Memory

### 13.1 Static Weight Storage Across Precisions ($3,443,136$ parameters)

| Storage Precision | Bytes per Param | Exact Bytes | Mebibytes (MiB) | Megabytes (MB) |
| :--- | :---: | :---: | :---: | :---: |
| **FP32 (Standard Float)** | $4$ | $13,772,544$ | $13.13\text{ MiB}$ | $13.77\text{ MB}$ |
| **FP16 / BF16 (Half Precision)**| $2$ | $6,886,272$ | $6.57\text{ MiB}$ | $6.89\text{ MB}$ |
| **INT8 (Quantized Integer)** | $1$ | $3,443,136$ | $3.28\text{ MiB}$ | $3.44\text{ MB}$ |
| **INT4 (Sub-byte Quantized)** | $0.5$ | $1,721,568$ | $1.64\text{ MiB}$ | $1.72\text{ MB}$ |

### 13.2 Rigorous Separation: Static Weights vs. Runtime Memory
The figures above represent **STATIC WEIGHT STORAGE ONLY**.  
Under no circumstances should static weight storage be confused with the total runtime memory footprint.

**Main Runtime Memory Consumers**:
1. **Training Activation Footprint**: Forward-pass activation tensors cached for backpropagation. For batch size $B=4, T=512$, peak activation memory reaches $\approx 45\text{--}80\text{ MB}$.
2. **Optimizer States**: AdamW tracks two full 32-bit float moments ($m_t, v_t$) per trainable parameter ($2 \times 4 = 8\text{ bytes/param} = 27.55\text{ MB}$). Master FP32 weights and gradients require an additional $8\text{ bytes/param}$. Peak training RAM budget is $\approx 180\text{ MB}$.
3. **Inference Key-Value (KV) Cache**: Described in Section 14 below.
4. **Intermediate Scratch Buffers**: Dot product matrices ($S \in \mathbb{R}^{B \times H \times T \times T}$), SwiGLU activation tensors, and temporary GEMM working memory.
5. **Runtime Engine & Allocator Overhead**: Memory fragmentation, PyTorch/NumPy slab allocators, and C runtime overhead ($\approx 30\text{--}100\text{ MB}$).

> **CACHE LOCALITY CLAIM POLICY**:
> We explicitly do **NOT** claim that Chakr-Micro is guaranteed to execute entirely within CPU L3 cache.
> Rather, the INT8 static footprint ($3.28\text{ MiB}$) makes cache-local residency plausible on modern processors with $16\text{--}32\text{ MB}$ L3 cache. This is formulated as an **empirical engineering hypothesis** to be verified by hardware benchmarking in future steps.

---

## 14. Key-Value (KV) Cache Memory Analysis

During autoregressive inference, previously computed Key and Value states are retained to avoid recomputing historical attention contexts.

### 14.1 Exact KV-Cache Dimension Formula
For batch size $B$, sequence context $T$, $L = 6$ layers, $H_{kv} = 6$ heads, and $d_{\text{head}} = 32$:
$$\text{Elements per Token (Single Layer)} = 2 \times H_{kv} \times d_{\text{head}} = 2 \times 6 \times 32 = 384\text{ elements}$$
$$\text{Elements per Token (All } L=6 \text{ Layers)} = 6 \times 384 = 2,304\text{ elements}$$
$$\text{Total Elements for Context } T = 2,304 \times T\text{ elements}$$

### 14.2 KV-Cache Memory Across Sequence Lengths ($B = 1$)

| Context Length ($T$) | Total Elements | FP32 ($4\text{ bytes}$) | FP16 ($2\text{ bytes}$) | INT8 ($1\text{ byte}$) |
| :---: | :---: | :---: | :---: | :---: |
| **$T = 128$ tokens** | $294,912$ | $1.125\text{ MiB}$ ($1.18\text{ MB}$) | $0.562\text{ MiB}$ ($0.59\text{ MB}$) | $0.281\text{ MiB}$ ($0.29\text{ MB}$) |
| **$T = 256$ tokens** | $589,824$ | $2.250\text{ MiB}$ ($2.36\text{ MB}$) | $1.125\text{ MiB}$ ($1.18\text{ MB}$) | $0.562\text{ MiB}$ ($0.59\text{ MB}$) |
| **$T = 512$ tokens ($T_{\text{max}}$)**| **$1,179,648$** | **$4.500\text{ MiB}$ ($4.72\text{ MB}$)** | **$2.250\text{ MiB}$ ($2.36\text{ MB}$)** | **$1.125\text{ MiB}$ ($1.18\text{ MB}$)** |

### 14.3 Theoretical Tensor Size vs. Implementation Overhead
The calculations above reflect theoretical raw tensor storage ($2,304 \times T \times \text{bytes}$). Real-world inference engines introduce overhead:
- **Allocation Granularity**: Allocating dynamically on every token step causes memory fragmentation. Production runtimes pre-allocate fixed blocks (e.g., in pages of 16 tokens).
- **Alignment Padding**: Tensors aligned to 64-byte boundaries add minor padding bytes per head.
- **Total Overhead**: At $T = 512$, the actual resident KV cache buffer is approximately $2.5\text{--}3.0\text{ MB}$ in FP16.

---

## 15. Computational Complexity & FLOPs Accounting

### 15.1 Decomposed FLOP Count (Forward Pass, Sequence Length $T$, Batch Size $B = 1$)
Following standard convention, a multiply-accumulate operation ($\text{MAC}$) is counted as $2\text{ FLOPs}$ ($1\text{ multiply} + 1\text{ add}$).

1. **Embedding Lookup**: Memory gather operation; **$0\text{ FLOPs}$**.
2. **Q, K, V Linear Projections** (per layer):
   - $3 \times (2 \times T \times d_{\text{model}} \times d_{\text{model}}) = 6 \times T \times 192^2 = 221,184 \cdot T\text{ FLOPs}$.
3. **RoPE Embedding Application** (per layer):
   - Pairwise Givens rotation: $4\text{ FLOPs}$ per element for $Q$ and $K$.
   - $2 \times (4 \times T \times d_{\text{model}}) = 8 \times 192 \times T = 1,536 \cdot T\text{ FLOPs}$.
4. **Attention Score Computation ($Q K^T$)** (per layer):
   - $H$ heads, each multiplying $(T \times d_{\text{head}}) \times (d_{\text{head}} \times T)$:
   - $H \times (2 \times T^2 \times d_{\text{head}}) = 2 \times T^2 \times (H \cdot d_{\text{head}}) = 2 \times 192 \times T^2 = 384 \cdot T^2\text{ FLOPs}$.
   - Scaling + Causal Mask Addition: $2 \times H \times T^2 = 12 \cdot T^2\text{ FLOPs}$.
   - Row-wise Softmax: $\approx 3 \times H \times T^2 = 18 \cdot T^2\text{ FLOPs}$.
   - Total Attention Score & Softmax: $414 \cdot T^2\text{ FLOPs}$.
5. **Attention Context Aggregation ($P V$)** (per layer):
   - $H$ heads, each multiplying $(T \times T) \times (T \times d_{\text{head}})$:
   - $H \times (2 \times T^2 \times d_{\text{head}}) = 2 \times 192 \times T^2 = 384 \cdot T^2\text{ FLOPs}$.
6. **Attention Output Projection ($W_O$)** (per layer):
   - $2 \times T \times d_{\text{model}} \times d_{\text{model}} = 2 \times 192^2 \times T = 73,728 \cdot T\text{ FLOPs}$.
7. **SwiGLU Sub-Layer** (per layer):
   - Gate Linear: $2 \times T \times d_{\text{model}} \times d_{\text{ff}} = 2 \times 192 \times 512 \times T = 196,608 \cdot T\text{ FLOPs}$.
   - Up Linear: $2 \times T \times d_{\text{model}} \times d_{\text{ff}} = 196,608 \cdot T\text{ FLOPs}$.
   - Swish non-linearity & Hadamard product: $\approx 4 \times T \times d_{\text{ff}} = 2,048 \cdot T\text{ FLOPs}$.
   - Down Linear: $2 \times T \times d_{\text{ff}} \times d_{\text{model}} = 2 \times 512 \times 192 \times T = 196,608 \cdot T\text{ FLOPs}$.
   - Total SwiGLU: $591,872 \cdot T\text{ FLOPs}$.
8. **RMSNorm Operations** (per layer):
   - 2 norms per layer: $2 \times (4 \times T \times d_{\text{model}}) = 1,536 \cdot T\text{ FLOPs}$.
9. **Final RMSNorm & Tied LM Head**:
   - Final RMSNorm: $4 \times T \times d_{\text{model}} = 768 \cdot T\text{ FLOPs}$.
   - Tied LM Head Projection: $2 \times T \times d_{\text{model}} \times V = 2 \times 192 \times 4096 \times T = 1,572,864 \cdot T\text{ FLOPs}$.

### 15.2 Consolidated Total FLOP Formulation
Summing over all $L = 6$ layers:
$$\text{Linear Terms (Per Layer)} = 221,184 + 1,536 + 73,728 + 591,872 + 1,536 = 889,856 \cdot T$$
$$\text{Quadratic Terms (Per Layer)} = 414 + 384 = 798 \cdot T^2$$
$$\text{Total Model Linear Terms} = 6 \times 889,856 \cdot T + 768 \cdot T + 1,572,864 \cdot T = 6,912,768 \cdot T$$
$$\text{Total Model Quadratic Terms} = 6 \times 798 \cdot T^2 = 4,788 \cdot T^2$$

$$\mathbf{FLOPs}(T) = 6,912,768 \cdot T + 4,788 \cdot T^2$$

### 15.3 Numerical FLOP Values for Target Sequence Contexts

| Sequence Context ($T$) | Linear FLOPs | Quadratic FLOPs | Total Forward FLOPs | Average FLOPs/Token |
| :---: | :---: | :---: | :---: | :---: |
| **$T = 1$ (Single Token Decode)** | $6.91\text{ MFLOPs}$ | $4.79\text{ KFLOPs}$ | **$6.91\text{ MFLOPs}$** | $6.91\text{ MFLOPs}$ |
| **$T = 128$ tokens** | $884.83\text{ MFLOPs}$ | $78.45\text{ MFLOPs}$ | **$963.28\text{ MFLOPs}$** | $7.53\text{ MFLOPs}$ |
| **$T = 256$ tokens** | $1.770\text{ GFLOPs}$ | $313.79\text{ MFLOPs}$ | **$2.083\text{ GFLOPs}$** | $8.14\text{ MFLOPs}$ |
| **$T = 512$ tokens ($T_{\text{max}}$)**| **$3.539\text{ GFLOPs}$** | **$1.255\text{ GFLOPs}$** | **$4.794\text{ GFLOPs}$** | **$9.36\text{ MFLOPs}$** |

### 15.4 Computational Bottleneck Analysis for CPU Inference
- **SwiGLU Dominance**: SwiGLU accounts for $51.3\%$ of the layer FLOPs ($591,872$ out of $1,153,600$ FLOPs/token/layer). Optimizing dense linear kernels for SwiGLU directly yields the highest throughput gains.
- **LM Head Overhead**: At $1.57\text{ MFLOPs/token}$, the unembedding projection to $V=4096$ accounts for $22.7\%$ of total linear FLOPs.
- **Bounded Quadratic Scaling**: At $T = 512$, quadratic attention accounts for only $26.2\%$ of total compute. The model remains primarily linear and memory-bandwidth bound.

---

## 16. Low-Specification CPU Engineering Analysis

The core mission of ChakrView is to discover whether structured, indigenous intelligence can be achieved under extreme hardware constraints. The following engineering principles govern future execution on low-spec CPUs:

### 16.1 Arithmetic Intensity & Memory Bandwidth Wall
During autoregressive generation ($B=1$), token prediction proceeds one token at a time. The processor must load all $3.44\text{M}$ weights from memory to compute a single token:
$$\text{Memory Traffic per Token (FP32)} \approx 13.77\text{ MB}$$
$$\text{Memory Traffic per Token (INT8)} \approx 3.44\text{ MB}$$
At $6.91\text{ MFLOPs/token}$, the operational intensity is:
$$I = \frac{6.91 \times 10^6\text{ FLOPs}}{3.44 \times 10^6\text{ bytes}} \approx 2.01\text{ FLOPs/byte}$$
Low-spec CPUs are **memory-bandwidth bound**, not compute bound. An older DDR3/DDR4 CPU with $15\text{ GB/s}$ effective bandwidth can theoretically stream weights at:
$$\text{Max Throughput (INT8)} = \frac{15,000\text{ MB/s}}{3.44\text{ MB/token}} \approx 4,360\text{ tokens/second (theoretical peak)}$$
Reducing parameter precision directly increases tokens-per-second linearly with bandwidth savings.

### 16.2 CPU Execution Considerations (Future Options Catalog)
The following techniques are documented as future engineering options and are **NOT implemented in Step 3.1**:
1. **INT8 / INT4 Quantization**: Reduces memory traffic by $2\times\text{--}4\times$, doubling or quadrupling memory-bound decode throughput.
2. **Operator Fusion**: Fusing RMSNorm directly into subsequent GEMM input buffers avoids redundant round-trips to main memory.
3. **Contiguous Memory Layout**: Ensuring weights and KV caches are stored in cache-line aligned, contiguous memory blocks prevents CPU TLB misses and cache thrashing.
4. **Cache-Aware Tiled GEMM**: Tiling matrix multiplications to match L1/L2 cache capacities ($32\text{ KB}$ L1, $256\text{--}512\text{ KB}$ L2).
5. **Incremental KV-Cache Updates**: Updating KV-cache circular buffers in-place without copying historical memory.

---

## 17. 28-nm & Legacy Hardware Constraint Analysis

### 17.1 Measurable Portability Targets
To guarantee execution on older microarchitectures (such as 28-nanometer fabrication class processors, legacy Intel Core/AMD Athlon CPUs, and low-power ARM Cortex-A53 / A7 cores), Chakr-Micro enforces strict architectural portability criteria:

| Design Dimension | Architectural Trait | Benefit on Legacy / 28-nm Hardware |
| :--- | :--- | :--- |
| **Instruction Dependencies** | Pure Standard Vector Math | Does not require AVX-512, AMX, or exotic matrix accelerators; runs on baseline SSE2/AVX or ARM NEON. |
| **Static Working Set** | $3.44\text{ MB}$ (INT8) | Fits inside standard RAM without swapping, even on systems with only $512\text{ MB}\text{--}1\text{ GB}$ total RAM. |
| **No GPU Prerequisite** | 100% Native CPU Algebra | Fully executable without discrete GPUs, OpenCL drivers, or CUDA toolkits. |
| **Uniform GEMM Topology** | Identical matrix sizes | Simplifies compiler optimization and unrolling on legacy in-order execution cores. |
| **Bounded Dynamic Memory** | Max KV cache $2.25\text{ MB}$ | Predictable memory ceiling; prevents out-of-memory crashes on embedded micro-controllers. |

### 17.2 Architectural Design vs. Hardware Performance Claims
We maintain an uncompromised scientific distinction:
- **Architectural Fact**: The mathematical structure of Chakr-Micro requires zero specialized hardware instructions and possesses a bounded footprint.
- **Empirical Boundary**: Actual tokens-per-second, power consumption, and thermal characteristics on specific 28-nm silicon must be empirically measured on physical target hardware during future runtime benchmark phases.

---

## 18. Training Dynamics & Numerical Stability Design

All hyperparameters in this section are designated as **PROVISIONAL** and subject to empirical calibration during training pipeline development:

### 18.1 Parameter Initialization Strategy
1. **Embedding Matrix ($E$)**:
   $$\mathbf{e}_v \sim \mathcal{N}\left(0, \frac{1}{\sqrt{d_{\text{model}}}}\right) = \mathcal{N}\left(0, \frac{1}{\sqrt{192}}\right) \approx \mathcal{N}(0, 0.0722)$$
2. **Standard Projections ($W_Q, W_K, W_V, W_{\text{gate}}, W_{\text{up}}$)**:
   $$W \sim \mathcal{N}\left(0, \frac{1}{\sqrt{d_{\text{model}}}}\right) \approx \mathcal{N}(0, 0.0722)$$
3. **Residual Projections ($W_O, W_{\text{down}}$)**:
   To prevent residual variance explosion across $L = 6$ layers, residual projections are scaled by modern deep-network residual factor $1/\sqrt{2L}$:
   $$W_{\text{res}} \sim \mathcal{N}\left(0, \frac{1}{\sqrt{2L \cdot d_{\text{model}}}}\right) = \mathcal{N}\left(0, \frac{1}{\sqrt{12 \times 192}}\right) \approx \mathcal{N}(0, 0.0208)$$
4. **RMSNorm Gains ($\boldsymbol{\gamma}$)**:
   Initialized uniformly to $\boldsymbol{\gamma} = \mathbf{1}_{192}$.

### 18.2 Optimizer & Hyperparameters (Provisional)
- **Primary Optimizer**: AdamW (Decoupled Weight Decay).
  - Learning Rate: $\eta_{\max} = 1.0 \times 10^{-3}$
  - Betas: $\beta_1 = 0.90, \beta_2 = 0.95$
  - Epsilon: $\epsilon_{\text{adam}} = 1.0 \times 10^{-8}$
  - Weight Decay: $\lambda = 0.10$ applied strictly to 2D projection matrices ($W_Q, W_K, W_V, W_O, W_{\text{gate}}, W_{\text{up}}, W_{\text{down}}$). Excluded from 1D normalizer scales ($\boldsymbol{\gamma}$) and embeddings ($E$).
- **Learning Rate Schedule**:
  - Warmup: Linear warmup over first $5\%$ of total training steps.
  - Decay: Cosine annealing down to $\eta_{\min} = 0.10 \times \eta_{\max} = 1.0 \times 10^{-4}$.
- **Gradient Clipping**: Maximum $L_2$ gradient norm $\|\mathbf{g}\|_2 \le 1.0$ to mitigate gradient spikes.
- **Arithmetic Precision**: Master weights, gradients, optimizer moments, and Softmax evaluations maintained in `float32`.

---

## 19. Causal Language Modeling Loss Function

### 19.1 Mathematical Objective
For a token sequence $X = (x_1, x_2, \dots, x_T)$, the training target is the autoregressive sequence shifted by one position:
$$\text{Inputs: } (x_1, x_2, \dots, x_{T-1}), \quad \text{Targets: } Y = (x_2, x_3, \dots, x_T)$$

The objective is the minimization of mean negative log-likelihood over all non-padding tokens:
$$\mathcal{L} = -\frac{1}{N_{\text{valid}}} \sum_{b=1}^B \sum_{t=1}^{T-1} \mathbb{I}(y_{b, t} \ne \text{PAD\_ID}) \log p(y_{b, t} \mid x_{b, \le t})$$
where $\text{PAD\_ID} = 2$, and $N_{\text{valid}} = \sum_{b, t} \mathbb{I}(y_{b, t} \ne \text{PAD\_ID})$.

### 19.2 Numerically Stable Log-Softmax Formulation
Log-probabilities are computed using max-subtraction to prevent floating-point overflow:
$$\mu_{b, t} = \max_{k \in [0, V-1]} z_{b, t, k}$$
$$\log p(y_{b, t} = c) = (z_{b, t, c} - \mu_{b, t}) - \log \sum_{j=0}^{V-1} \exp(z_{b, t, j} - \mu_{b, t})$$

### 19.3 Expected Initial Loss Verification
Under random initialization, the model predicts a uniform distribution over the $V = 4096$ vocabulary:
$$\mathcal{L}_{\text{init}} \approx -\ln\left(\frac{1}{V}\right) = \ln(4096) = 12 \cdot \ln(2) \approx 8.317766$$
An initial cross-entropy loss significantly differing from $8.32 \pm 0.30$ indicates an initialization or scaling defect.

---

## 20. Synthetic Diagnostic Learnability Benchmark Protocol

Before training on natural language corpora, the neural core must be tested against a minimal synthetic learnability test to verify gradient propagation and capacity:

### 20.1 Diagnostic Tasks
1. **Deterministic Copy Task**: Model must memorize and reproduce a fixed token sequence:
   $$X = [s_1, s_2, \dots, s_k, \langle\text{BOS}\rangle, s_1, s_2, \dots, s_k]$$
2. **Associative Key-Value Recall**: Model must retrieve value $V_i$ when queried with key $K_i$:
   $$X = [\langle\text{BOS}\rangle, K_1, V_1, K_2, V_2, \dots, K_N, V_N, \langle\text{SEP}\rangle, K_r \to V_r]$$
3. **Sequential Pattern Counting**: Model must predict deterministic increment sequences ($k \to k+1 \pmod M$).

### 20.2 Measurable Pass Criteria
- **Batch Size**: $N = 32$ synthetic sequences of length $T = 32$.
- **Convergence Target**: Cross-entropy loss $\mathcal{L} < 0.05$ within $150$ optimization steps using AdamW ($\eta = 10^{-3}$).
- **Target Accuracy**: $\ge 99.0\%$ next-token accuracy on the memorized validation split.
- **Scientific Qualification**: This test is strictly a diagnostic validation of gradient health and memory capacity, not proof of general reasoning or language proficiency.

---

## 21. Formal Architectural Invariants Catalog

The following hard invariants must hold true across all implementations and automated tests:

1. **Vocabulary Bounds Invariant**: Every input token ID must strictly satisfy $x \in [0, 4095]$. Any ID outside this range raises a runtime bounds error.
2. **Residual Dimension Invariant**: The residual stream dimension is strictly $\dim(h_l) = 192$ across all layers $l \in [0, L]$.
3. **Exact Head Factorization Invariant**: $d_{\text{model}} \pmod H \equiv 0$ and $d_{\text{head}} \times H \equiv d_{\text{model}}$ ($32 \times 6 \equiv 192$).
4. **Causal Information Barrier Invariant**: $\frac{\partial z_i}{\partial x_j} \equiv 0.0$ for all $j > i$ (strictly zero future-token information leakage).
5. **Exact Weight Tying Invariant**: $\text{shape}(W_{\text{head}}) = \text{shape}(E)^T = [192, 4096]$, sharing identical underlying storage.
6. **Exact Parameter Count Invariant**: Full model parameter count must equal analytically calculated **$3,443,136$**.
7. **Logits Shape Invariant**: Forward pass on input `(B, T)` must emit logits of shape strictly `(B, T, 4096)`.
8. **RMSNorm Scale Invariance**: $\text{RMSNorm}(\alpha \mathbf{u}) = \text{RMSNorm}(\mathbf{u})$ for any scalar $\alpha > 0$.
9. **Numerical Determinism Invariant**: Repeated forward passes on identical inputs under fixed weights produce bit-exact identical outputs.
10. **Zero NaN / Inf Invariant**: No NaN or Inf values produced in activations or logits under inputs drawn from standard token distributions.

---

## 22. Quantitative Evaluation of Design Alternatives

Before freezing Chakr-Micro v0.1, three candidate architectures were evaluated:

| Metric / Dimension | Candidate A (Deeper / Narrower) | Candidate B (Chakr-Micro — Selected) | Candidate C (Shallower / Wider) |
| :--- | :---: | :---: | :---: |
| **Number of Layers ($L$)** | $8$ | **$6$** | $4$ |
| **Hidden Dimension ($d_{\text{model}}$)** | $160$ | **$192$** | $256$ |
| **Attention Heads ($H$)** | $5$ | **$6$** | $8$ |
| **Head Dimension ($d_{\text{head}}$)** | $32$ | **$32$** | $32$ |
| **SwiGLU Dimension ($d_{\text{ff}}$)**| $416$ ($\approx 2.6d$) | **$512$ ($= \frac{8}{3}d$)** | $672$ ($\approx 2.625d$) |
| **Total Parameters** | $3,074,720$ (~3.07M) | **$3,443,136$ (~3.44M)** | $4,163,840$ (~4.16M) |
| **Static FP16 Memory** | $5.86\text{ MB}$ | **$6.57\text{ MB}$** | $7.94\text{ MB}$ |
| **Static INT8 Memory** | $2.93\text{ MB}$ | **$3.28\text{ MB}$** | $3.97\text{ MB}$ |
| **KV-Cache ($T=512$, FP16)** | $2.50\text{ MB}$ | **$2.25\text{ MB}$** | $2.00\text{ MB}$ |
| **Forward FLOPs ($T=512$)** | $4.558\text{ GFLOPs}$ | **$4.794\text{ GFLOPs}$** | $5.392\text{ GFLOPs}$ |
| **FLOPs/Token Average** | $8.90\text{ MFLOPs}$ | **$9.36\text{ MFLOPs}$** | $10.53\text{ MFLOPs}$ |

### Technical Selection Rationale for Candidate B
1. **Symmetric Head Factorization**: Candidate A requires $H = 5$ heads. An odd head count of $5$ prevents even division into SIMD pairs, complicates GQA/MQA extensions, and creates awkward parallel scheduling. Candidate B ($H = 6$) divides cleanly into 1, 2, 3, or 6 groups.
2. **Exact SwiGLU Integer Alignment**: In Candidate B, $d_{\text{ff}} = \frac{8}{3} \times 192 = 512$ is an exact integer and power of 2 ($2^9$), aligning with CPU cache lines. Candidate A and C require non-integer rounding.
3. **Hierarchical Representational Depth**: Candidate C ($L = 4$) is overly shallow, restricting syntactic and semantic hierarchy. Candidate B ($L = 6$) provides balanced representational depth without vanishing gradient risks.

---

## 23. Anti-Overengineering Boundary Statement

Chakr-Micro v0.1 represents the **foundational neural core** of ChakrView. To ensure focused execution, the following advanced capabilities are **EXPLICITLY EXCLUDED** from Step 3.1 and v0.1:

- **Mixture of Experts (MoE)**: Single dense feed-forward network only.
- **Recurrent Memory & External Retrieval**: No external vector stores or RAG pipelines.
- **Multimodal Inputs**: Strictly discrete text/byte token processing.
- **Autonomous Agents & Tool Schemas**: Decoupled at higher system levels.
- **Sparse / Speculative Attention**: Pure dense causal self-attention.
- **Reinforcement Learning**: Autoregressive cross-entropy next-token prediction only.

The priority is to establish a mathematically proven, stable, and verifiable indigenous neural core.

---

## 24. File Manifest & Implementation Prerequisites

### Created/Updated Documents
1. `docs/STEP_03_1_NEURAL_CORE_SPEC.md` (This complete specification)
2. `docs/NEURAL_CORE_DECISIONS.md` (Architecture Decision Records ADR 11–ADR 15)
3. `docs/PROJECT_STATUS.md` (Updated status milestone)

### Mandatory Implementation Prerequisite
> **MANDATORY RULE**: Step 3.1 is strictly an architectural and mathematical specification.
> No Python model code shall be written, no neural network layers implemented, no datasets downloaded, and no training loops initiated until Step 3.1 has been verified and committed.

---
*End of Specification — ChakrView Research Team*
