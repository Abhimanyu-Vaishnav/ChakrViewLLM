# ChakrView Neural Core Specification (Chakr-Core v0.1)

**Document Version**: 0.1.0  
**Project Phase**: Step 1 — Foundational Neural Core Specification  
**Status**: APPROVED RESEARCH SPECIFICATION  
**Author**: ChakrView Core Research & Architecture Team  

---

## 1. Executive Summary & Core Philosophy

ChakrView is an indigenous AI research initiative committed to building an intelligent, hardware-efficient neural cognitive system from first principles without relying on third-party pretrained weights (such as Llama, Mistral, Qwen, Gemma, or GPT).

The objective of **Step 1** is to establish the complete mathematical, structural, and architectural specification for **Chakr-Core v0.1**: the foundational neural language core.

> **MANDATORY ENGINEERING RULE**: No implementation should begin until the tokenizer specification and the final v0.1 tensor contracts have been reviewed.

### Design Principles:
1. **Indigenous & Foundational**: Designed and implemented from scratch without borrowing black-box code or pretrained model checkpoints.
2. **Hardware Realism & Efficiency**: Architected specifically to enable efficient CPU inference, low RAM consumption, and seamless deployment on older microprocessors, edge systems, and low-resource devices.
3. **Mathematical Determinism & Stability**: Prioritizing numerical stability, clean gradient flow, and predictable optimization dynamics.
4. **Strict Modular Decoupling**: Enforcing uncompromising boundaries between the neural core, tokenization, training infrastructure, inference runtime, memory, and cognitive skills.
5. **Future-Proof Extensibility**: Laying the architectural hooks for Grouped-Query Attention (GQA), weight tying, quantization, and KV-caching without introducing premature complexity.

---

## 2. Architectural Paradigm Evaluation & Selection

To identify the optimal foundation for Chakr-Core v0.1, four fundamental autoregressive sequence modeling paradigms were analyzed:

| Evaluation Dimension | 1. Autoregressive Transformer Decoder | 2. Modern Recurrent Architecture (Linear RNN / RWKV) | 3. State-Space Model (SSM / Selective SSM) | 4. Hybrid Architecture (Attention + SSM/RNN) |
| :--- | :--- | :--- | :--- | :--- |
| **CPU Inference ($B=1$)** | Exceptional for $T \le 1024$. Standard GEMM kernels saturate CPU SIMD (AVX2/AVX-512). | Excellent $O(1)$ token step; low CPU memory bandwidth. | Very fast per-token generation; bounded state size. | Moderate; depends on layer scheduling and caching. |
| **RAM Footprint** | Extremely low for small context ($< 10\text{ MB}$ KV-cache for $T \le 1024$). | Constant $O(1)$ memory; smallest working footprint. | Constant $O(1)$ memory; fixed state vector. | Moderate; requires both KV-cache and recurrent states. |
| **Training Complexity** | **Lowest**. Fully parallel across sequence; highly stable causal masking; well-behaved gradients. | High. Requires parallel associative scans to avoid slow sequential loops. | Very High. Requires custom chunked associative scans and hardware-aware memory tiling. | Highest. Unbalanced gradient propagation between disparate block types. |
| **Implementation Complexity** | **Minimal & Clean**. Pure standard tensor algebra (MatMul, Softmax, RMSNorm). Zero custom C/CUDA kernels required. | High. Numerical stability of exponential decay requires custom backward passes. | High. Discretization dynamics ($\Delta, A, B, C$) require intricate parameterizations. | Extremely high. Over-engineered for an indigenous v0.1 prototype. |
| **Long-Context Potential** | $O(T^2)$ compute/memory without sparse/chunked attention, but trivial for $T \le 2048$. | $O(T)$ compute, $O(1)$ state; but prone to context compression loss. | Excellent linear scaling; retains long-range memories efficiently. | Combines associative recall with linear state efficiency. |
| **Edge Deployment & Portability**| **Universal**. Supported natively by every compiler, ONNX, GGML, WebAssembly, and bare-metal C. | Limited out-of-the-box runtime and compiler support; non-standard graph ops. | Poor runtime portability; non-standard ops lack CPU/NPU native kernels. | Complex export pathways; fragmented runtime support. |

### Architectural Recommendation: Chakr-Core Causal Decoder v0.1
**Recommendation**: **Pre-LayerNorm / Pre-RMSNorm Causal Transformer Decoder with Rotary Position Embeddings (RoPE) and SwiGLU Feed-Forward Networks.**

#### Justification for v0.1:
1. **Mathematical Robustness**: The causal transformer provides an exceptionally well-conditioned optimization landscape, ensuring that training stability issues do not obscure algorithmic bugs during initial development.
2. **Zero Proprietary/Custom Kernel Dependencies**: Can be implemented in standard vectorized linear algebra libraries (NumPy, PyTorch CPU, or native C/BLAS) without requiring CUDA toolkits (`nvcc`), Triton, or specialized scan kernels.
3. **Universal Portability**: Enables seamless compilation to CPU-only targets, low-power edge chips, and micro-servers.
4. **Clean Evolution Path**: The attention block can smoothly transition into Grouped-Query Attention (GQA) and linear attention in later steps with zero modifications to the external interface.

---

## 3. Complete Mathematical Data Flow

The forward pass of Chakr-Core v0.1 represents a strict, deterministic mapping from discrete token indices to unnormalized next-token logits:

$$\mathbf{x} \in \mathbb{N}^{B \times T} \longrightarrow \mathbf{z} \in \mathbb{R}^{B \times T \times V}$$

```
Input Token IDs [B, T]
       │
       ▼
Token Embedding Matrix (W_E) ────────────┐ (Optional Weight Tying)
       │                                  │
       ▼                                  │
Hidden State h_0 [B, T, d]                │
       │                                  │
 ┌─────┴────────────────────────┐         │
 │ Layer l = 1 ... L            │         │
 │                              │         │
 │  h_{l-1}                     │         │
 │    │                         │         │
 │    ├──► RMSNorm_1 ──► Q, K, V│         │
 │    │                    │    │         │
 │    │                   RoPE  │         │
 │    │                    │    │         │
 │    │             Causal MHA  │         │
 │    │                    │    │         │
 │    │                 W_O     │         │
 │    │                    │    │         │
 │    ▼                    ▼    │         │
 │   (+) ◄─────────────────┘    │         │
 │    │ (Residual 1)            │         │
 │    │                         │         │
 │    ├──► RMSNorm_2            │         │
 │    │         │               │         │
 │    │      SwiGLU FFN         │         │
 │    │         │               │         │
 │    ▼         ▼               │         │
 │   (+) ◄──────┘               │         │
 │    │ (Residual 2)            │         │
 │    ▼                         │         │
 │   h_l                        │         │
 └─────┬────────────────────────┘         │
       │                                  │
       ▼                                  │
 Final RMSNorm                            │
       │                                  │
       ▼                                  │
 Linear Output Projection (W_head) ◄──────┘
       │
       ▼
 Output Logits [B, T, V]
```

### Detailed Sub-Stage Formulations

#### Stage 1: Token Embedding
Given an input sequence of token IDs $X = (x_{b, t}) \in \mathbb{N}^{B \times T}$, where $x_{b, t} \in \{0, 1, \dots, V-1\}$:
$$h_0 = \text{Embedding}(X) = W_E[X] \in \mathbb{R}^{B \times T \times d}$$
where $W_E \in \mathbb{R}^{V \times d}$ is the learned token embedding matrix.

#### Stage 2: Rotary Position Embeddings (RoPE)
Instead of static additive positional embeddings, Chakr-Core v0.1 employs Rotary Position Embeddings applied directly to the query and key representations within each attention head.
For head dimension $d_k = d / H$, rotary frequencies are defined as:
$$\theta_i = b^{-2i / d_k}, \quad i \in \left\{0, 1, \dots, \frac{d_k}{2} - 1\right\}$$
where base frequency $b = 10000.0$.
For a token at sequence position $m \in \{0, \dots, T-1\}$, and a 2D component vector $\begin{pmatrix} v_{2i} \\ v_{2i+1} \end{pmatrix}$:
$$\mathcal{R}_m^{(i)} \begin{pmatrix} v_{2i} \\ v_{2i+1} \end{pmatrix} = \begin{pmatrix} \cos(m \theta_i) & -\sin(m \theta_i) \\ \sin(m \theta_i) & \cos(m \theta_i) \end{pmatrix} \begin{pmatrix} v_{2i} \\ v_{2i+1} \end{pmatrix}$$
This guarantees that the inner product depends exclusively on the relative token displacement $(m - n)$:
$$\langle \text{RoPE}(q_m), \text{RoPE}(k_n) \rangle = g(q, k, m - n)$$

#### Stage 3: Root Mean Square Normalization (RMSNorm)
Pre-normalization is applied before attention and feed-forward sub-layers. RMSNorm provides equivalent training stability to LayerNorm while eliminating mean-centering, saving $7\%$ to $12\%$ of normalization compute:
$$\text{RMSNorm}(u) = \frac{u}{\text{RMS}(u) + \epsilon} \odot \gamma$$
where:
$$\text{RMS}(u) = \sqrt{\frac{1}{d} \sum_{j=1}^d u_j^2}, \quad \gamma \in \mathbb{R}^d, \quad \epsilon = 10^{-6}$$

#### Stage 4: Causal Self-Attention (Multi-Head Attention - MHA)
> **IMPORTANT ARCHITECTURAL CLARIFICATION**: Chakr-Micro v0.1 will implement **SIMPLE MULTI-HEAD ATTENTION (MHA)** where $H_{kv} = H$. Grouped-Query Attention (GQA) is strictly an architectural extension point for future model iterations and will **NOT** be implemented in v0.1.

1. Normalized input: $\hat{h} = \text{RMSNorm}_1(h_{l-1}) \in \mathbb{R}^{B \times T \times d}$.
2. Linear projections (for MHA, $H_{kv} = H$, meaning key and value projections match query dimension $H \cdot d_k = d$):
   $$Q = \hat{h} W_Q \in \mathbb{R}^{B \times T \times d}$$
   $$K = \hat{h} W_K \in \mathbb{R}^{B \times T \times d}$$
   $$V = \hat{h} W_V \in \mathbb{R}^{B \times T \times d}$$
3. Reshape into head tensors:
   $$Q, K, V \in \mathbb{R}^{B \times H \times T \times d_k}$$
4. Position encoding:
   $$\tilde{Q} = \text{RoPE}(Q), \quad \tilde{K} = \text{RoPE}(K)$$
5. Scaled Causal Dot-Product Attention:
   $$S = \frac{\tilde{Q} \tilde{K}^T}{\sqrt{d_k}} + M_{\text{causal}} \in \mathbb{R}^{B \times H \times T \times T}$$
   where the causal mask $M_{\text{causal}}$ is:
   $$M_{\text{causal}}(i, j) = \begin{cases} 0 & \text{if } i \ge j \\ -\infty & \text{if } i < j \end{cases}$$
6. Attention probabilities and context aggregation:
   $$P = \text{Softmax}(S, \text{dim}=-1) \in \mathbb{R}^{B \times H \times T \times T}$$
   $$O_{\text{heads}} = P V \in \mathbb{R}^{B \times H \times T \times d_k}$$
7. Out projection and first residual merge:
   $$h_{\text{attn}} = \text{Concat}(O_{\text{heads}}) W_O \in \mathbb{R}^{B \times T \times d}$$
   $$h_{\text{mid}} = h_{l-1} + h_{\text{attn}}$$

#### Stage 5: SwiGLU Feed-Forward Network (FFN)
The feed-forward sub-layer utilizes the SwiGLU (Swish Gated Linear Unit) activation, which demonstrates superior inductive bias and parameter efficiency compared to standard ReLU/GELU MLPs:
$$\hat{h}_{\text{mid}} = \text{RMSNorm}_2(h_{\text{mid}}) \in \mathbb{R}^{B \times T \times d}$$
$$\text{SwiGLU}(\hat{h}_{\text{mid}}) = \left( \text{Swish}(\hat{h}_{\text{mid}} W_{\text{gate}}) \odot (\hat{h}_{\text{mid}} W_{\text{up}}) \right) W_{\text{down}}$$
where:
- $\text{Swish}(z) = z \cdot \sigma(z) = \frac{z}{1 + e^{-z}}$
- $W_{\text{gate}} \in \mathbb{R}^{d \times d_{ff}}$
- $W_{\text{up}} \in \mathbb{R}^{d \times d_{ff}}$
- $W_{\text{down}} \in \mathbb{R}^{d_{ff} \times d}$
- $\odot$ denotes Hadamard (elementwise) multiplication.
- Second residual merge:
  $$h_l = h_{\text{mid}} + \text{SwiGLU}(\hat{h}_{\text{mid}})$$

#### Stage 6: Final Normalization & Logits Projection
After passing through all $L$ stacked blocks:
$$h_{\text{final}} = \text{RMSNorm}_{\text{final}}(h_L) \in \mathbb{R}^{B \times T \times d}$$
$$\mathbf{z} = h_{\text{final}} W_{\text{head}} \in \mathbb{R}^{B \times T \times V}$$
where $W_{\text{head}} \in \mathbb{R}^{d \times V}$. When **Weight Tying** is enabled, $W_{\text{head}} = W_E^T$.

---

## 4. Tensor Shape Catalog

Every tensor across all computational stages is strictly typed and dimensionally bounded:

| Tensor Description | Variable Name | Dimensional Shape | Data Type (Dev / Train) |
| :--- | :--- | :--- | :--- |
| Batch Token Identifiers | $X$ | `(B, T)` | `int64` |
| Token Embeddings | $h_0$ | `(B, T, d)` | `float32` |
| Pre-Attention Normalized State | $\hat{h}$ | `(B, T, d)` | `float32` |
| Query Projection Matrix | $Q$ | `(B, T, H, d_k)` | `float32` |
| Key Projection Matrix | $K$ | `(B, T, H_{kv}, d_k)` | `float32` |
| Value Projection Matrix | $V$ | `(B, T, H_{kv}, d_k)` | `float32` |
| RoPE Sin/Cos Table | $\sin, \cos$ | `(1, T, 1, d_k)` | `float32` |
| Rotated Query / Key | $\tilde{Q}, \tilde{K}$ | `(B, H, T, d_k)` / `(B, H_{kv}, T, d_k)` | `float32` |
| Causal Attention Mask | $M_{\text{causal}}$ | `(1, 1, T, T)` | `float32` |
| Attention Raw Logits | $S$ | `(B, H, T, T)` | `float32` |
| Softmax Attention Weights | $P$ | `(B, H, T, T)` | `float32` |
| Aggregated Head Context | $O_{\text{heads}}$ | `(B, T, d)` | `float32` |
| Attention Sublayer Output | $h_{\text{attn}}$ | `(B, T, d)` | `float32` |
| Mid-Block Residual State | $h_{\text{mid}}$ | `(B, T, d)` | `float32` |
| Pre-FFN Normalized State | $\hat{h}_{\text{mid}}$ | `(B, T, d)` | `float32` |
| SwiGLU Gate Projection | $G$ | `(B, T, d_{ff})` | `float32` |
| SwiGLU Up Projection | $U$ | `(B, T, d_{ff})` | `float32` |
| SwiGLU Activated Hidden | $\text{Swish}(G) \odot U$ | `(B, T, d_{ff})` | `float32` |
| FFN Sublayer Output | $h_{\text{ffn}}$ | `(B, T, d)` | `float32` |
| Layer Output State | $h_l$ | `(B, T, d)` | `float32` |
| Final Normalized Representation | $h_{\text{final}}$ | `(B, T, d)` | `float32` |
| Unnormalized Next-Token Logits | $\mathbf{z}$ | `(B, T, V)` | `float32` |
| Target Token Identifiers | $Y$ | `(B, T)` | `int64` |
| Scalar Cross-Entropy Loss | $\mathcal{L}$ | `()` (scalar) | `float32` |

---

## 5. Trainable Parameters Specification

Linear layers in Chakr-Core v0.1 deliberately omit additive bias vectors ($b = 0$). This reduces memory footprint, eliminates dead-neuron drift, and improves inference throughput on CPUs:

| Component | Parameter Symbol | Parameter Shape | Count (Standard / MHA) | Count Formula |
| :--- | :--- | :--- | :--- | :--- |
| **Token Embeddings** | $W_E$ | `(V, d)` | $V \cdot d$ | $V \cdot d$ |
| **Layer RMSNorm 1** | $\gamma_{1, l}$ | `(d,)` | $d$ | $d$ |
| **Query Projection** | $W_{Q, l}$ | `(d, H * d_k)` | $d^2$ | $d \cdot (H \cdot d_k)$ |
| **Key Projection** | $W_{K, l}$ | `(d, H_kv * d_k)`| $d^2$ | $d \cdot (H_{kv} \cdot d_k)$ |
| **Value Projection**| $W_{V, l}$ | `(d, H_kv * d_k)`| $d^2$ | $d \cdot (H_{kv} \cdot d_k)$ |
| **Output Projection**| $W_{O, l}$ | `(H * d_k, d)` | $d^2$ | $(H \cdot d_k) \cdot d$ |
| **Layer RMSNorm 2** | $\gamma_{2, l}$ | `(d,)` | $d$ | $d$ |
| **FFN Gate Linear** | $W_{\text{gate}, l}$ | `(d, d_ff)` | $d \cdot d_{ff}$ | $d \cdot d_{ff}$ |
| **FFN Up Linear** | $W_{\text{up}, l}$ | `(d, d_ff)` | $d \cdot d_{ff}$ | $d \cdot d_{ff}$ |
| **FFN Down Linear** | $W_{\text{down}, l}$ | `(d_ff, d)` | $d_{ff} \cdot d$ | $d_{ff} \cdot d$ |
| **Final RMSNorm** | $\gamma_{\text{final}}$ | `(d,)` | $d$ | $d$ |
| **Unembedding Head** | $W_{\text{head}}$ | `(d, V)` | $0$ (Tied) or $d \cdot V$ | Tied to $W_E^T$ |

### Total Parameter Formulation:
$$\mathbf{P}_{\text{total}} = V \cdot d + L \cdot \left[ 2d + d \cdot d_k (H + 2 H_{kv}) + d^2 + 3 d \cdot d_{ff} \right] + d + \mathbf{P}_{\text{head}}$$
Under weight tying ($\mathbf{P}_{\text{head}} = 0$) and standard MHA ($H_{kv} = H, H \cdot d_k = d$):
$$\mathbf{P}_{\text{tied}} = V \cdot d + L \cdot \left[ 4 d^2 + 3 d \cdot d_{ff} + 2d \right] + d$$

---

## 6. Loss Function & Optimization Formulation

### Autoregressive Next-Token Cross-Entropy Loss
For a sequence length $T$, the targets $Y$ are the input sequence shifted to the left by one position:
$$X = (t_1, t_2, \dots, t_{T-1}), \quad Y = (t_2, t_3, \dots, t_T)$$
The objective is the minimization of mean negative log-likelihood over all non-padding tokens:
$$\mathcal{L}(X, Y) = -\frac{1}{N_{\text{valid}}} \sum_{b=1}^B \sum_{t=1}^{T-1} \mathbb{I}(y_{b, t} \ne \text{PAD}) \log p(y_{b, t} \mid x_{b, \le t})$$
where the predictive distribution is computed via numerically stable Log-Softmax:
$$\log p(y_{b, t} = c) = z_{b, t, c} - \max_k(z_{b, t, k}) - \log \sum_{j=1}^V \exp\left(z_{b, t, j} - \max_k(z_{b, t, k})\right)$$

### Optimizer Selection & Specification
1. **Primary Recommended Optimizer: AdamW (Decoupled Weight Decay)**
   - Hyperparameters: $\beta_1 = 0.90, \beta_2 = 0.95, \epsilon = 10^{-8}, \text{weight\_decay} = 0.10$.
   - Weight decay is applied strictly to 2D projection weights ($W_Q, W_K, W_V, W_O, W_{\text{gate}}, W_{\text{up}}, W_{\text{down}}$) and **excluded** from 1D normalizer scales ($\gamma$) and embeddings ($W_E$).
   - Learning Rate Schedule: Linear Warmup (first 5% of training steps) followed by Cosine Annealing decay down to $10\%$ of maximum learning rate $\eta_{\min} = 0.10 \cdot \eta_{\max}$.
   - Gradient Clipping: Maximum gradient norm $\|\mathbf{g}\|_2 \le 1.0$ to prevent optimization divergence.
2. **Alternative Edge Optimizer Candidate: Lion (EvoLved Sign Momentum)**
   - Tracks only first-moment momentum ($m_t$), reducing optimizer state RAM by $50\%$ compared to AdamW.
   - Designated as the secondary optimizer for memory-constrained training experiments.

---

## 7. Model Configurations & Hardware Resource Budgets

To validate ChakrView across varied computational constraints, three discrete model configurations are defined:

> **Important Clarification on Memory Estimates**:
> Model-weight memory estimates (FP16, INT8, INT4) listed below represent **static weight storage only** and strictly **exclude**:
> - Dynamic activations during training and inference
> - Runtime memory allocator overhead and fragmentation
> - Temporary scratch tensors and intermediate buffers
> - Tokenizer memory and vocabulary lookup tables
> - Key-Value (KV) cache memory (unless explicitly stated in the dedicated row below)
> - Framework and runtime engine overhead.

| Metric / Dimension | A. Chakr-Micro (Prototype) | B. Chakr-Small (CPU Core) | C. Chakr-Research (Workstation) |
| :--- | :--- | :--- | :--- |
| **Primary Target Hardware** | Low-spec CPU, Micro-VMs, Edge | Quad-Core CPU, 8GB-16GB RAM | Workstation CPU, 6GB VRAM GPU |
| **Target Vocabulary Size ($V$)** | $4,096$ tokens *(provisional)* | $8,192$ tokens *(provisional)* | $16,384$ tokens *(provisional)* |
| **Hidden Dimension ($d$)** | $192$ | $384$ | $640$ |
| **Number of Layers ($L$)** | $6$ | $8$ | $12$ |
| **Attention Query Heads ($H$)**| $6$ | $6$ | $10$ |
| **KV Heads ($H_{kv}$)** | $6$ (Simple MHA) | $6$ (Simple MHA) | $2$ (Future GQA 5:1 extension) |
| **Head Dimension ($d_k$)** | $32$ | $64$ | $64$ |
| **Feed-Forward Dimension ($d_{ff}$)**| $512$ ($\approx 2.67d$) | $1,024$ ($\approx 2.67d$) | $1,728$ ($\approx 2.70d$) |
| **Context Length ($T$)** | $512$ tokens | $1,024$ tokens | $2,048$ tokens |
| **Weight Tying** | Enabled ($W_{\text{head}} = W_E^T$) | Enabled ($W_{\text{head}} = W_E^T$) | Enabled ($W_{\text{head}} = W_E^T$) |
| **Total Parameter Count** | **$3,443,136$ (~3.44M)** | **$17,308,032$ (~17.31M)** | **$62,111,360$ (~62.11M)** |
| **FP16 Static Weight Memory** | **$6.89\text{ MB}$** | **$34.62\text{ MB}$** | **$124.22\text{ MB}$** |
| **INT8 Static Weight Memory** | **$3.44\text{ MB}$** | **$17.31\text{ MB}$** | **$62.11\text{ MB}$** |
| **INT4 Static Weight Memory** | **$1.72\text{ MB}$** | **$8.65\text{ MB}$** | **$31.06\text{ MB}$** |
| **KV-Cache Size ($B=1$, Max $T$)** | **$2.36\text{ MB}$** | **$12.58\text{ MB}$** | **$12.58\text{ MB}$** |
| **Estimated Peak CPU Train RAM**| $\approx 180\text{ MB}$ | $\approx 850\text{ MB}$ | $\approx 3.2\text{ GB}$ |

*\*Note on Vocabulary Size: The vocabulary size of 4096 is provisional until tokenizer research is completed.*

### First Trainable Prototype Recommendation: **Chakr-Micro (~3.44M parameters)**
- **Why**: Allows instant feedback cycles. Forward, backward, and optimization passes execute in milliseconds on a standard CPU core. Overfitting validation on synthetic tokens completes in less than 60 seconds. Guarantees that architectural or numerical bugs are diagnosed instantly without waiting on hardware bottlenecks.
- **Attention Mode**: Implements **Simple Multi-Head Attention (MHA)** ($H_{kv} = H = 6$). GQA is an architectural extension point only and will not be implemented in v0.1.
- **Cache-Locality Engineering Hypothesis**: The quantized weight footprint is small enough to make cache-local execution plausible on some CPUs; this must be verified by benchmark.

---

## 8. Architectural Extensibility Hooks (Design for Future Steps)

While v0.1 remains strictly minimal, every module interface is architected to support future optimization techniques without breaking backwards compatibility:

1. **Weight Tying Abstraction**:
   - The output projection interface accepts a reference to the input embedding matrix. A boolean configuration flag `tie_word_embeddings: bool` governs whether an independent linear layer or the transposed embedding matrix is invoked during the projection step.
2. **Grouped-Query Attention (GQA) & Multi-Query Attention (MQA) Extension Point**:
   - GQA is strictly an architectural extension point for future versions and will **NOT** be implemented in v0.1 (Chakr-Micro v0.1 implements simple MHA).
   - In future phases, the key/value projection dimension can be parametrized by `num_key_value_heads` ($H_{kv}$) independently from `num_attention_heads` ($H$). Expanding keys and values via a broadcast repetition operation will allow GQA ($H_{kv} < H$) or MQA ($H_{kv} = 1$) without altering the outer block contracts.
   - A single vector broadcast `repeat_kv()` operation expands keys and values to match query heads during attention calculation. Setting $H_{kv} = H$ yields MHA; $H_{kv} < H$ yields GQA; $H_{kv} = 1$ yields MQA.
3. **KV-Cache Interface Compatibility**:
   - The attention layer forward pass signature explicitly specifies:
     `forward(x, past_key_value=None, use_cache=False) -> (output, present_key_value)`
   - In v0.1, `use_cache=False` is the default. The interface is already shaped to receive cached key-value tensors in inference step execution.
4. **Quantization Hooks**:
   - Every linear operation is constructed as a decoupled affine transformation without fused biases or non-standard activations. This ensures direct compatibility with standard INT8/INT4 weight-only and dynamic activation quantization algorithms.
5. **Dynamic Depth / Early Exit Hooks**:
   - The residual state dimension $d$ is invariant across all layers $l \in \{1, \dots, L\}$. Intermediate hidden states can be directly routed to the final normalizer and tied unembedding head to evaluate early-exit confidence thresholds.
6. **Sparse Computation & MoE Preparation**:
   - The SwiGLU FFN is encapsulated in a dedicated block module. In future iterations, this can be seamlessly substituted with a Mixture-of-Experts (MoE) router directing tokens to $N$ parallel experts without altering the residual structure.

---

## 9. Architectural Boundaries & System Decoupling

To prevent architectural debt, ChakrView enforces impenetrable component boundaries:

```
┌────────────────────────────────────────────────────────┐
│                   ChakrView System                     │
├─────────────────┬──────────────────┬───────────────────┤
│   tokenization/ │      brain/      │     training/     │
│                 │                  │                   │
│ • Text -> IDs   │ • Embeddings     │ • Optimizer state │
│ • IDs -> Text   │ • RoPE           │ • Loss function   │
│ • Vocab mapping │ • ChakrBlock     │ • Backprop engine │
│ • No tensor ops │ • RMSNorm        │ • Learning rate   │
│ • No model refs │ • Pure Tensor In │ • Gradient clip   │
│                 │   -> Tensor Out  │ • Checkpoint I/O  │
├─────────────────┼──────────────────┼───────────────────┤
│   inference/    │     memory/      │   skills/tools/   │
│                 │                  │                   │
│ • Autoregressive│ • Working memory │ • Tool definition │
│   generation    │ • Episodic store │ • Environment I/O │
│ • KV-Cache state│ • External state │ • Skill execution │
│ • Temp / Top-P  │ • Decoupled from │ • Decoupled from  │
│   sampling      │   neural weights │   neural weights  │
└─────────────────┴──────────────────┴───────────────────┘
```

### Strict Architectural Boundaries:
- `brain/` **Boundary**: Contains ONLY PyTorch/tensor neural block definitions. Inputs are strictly integer token tensors; outputs are unnormalized float logits. It has **zero awareness** of raw text, strings, datasets, tokenizers, files, training loops, optimizers, or loss calculations.
- `tokenization/` **Boundary**: Handles text parsing, BPE/character encodings, and string operations. Contains **zero neural network weights** and **zero model references**.
- `training/` **Boundary**: Orchestrates datasets, batches, forward passes, cross-entropy loss computation, optimizer updates, and checkpointing. Does not alter neural core layer logic.
- `inference/` **Boundary**: Wraps the neural core for decoding, KV-caching, and token generation. Keeps generation heuristics out of the core neural layers.
- `memory/` **Boundary**: Reserved for future persistent memory, episodic indexing, or external key-value state. Completely external to the foundational language core.
- `skills/` & `tools/` **Boundary**: Manages tool invocation schemas, API calling, and task execution logic. Completely external to the foundational language core.

---

## 10. Explicit Exclusions for Version 0.1

To prevent scope creep and maintain strict research discipline, the following elements are **EXPLICITLY EXCLUDED** from ChakrView v0.1:

1. **Pretrained Weights**: Zero importation of external checkpoints or weights (no Llama, Mistral, Qwen, Gemma, BERT, GPT).
2. **Third-Party Model Architectures**: No direct copying or wrapping of HuggingFace transformers library model classes (`LlamaForCausalLM`, etc.).
3. **Multimodal Inputs**: No vision encoders, speech processors, or cross-attention projections.
4. **Agent & Tool Frameworks**: No LangChain, LlamaIndex, tool-calling agents, or external system integrations.
5. **Memory & Retrieval Modules**: No vector databases, RAG pipelines, or external episodic memories.
6. **Reinforcement Learning**: No RLHF, DPO, PPO, or reward model components.
7. **Proprietary GPU Kernels**: No CUDA-only compiler dependencies (`nvcc`, custom C++ CUDA extensions, Triton-only kernels) that impair universal CPU execution.
8. **Heavy Distributed Frameworks**: No DeepSpeed, Megatron-LM, or FSDP configurations.

---

## 11. Verification & Testing Specifications

### A. Unit Tests (`tests/test_neural_core.py`)
1. **`test_token_embedding_contract`**:
   - Verify output tensor shape is strictly `(B, T, d)`.
   - Verify embedding lookup is deterministic and gradients flow to $W_E$.
2. **`test_rmsnorm_invariance_and_properties`**:
   - Verify scale invariance: $\text{RMSNorm}(\alpha \mathbf{u}) = \text{RMSNorm}(\mathbf{u})$ for $\alpha > 0$.
   - Verify output root-mean-square equals approximately $1.0$.
   - Verify stability on zero vectors (no division by zero; output equals zeros).
3. **`test_rope_relative_displacement`**:
   - Verify that $\langle \text{RoPE}(q, m), \text{RoPE}(k, n) \rangle$ is identical for any equal displacement $(m - n)$ regardless of absolute position offset.
4. **`test_causal_mask_non_leakage`**:
   - Compute forward pass on input sequence $X$.
   - Compute gradient of logit at position $t_i$ with respect to input at position $t_j$ where $j > i$.
   - Assert gradient is **identically 0.0** (strictly zero future information leakage).
5. **`test_swiglu_shape_and_gating`**:
   - Verify output tensor matches input tensor shape `(B, T, d)`.
   - Verify that if gate projection is zero, the output is strictly zero.
6. **`test_single_block_residual_flow`**:
   - Verify that gradients backpropagate through both attention and FFN residual branches without dying.
7. **`test_full_model_logits_shape`**:
   - Assert model output on `(B, T)` is exactly `(B, T, V)`.
8. **`test_weight_tying_gradients`**:
   - When weight tying is enabled, assert that output projection does not possess independent parameters and its backward pass updates $W_E$.
9. **`test_eval_mode_determinism`**:
   - Verify that repeated evaluations on identical input produce bit-exact identical output logits.

### B. Numerical Sanity Tests
1. **Initial Loss Sanity Check**:
   - Under standard Gaussian/orthogonal initialization, initial cross-entropy loss before any gradient step must satisfy:
     $$\mathcal{L}_{\text{init}} \approx -\ln\left(\frac{1}{V}\right) = \ln(V)$$
   - For $V = 4096$, expected $\mathcal{L}_{\text{init}} \approx \ln(4096) \approx 8.317 \pm 0.25$.
   - Any significant deviation ($< 7.0$ or $> 10.0$) indicates initialization scale errors or broken normalization.
2. **Finite Difference Gradient Verification**:
   - For a miniature test core ($d=16, L=1, V=32$), compute analytical backprop gradients $\nabla_{\theta}^{\text{analytic}} \mathcal{L}$.
   - Approximate gradients numerically via central difference:
     $$\nabla_{\theta_i}^{\text{numeric}} \mathcal{L} = \frac{\mathcal{L}(\theta_i + \delta) - \mathcal{L}(\theta_i - \delta)}{2\delta}, \quad \delta = 10^{-5}$$
   - Relative error criterion:
     $$\frac{\|\nabla^{\text{analytic}} - \nabla^{\text{numeric}}\|_2}{\|\nabla^{\text{analytic}}\|_2 + \|\nabla^{\text{numeric}}\|_2} < 10^{-4}$$
3. **Activation Variance Stability**:
   - Track activation variance across all $L$ layers during forward pass.
   - Assert variance does not explode ($> 100$) or collapse ($< 0.01$).

### C. Synthetic Dataset Learning Verification (Diagnostic Learnability Benchmark)
To verify empirical learning capability without downloading external datasets:
1. **Synthetic Task**: *Associative Pattern Memorization & Sequence Inversion*
   - Construct $N = 32$ deterministic synthetic sequences of length $T = 32$.
   - Format: `[START] K_1 V_1 K_2 V_2 ... [SEP] K_1 -> V_1 [END]`
2. **Initial Benchmark Target**:
   - Train Chakr-Micro on this tiny batch for $150$ optimization steps using AdamW ($\eta = 10^{-3}$).
   - **Initial Benchmark Target Values**:
     - Loss: $\mathcal{L} < 0.05$
     - Accuracy: $\ge 99\%$ accuracy on next-token prediction for the memorized sequences
     - Duration: $\le 150$ steps
   - **Important Clarification**: The synthetic associative-recall benchmark is a diagnostic learnability test, not a universal pass/fail criterion for model quality. The target ($\text{loss} < 0.05$, $\ge 99\%$ accuracy, $\le 150$ steps) should be documented as an initial benchmark target, not a proof that the architecture is universally correct.

---

## 12. Minimum Success Criteria for Step 1

The success of Step 1 is established exclusively upon meeting the following criteria:
1. **Rigorous Technical Specification**: Delivery of `docs/STEP_01_NEURAL_CORE_SPEC.md` defining all mathematical flows, tensor shapes, parameter counts, and formulas.
2. **Architectural Decisions Documentation**: Delivery of `docs/ARCHITECTURE_DECISIONS.md` documenting all architectural trade-offs, candidate comparisons, and design rationale.
3. **Mathematical Consistency**: Complete mathematical continuity from integer token IDs to cross-entropy loss.
4. **Hardware Feasibility**: Pinned parameter budgets demonstrating realistic CPU execution within the detected hardware profile (32GB RAM, 6GB GPU).
5. **No Code Violation**: Zero model implementations, zero downloaded weights, and zero dependency additions prior to Step 1 review.

---

## 13. Major Engineering Risks & Mitigation Strategies

| Risk Description | Severity | Potential Impact | Mitigation Strategy |
| :--- | :--- | :--- | :--- |
| **CPU Memory Bandwidth Saturation** | High | Slow token-generation latency during autoregressive CPU inference. | Enforce small hidden dimensions ($d \le 384$ for CPU targets), weight tying, and prepare GQA / INT8 quantization. |
| **Numerical Instability in Float16 / Bfloat16** | Medium | Gradient underflow/overflow or NaN in attention softmax. | Maintain RMSNorm calculations and Softmax accumulation in `float32`; scale dot products strictly by $1/\sqrt{d_k}$. |
| **Residual Variance Explosion in Deep Networks** | Medium | Training divergence when scaling layers $L > 8$. | Scale residual projections ($W_O, W_{\text{down}}$) initialization by $1/\sqrt{2L}$ as per modern scaling practice. |
| **Python Interpreter Overhead on CPU** | Low | High loop overhead per token during generation. | Ensure pure vectorized tensor operations; isolate core layers so runtime can later be compiled to native C/C++ or ONNX. |
| **Module Coupling & Premature Complexity** | High | Architectural debt preventing independent testing of memory/skills. | Enforce strict input/output tensor contracts across module boundaries; no circular references. |

---

## 14. Step 1 Frozen vs Unfrozen Decisions

### Frozen for v0.1:
- decoder-only causal architecture
- pre-RMSNorm
- RoPE
- SwiGLU
- bias-free projections
- weight tying
- MHA
- maximum context of 512 tokens for Micro
- no pretrained weights

### Unfrozen:
- exact tokenizer/vocabulary
- optimizer hyperparameters
- initialization details
- exact dropout policy
- quantization implementation
- future GQA
- future dynamic depth
- future sparse computation

---

## 15. Implementation Prerequisite Rule

> **MANDATORY RULE**: No implementation should begin until the tokenizer specification and the final v0.1 tensor contracts have been reviewed.

---
*End of Specification — ChakrView Research Team*
