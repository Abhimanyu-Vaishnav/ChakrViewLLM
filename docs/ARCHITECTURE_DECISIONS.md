# Architecture Decision Records (ADR) — ChakrView Step 1

**Project**: ChakrView  
**Phase**: Step 1 — Foundational Neural Core Specification  
**Status**: ACTIVE & RATIFIED  

---

## Overview

This document records the foundational architectural decisions made during Step 1 for the design of the **ChakrView Neural Core (Chakr-Core v0.1)**. Each record details the context, the decision made, the alternatives considered, and the technical reasoning behind it.

> **MANDATORY ENGINEERING RULE**: No implementation should begin until the tokenizer specification and the final v0.1 tensor contracts have been reviewed.

---

## ADR 01: Selection of Core Sequence Modeling Paradigm

### Context
ChakrView is an indigenous AI research initiative aimed at developing an intelligent, hardware-efficient AI core. A sequence modeling paradigm had to be chosen for the foundational language core. The system must prioritize CPU inference feasibility, minimal RAM consumption, deployment on low-spec/edge devices, implementation clarity, and absolute mathematical stability.

### Candidates Evaluated
1. **Autoregressive Causal Transformer Decoder** (e.g., GPT/Llama-style causal self-attention).
2. **Modern Recurrent Architectures** (e.g., Linear RNN, RWKV, Griffin-style recurrence).
3. **State-Space Models (SSM)** (e.g., S4, S5, Mamba selective state spaces).
4. **Hybrid Architectures** (e.g., Interleaved Attention + SSM layers).

### Decision
Adopt the **Autoregressive Causal Transformer Decoder** as the foundational paradigm for Chakr-Core v0.1.

### Reasoning & Trade-offs
- **Zero Proprietary/Custom Kernel Dependencies**: SSMs and modern linear RNNs require parallel associative scans to achieve reasonable training speeds. In practice, this demands custom CUDA or C extensions (e.g., Triton, hardware-specific SRAM tiling), which directly violates the requirement for zero unnecessary dependencies and universal CPU reproducibility.
- **CPU BLAS Saturation**: Matrix multiplications in transformers directly leverage standard, highly mature BLAS libraries (OpenBLAS, Intel MKL, OneDNN) and SIMD instruction sets (AVX2, AVX-512, ARM NEON).
- **Optimization Stability**: Causal attention provides the most stable and well-conditioned gradient flow, ensuring early research is not derailed by vanishing or exploding hidden recurrent states.
- **Portability**: Transformers export seamlessly to ONNX, GGML, WebAssembly, and bare-metal embedded C.
- **Negligible Quadratic Penalty at Target Scales**: At small context lengths ($T \le 1024$), the quadratic memory penalty of attention is under $10\text{ MB}$, completely eliminating the primary disadvantage of transformers on low-resource hardware.

---

## ADR 02: Normalization Technique — Pre-RMSNorm over LayerNorm

### Context
Deep networks require normalization to maintain stable gradient dynamics across stacked layers. The placement and mathematical formulation of normalization significantly impact both compute overhead and training stability.

### Candidates Evaluated
1. **Post-LayerNorm** (Standard Attention Is All You Need).
2. **Pre-LayerNorm** (Mean-centered normalization before sub-layers).
3. **Pre-RMSNorm** (Root Mean Square Normalization before sub-layers).

### Decision
Adopt **Pre-RMSNorm** before each sub-layer (attention and feed-forward), with an additional final RMSNorm before the output projection.

### Reasoning & Trade-offs
- **Computational Efficiency**: LayerNorm calculates both the mean and variance: $\mu = \frac{1}{d}\sum x_i$ and $\sigma^2 = \frac{1}{d}\sum (x_i - \mu)^2$. RMSNorm calculates only the root mean square: $\text{RMS}(x) = \sqrt{\frac{1}{d}\sum x_i^2 + \epsilon}$. By dispensing with mean-centering, RMSNorm achieves a $7\text{--}12\%$ speedup in normalization kernels on CPU architectures.
- **Training Stability**: Pre-normalization establishes an unobstructed residual highway where activations can backpropagate directly across the entire network depth, preventing the vanishing gradient problems typical of Post-LayerNorm without needing delicate warmup schedules.

---

## ADR 03: Positional Encoding — Rotary Position Embeddings (RoPE)

### Context
Autoregressive transformers are permutation-invariant and require an encoding of token order.

### Candidates Evaluated
1. **Learned Absolute Positional Embeddings** (e.g., original GPT-2).
2. **Sinusoidal Absolute Positional Embeddings** (e.g., original Transformer).
3. **Linear Biases / ALiBi** (e.g., Press et al.).
4. **Rotary Position Embeddings (RoPE)** (Su et al.).

### Decision
Adopt **Rotary Position Embeddings (RoPE)** applied directly to query and key head vectors.

### Reasoning & Trade-offs
- **Zero Additional Trainable Parameters**: RoPE encodes positions through fixed orthogonal rotation matrices applied to existing 2D projection slices, introducing zero parameter overhead.
- **Relative Distance Awareness**: The inner product between query at position $m$ and key at position $n$ is a strict function of relative displacement $(m - n)$, which aligns with natural language semantics.
- **Context Extensibility**: RoPE possesses proven mathematical properties for sequence length extrapolation (via frequency scaling/interpolation) in subsequent iterations without retraining core weights.

---

## ADR 04: Feed-Forward Sub-Layer — SwiGLU over GELU/ReLU

### Context
The feed-forward sub-layer (FFN) accounts for approximately two-thirds of a transformer's total parameters and non-linear expressive capacity.

### Candidates Evaluated
1. **Standard Two-Matrix MLP with ReLU** ($W_2 \text{ReLU}(W_1 x)$).
2. **Standard Two-Matrix MLP with GELU** ($W_2 \text{GELU}(W_1 x)$).
3. **Three-Matrix Gated Linear Unit with Swish (SwiGLU)** ($W_{\text{down}} (\text{Swish}(W_{\text{gate}} x) \odot W_{\text{up}} x)$).

### Decision
Adopt **SwiGLU** with intermediate dimension $d_{ff} \approx \frac{8}{3}d$ (aligned to multiples of 32/64).

### Reasoning & Trade-offs
- **Enhanced Expressive Density**: Extensive empirical literature (Chinchilla, PaLM, LLaMA) demonstrates that gated linear units provide substantially lower perplexity per parameter and FLOP than two-matrix GELU/ReLU networks.
- **Smooth Gradient Landscape**: The combination of bilinear gating and smooth Swish ($\text{SiLU}$) activation eliminates dead-neuron pathologies and produces superior gradient flow during backpropagation.
- **Resource Neutrality**: To keep FLOPs and parameter counts comparable to a standard $4d$ two-matrix FFN, the intermediate dimension is scaled down to $\approx \frac{8}{3}d$, yielding equal compute with superior representation quality.

---

## ADR 05: Bias-Free Linear Projections

### Context
Traditional dense layers include additive bias vectors ($y = Wx + b$).

### Decision
**Omit additive bias vectors** ($b = 0$) from all linear projections ($W_Q, W_K, W_V, W_O, W_{\text{gate}}, W_{\text{up}}, W_{\text{down}}$).

### Reasoning & Trade-offs
- **Memory Footprint & Bandwidth**: Eliminates redundant 1D parameter vectors and simplifies memory access patterns during inference.
- **Inference Speed**: Bias-free GEMMs map directly to raw matrix multiplication routines without trailing vector-add operations, benefiting constrained CPU caches.
- **Length Extrapolation & Stability**: Removing biases eliminates systematic drift in residual activation norms across longer sequence lengths.

---

## ADR 06: Weight Tying between Embeddings and Unembedding Head

### Context
In language models, the token embedding matrix $W_E \in \mathbb{R}^{V \times d}$ and the output unembedding matrix $W_{\text{head}} \in \mathbb{R}^{d \times V}$ perform dual operations between the discrete vocabulary space and continuous hidden space.

### Decision
Enable **Weight Tying** as the primary configuration: $W_{\text{head}} = W_E^T$.

### Reasoning & Trade-offs
- **Dramatic Parameter Savings**: In small and micro models, the vocabulary matrix constitutes a large fraction of the total parameter budget. For Chakr-Micro ($V=4096, d=192$, where $V=4096$ is provisional until tokenizer research is completed), weight tying eliminates $786,432$ parameters (a nearly $20\%$ reduction). For Chakr-Small, it saves $3.14\text{M}$ parameters.
- **Cache Locality on Low-RAM Devices**: Reusing the same matrix in memory reduces resident working set size during both training and CPU inference.
- **Theoretical Coherence**: Mapping tokens into and out of the identical semantic manifold enforces geometric consistency between input representations and output predictions.

---

## ADR 07: Prototype Sizing — Adopting Chakr-Micro (~3.44M Params)

### Context
ChakrView requires a concrete model sizing for the initial implementation and empirical verification phase.

### Decision
Select **Chakr-Micro (~3.44M parameters, 6 layers, hidden dim 192, 6 heads, context 512, provisional vocabulary size 4096)** as the FIRST trainable prototype.

### Reasoning & Trade-offs
- **Rapid Iterative Feedback Loop**: A complete forward, backward, and optimization cycle on Chakr-Micro takes only a few milliseconds on a standard CPU core.
- **Simple MHA Execution**: Implements simple Multi-Head Attention (MHA). GQA is strictly an extension point for future versions and will not be implemented in v0.1.
- **Cache-Locality Engineering Hypothesis**: The quantized weight footprint is small enough to make cache-local execution plausible on some CPUs; this must be verified by benchmark.
- **Memory Estimates Scope Clarification**: Static model-weight memory estimates (FP16: ~6.89 MB, INT8: ~3.44 MB, INT4: ~1.72 MB) strictly exclude dynamic activations, runtime allocator overhead, temporary tensors, tokenizer memory, KV cache unless explicitly stated, and framework/runtime engine overhead.
- **Deterministic Bug Diagnosis**: Architectural errors, incorrect masking, or gradient bugs are revealed within seconds on synthetic test suites, eliminating wasted compute time.

---

## ADR 08: Extensibility Design for Future Optimization Steps

### Context
While Step 1 prohibits implementing advanced optimizations prematurely, the architecture must not require costly rewrites when efficiency features are introduced in future steps.

### Decision
Integrate architectural hooks directly into layer interfaces:
1. **Simple Multi-Head Attention (MHA) for v0.1**: Chakr-Micro v0.1 will implement simple MHA ($H_{kv} = H$). Grouped-Query Attention (GQA) is strictly an architectural extension point for future model iterations and will NOT be implemented in v0.1.
2. **KV-Cache Ready**: Design attention signatures with optional `past_key_value` inputs.
3. **Quantization Ready**: Keep all operations as standard decoupled GEMMs compatible with INT8/INT4 weight-only and activation quantization.
4. **Modular FFN**: Encapsulate the feed-forward mechanism so it can later be upgraded to a Mixture-of-Experts (MoE) block.

---

## ADR 09: Decoupled Architectural Boundaries

### Context
To maintain indigenous code hygiene and prevent spaghettification as ChakrView grows to incorporate memory, tools, and skills, architectural boundaries must be strictly established from Day 1.

### Decision
Enforce the following non-negotiable boundaries:
1. `brain/`: Strictly pure neural network module definitions. Pure tensor in $\to$ pure tensor out. Zero awareness of raw text, files, tokenizers, datasets, or training loops.
2. `tokenization/`: Manages text $\leftrightarrow$ token IDs. Contains zero neural network parameters.
3. `training/`: Orchestrates datasets, loss functions, optimizers, and learning rate schedules. Does not redefine model layers.
4. `inference/`: Manages generation loops, KV-cache updates, and sampling. Keeps runtime logic out of the core network.
5. `memory/`, `skills/`, `tools/`: Fully decoupled subsystems interacting only through clean higher-level interfaces.

---

## ADR 10: Provisional Adoption of Vocabulary Size V = 4096 and Tokenizer Interface Contract

### Context
Step 2.3 conducted an empirical benchmark across four candidate vocabulary sizes: $V = 2048$, $V = 4096$, $V = 8192$, and $V = 16384$ on an 8-category research corpus (Devanagari Hindi, English, Hinglish, Code, Numbers, Math, Unicode, and Mixed).

### Decision
Adopt **$V = 4096$ as the provisional tokenizer vocabulary configuration** for the Chakr-Micro neural core, and freeze the tokenizer-to-neural-core handoff interface:
- **Valid token IDs**: $[0, 4095]$
- **Special tokens**: $\langle\text{BOS}\rangle=0, \langle\text{EOS}\rangle=1, \langle\text{PAD}\rangle=2$
- **Foundational byte primitives**: $3 \dots 258$ (256 bytes)
- **Learned merges**: $259 \dots 4095$ (3,837 merges)
- **Input embedding shape**: $[4096, 192]$
- **Tied output projection shape**: $[192, 4096]$

> **PROVISIONAL STATUS NOTICE**:  
> **"V=4096 is the current provisional tokenizer configuration selected from the Step 2.3 benchmark corpus. It remains replaceable if later neural-core experiments demonstrate that another vocabulary size provides a better system-level tradeoff."**

### Reasoning & Trade-offs
- **Pareto Efficiency on Benchmark**: $V = 4096$ reduced validation tokens by $9.1\%$ compared to $V = 2048$, achieving $2.43\text{ tokens/word}$ for Hindi and $1.77$ for Hinglish on the research corpus.
- **Healthy Parameter Ratio**: In Chakr-Micro ($d=192$), an embedding table of 4096 tokens consumes $786,432$ parameters ($22.8\%$ of total model parameters), leaving $77.2\%$ for relational reasoning blocks.
- **Diminishing Returns on Larger Vocabularies**: $V = 16384$ exhausted corpus merge combinations and provided only a $0.1\%$ token improvement over $V = 8192$ while doubling encoding latency and inflating vocabulary RAM.
- **Decoupled Architecture**: All transformer layers take pure tensors of shape $[\text{batch}, \text{sequence}, 192]$ and emit $[\text{batch}, \text{sequence}, 192]$. Replacing $V = 4096$ with $V = 8192$ later requires zero structural changes to attention blocks, MLP blocks, normalization layers, or RoPE.

---

## ADR 16: Empirical Ratification of Vocabulary Size V = 4096, Numeric Strategy & Pre-tokenization Configuration

### Context
Step 3 executed a comprehensive empirical benchmark across four candidate vocabulary sizes ($V = 2048, 4096, 8192, 16384$) on a controlled multi-domain research corpus across 8 distinct categories, evaluated three candidate numeric tokenization strategies (Candidate A: single digits, Candidate B: 2-digit chunks, Candidate C: normal BPE), and compared two pre-tokenization variants (Variant A: raw byte BPE vs Variant B: grapheme-aware pre-tokenization + BPE).

### Decision
1. **Ratify $V = 4096$** as the definitive vocabulary configuration for Chakr-Micro v0.1:
   - $3$ special tokens ($\langle\text{BOS}\rangle=0, \langle\text{EOS}\rangle=1, \langle\text{PAD}\rangle=2$)
   - $256$ foundational byte primitives (IDs $3\dots258$)
   - $3,837$ learned BPE merges (IDs $259\dots4095$)
2. **Retain Candidate C (Normal BPE)** as the primary default numeric tokenization strategy, with **Candidate A (Single Digits)** preserved as an isolated modular adapter for arithmetic-intensive tasks.
3. **Select Variant A (Raw Byte BPE)** as the pre-tokenization strategy over Variant B, avoiding a $+22.6\%$ sequence expansion and $2\times$ CPU regex latency overhead.

### Reasoning & Trade-offs
- **Pareto Optimality for Micro Scale**: $V = 4096$ delivers $2.17\text{ tokens/word}$ on Hindi and $2.49$ on English on unseen validation data, outperforming $V = 2048$ ($2.40$ Hindi, $2.88$ English) while consuming only $902.4\text{ KB}$ of static memory.
- **Model Parameter Balance**: In Chakr-Micro ($d_{\text{model}} = 192$), an embedding matrix of $4096$ tokens consumes $786,432$ parameters ($22.84\%$ of the total $3,443,712$ model parameters), leaving $77.16\%$ for relational reasoning Transformer blocks. In contrast, $V = 8192$ consumes $1,572,864$ parameters ($45.67\%$ of total weights), over-allocating capacity to static lookup tables.
- **Merge Exhaustion at Scale**: Candidate $V = 16384$ completely exhausted available unique pairs on the controlled corpus at $12,018$ tokens ($11,759$ merges), providing almost zero additional compression (+0.032 bytes/tok) while consuming $67.0\%$ of the entire model parameter budget.
- **Lossless Invariant Across All Suites**: All candidates passed $100\%$ lossless round-trip reconstruction on all corpus categories, the Unicode/Indic adversarial suite, and arbitrary raw byte test cases.

---

## Step 4 Frozen vs Unfrozen Decisions

### Frozen:
- decoder-only causal architecture
- model dimension: $d_{\text{model}} = 192$
- number of layers: $N = 6$
- attention heads: $H = 6, H_{kv} = 6$ (Simple MHA)
- head dimension: $d_{\text{head}} = 32$
- FFN intermediate dimension: $d_{\text{ff}} = 512$ (exact $\frac{8}{3} d_{\text{model}}$ power-of-2 cache alignment)
- pre-RMSNorm ($\epsilon = 10^{-5}$)
- RoPE positional embedding ($\Theta = 10000.0, d_{\text{rot}} = 32$)
- SwiGLU feed-forward activation
- bias-free linear projections ($b = 0$)
- weight tying ($W_{\text{out}} = E^T$)
- maximum context of 512 tokens for Chakr-Micro ($T_{\text{max}} = 512$)
- total parameter count: $3,443,136$
- no pretrained weights (100% indigenous architecture)
- byte-level fallback (256 base bytes, zero `<UNK>`)
- special token IDs: `<BOS>`: 0, `<EOS>`: 1, `<PAD>`: 2
- vocabulary size: $V = 4096$ ratified for v0.1
- pre-tokenization strategy: Variant A (Raw Byte BPE) ratified for v0.1
- numeric default strategy: Candidate C (Normal BPE) ratified for v0.1
- lossless round-trip invariant: $\text{Decode}(\text{Encode}(S)) \equiv S$
- tokenizer-to-neural-core handoff contract: token IDs $\in [0, 4095] \to [B, T, 192]$
- neural core output logits contract: $[B, T, 4096]$

### Unfrozen (Provisional / Experimental):
- learning rate schedule (cosine vs linear warmup decay)
- optimizer selection (AdamW vs Lion vs Sophia)
- weight initialization scale tuning
- post-training quantization (INT8/INT4 kernels)
- future GQA (extension point for larger models $d \ge 512$)
- future dynamic depth / sparse computation
- inference execution engines (pure C++, GGML, ONNX, WASM)

---

## Implementation Prerequisite Rule

> **MANDATORY RULE**: Step 4 Neural Core Architecture Specification and Tensor Contracts are complete and frozen. Implementation of the neural core model class, transformer layers, and training pipeline shall strictly commence upon official approval of Step 5 instructions.

---
*End of Architecture Decision Records*


