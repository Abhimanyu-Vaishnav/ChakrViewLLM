# ChakrView — Step 4.1: Neural Core Comprehensive Verification Report

**Document Version**: 2.0.0  
**Phase**: Step 4.1 — Neural Core Verification, CPU Baseline & Architecture Freeze  
**Model Target**: ChakrMicro v0.1 ($3,443,136$ Unique Trainable Parameters)  
**Date**: September 26, 2026  
**Status**: COMPLETE, AUDITED & ARCHITECTURE FROZEN  

---

## 1. Environment

The computational environment was fully verified on the physical development host:

| Environment Property | Specification / Measured Value | Verification Basis |
| :--- | :--- | :---: |
| **Operating System** | Microsoft Windows 11 Enterprise (x86_64 AMD64) | `[MEASURED]` |
| **Processor (CPU)** | Intel Core i9-13900H (14 Cores / 20 Threads, up to 5.4 GHz) | `[MEASURED]` |
| **System Memory (RAM)** | 32.0 GB DDR5 | `[MEASURED]` |
| **Python Version** | Python 3.14.7 (MSC v.1944 64-bit AMD64) | `[MEASURED]` |
| **PyTorch Version** | PyTorch 2.14.0+cpu | `[MEASURED]` |
| **BLAS Engine** | Intel MKL 2026.1, Intel MKL-DNN v3.12.0, AVX2 enabled | `[MEASURED]` |
| **GPU / CUDA Availability** | `torch.cuda.is_available() == False` (Strict CPU target) | `[MEASURED]` |
| **NumPy Version** | NumPy 2.5.3 | `[MEASURED]` |
| **PyTest Engine** | PyTest 9.1.1 (Pluggy 1.6.0) | `[MEASURED]` |
| **Git Working Tree** | Clean; zero untracked binary dependencies or wheels | `[VERIFIED]` |

Full environment audit details are archived in `docs/STEP_04_ENVIRONMENT.md`.

---

## 2. Architecture Implemented

The ChakrMicro v0.1 architecture was completely inspected against `docs/STEP_01_NEURAL_CORE_SPEC.md` and `docs/ARCHITECTURE_DECISIONS.md`:

| Component | Specification | Implementation Module | Status | Verification Finding |
| :--- | :--- | :--- | :---: | :--- |
| **Paradigm** | Decoder-only Causal Transformer | `chakrview/brain/model.py` | PASS | Autoregressive sequence modeling |
| **Layers ($N$)** | $6$ Stacked Blocks | `chakrview/brain/model.py` | PASS | Sequential `TransformerBlock` stack |
| **Model Dimension ($d_{\text{model}}$)** | $192$ | `chakrview/brain/config.py` | PASS | $192 \equiv 0 \pmod{64}$ |
| **Query Heads ($H$)** | $6$ | `chakrview/brain/attention.py` | PASS | $d_{\text{head}} = 32$ |
| **KV Heads ($H_{kv}$)** | $6$ (Simple MHA) | `chakrview/brain/attention.py` | PASS | $H_{kv} = H = 6$ (Standard MHA, no GQA in v0.1) |
| **Feed-Forward Network** | SwiGLU ($d_{\text{ff}} = 512$) | `chakrview/brain/feedforward.py` | PASS | $(\text{SiLU}(xW_{\text{gate}}) \odot xW_{\text{up}})W_{\text{down}}$ |
| **Normalization** | Pre-RMSNorm ($\epsilon = 10^{-5}$) | `chakrview/brain/normalization.py` | PASS | Applied before attention and FFN; bias-free |
| **Positional Encoding** | RoPE on $Q$ and $K$ | `chakrview/brain/rotary.py` | PASS | $\Theta = 10000.0$; pairwise 2D Givens rotation |
| **Value State RoPE** | $V$ Unrotated | `chakrview/brain/attention.py` | PASS | Value vectors bypass rotary encoding |
| **Linear Projections** | Bias-free ($b=0$) | All brain projection layers | PASS | `bias=False` across all linear transforms |
| **Weight Tying** | $W_{\text{out}} \equiv E^T$ | `chakrview/brain/output.py` | PASS | Shares identical storage with `TokenEmbedding.weight` |
| **Vocabulary Size ($V$)** | $V = 4096$ | `chakrview/brain/config.py` | PASS | Ratified provisional vocabulary |
| **Maximum Context ($T_{\text{max}}$)** | $512$ Tokens | `chakrview/brain/config.py` | PASS | Bounds verified; sequences $>512$ rejected |
| **Pretrained Weights** | Zero external weights | Whole repository | PASS | $100\%$ indigenous neural core |

Component-by-component implementation audit details are recorded in `docs/STEP_04_IMPLEMENTATION_AUDIT.md`.

---

## 3. Exact Parameter Count

Programmatically calculated directly from PyTorch tensor storage with weight tying confirmed through tensor identity (`model.lm_head.weight is model.embedding.weight`):

| Component | Dimensions / Derivation | Exact Parameters | Code Measured | Status |
| :--- | :--- | :---: | :---: | :---: |
| **Token Embedding ($E$)** | $4096 \times 192$ | $786,432$ | $786,432$ | PASS |
| **Attention $W_Q$ (6 layers)** | $6 \times (192 \times 192)$ | $221,184$ | $221,184$ | PASS |
| **Attention $W_K$ (6 layers)** | $6 \times (192 \times 192)$ | $221,184$ | $221,184$ | PASS |
| **Attention $W_V$ (6 layers)** | $6 \times (192 \times 192)$ | $221,184$ | $221,184$ | PASS |
| **Attention $W_O$ (6 layers)** | $6 \times (192 \times 192)$ | $221,184$ | $221,184$ | PASS |
| **Attention Subtotal (6 layers)** | $6 \times 147,456$ | **$884,736$** | **$884,736$** | PASS |
| **SwiGLU $W_{\text{gate}}$ (6 layers)** | $6 \times (192 \times 512)$ | $589,824$ | $589,824$ | PASS |
| **SwiGLU $W_{\text{up}}$ (6 layers)** | $6 \times (192 \times 512)$ | $589,824$ | $589,824$ | PASS |
| **SwiGLU $W_{\text{down}}$ (6 layers)** | $6 \times (512 \times 192)$ | $589,824$ | $589,824$ | PASS |
| **SwiGLU Subtotal (6 layers)** | $6 \times 294,912$ | **$1,769,472$** | **$1,769,472$** | PASS |
| **Layer RMSNorm Scales (6 layers)** | $6 \times 2 \times 192$ | $2,304$ | $2,304$ | PASS |
| **Final RMSNorm Scale** | $1 \times 192$ | $192$ | $192$ | PASS |
| **RMSNorm Subtotal** | $2304 + 192$ | **$2,496$** | **$2,496$** | PASS |
| **Output Head ($W_{\text{out}}$)** | Tied to $E$ ($W_{\text{out}} = E^T$) | **$0$ (Unique)** | **$0$** | PASS |
| **TOTAL UNIQUE PARAMETERS** | $786,432 + 884,736 + 1,769,472 + 2,496$ | **$3,443,136$** | **$3,443,136$** | PASS |
| **TOTAL TRAINABLE PARAMETERS** | All unique parameters requiring grad | **$3,443,136$** | **$3,443,136$** | PASS |
| **NON-TRAINABLE PARAMETERS** | Buffers (RoPE cos/sin, Causal Mask) | **$0$** | **$0$** | PASS |

Detailed accounting verified in `docs/STEP_04_EXACT_PARAMETER_COUNT.md`.

---

## 4. Tensor Shape Verification

Automated shape tests (`tests/test_brain_shapes_contract.py`) verified every major tensor transition across combinations of batch sizes $B \in \{1, 2, 4\}$ and sequence lengths $T \in \{1, 16, 128, 512\}$:

- Input token IDs: $[B, T]$
- Embedding lookup: $[B, T, 192]$
- Attention Q/K/V projections: $[B, 6, T, 32]$
- Rotary Positional Embedding (RoPE): $[B, 6, T, 32]$
- Attention score matrix: $[B, 6, T, T]$
- Attention output projection: $[B, T, 192]$
- SwiGLU FFN output: $[B, T, 192]$
- Final RMSNorm hidden state: $[B, T, 192]$
- Output LM Head logits: $[B, T, 4096]$
- Invariant confirmed: $H \times d_{\text{head}} = 6 \times 32 = 192 = d_{\text{model}}$ (zero accidental broadcasting).

---

## 5. Causal Attention Verification

Strict mathematical and empirical causality verified in `tests/test_brain_causality_strict.py`:

1. **Attention Mask Structure**:
   - Lower triangle & diagonal: Exactly $0.0$.
   - Upper triangle: Negative infinity converted to $-10^9$ in float32.
   - Post-softmax future token probabilities: Strictly $0.000000$.
2. **Deterministic ABCD Test**:
   - Given sequence $[A, B, C, D]$ vs $[A, B, C, D']$:
   - Hidden states and logits for prefix $[A, B, C]$: $\|\mathbf{z}_{\text{orig}}[:3] - \mathbf{z}_{\text{pert}}[:3]\|_{\infty} < 10^{-6}$ (floating-point identical).
   - Logits for perturbed token $D$ diverge significantly ($\Delta > 10^{-2}$).
3. **Layerwise Isolation**:
   - Verified that causality holds at every single Transformer block layer ($0$ through $5$), proving that residual streams and Pre-RMSNorm do not leak future information across steps.
4. **Batch Sample Independence**:
   - Perturbing sample $1$ in batch $B=2$ creates zero divergence ($0.000000$) in sample $0$.

---

## 6. Gradient Verification

Strict gradient verification conducted in `tests/test_brain_gradient_flow_strict.py`:

- **Synthetic Forward/Backward**: Random batch $B=2, T=16$.
- **Loss**: Finite, positive ($8.3386$), zero NaN, zero Inf.
- **Gradient Existence**: $100\%$ of all $56$ parameter tensors received gradients.
- **Gradient Finiteness**: Zero NaN, zero Inf across all gradient tensors.
- **Component Coverage**:
  - `model.embedding.weight.grad`: Non-zero, finite norm.
  - Attention projections ($W_Q, W_K, W_V, W_O$) across all 6 layers: Non-zero, finite.
  - SwiGLU projections ($W_{\text{gate}}, W_{\text{up}}, W_{\text{down}}$) across all 6 layers: Non-zero, finite.
  - Layer RMSNorm ($W_{\text{norm1}}, W_{\text{norm2}}$) across all 6 layers: Non-zero, finite.
  - Final RMSNorm ($W_{\text{final\_norm}}$): Non-zero, finite.
- **Weight Tying Gradient Behavior**:
  - Storage pointer equality confirmed: `model.lm_head.weight.grad is model.embedding.weight.grad`.
  - Gradient accumulates contributions from both input token lookup and output logits projection.

---

## 7. Initialization Health

Measured across all $3,443,136$ parameters in `docs/STEP_04_INITIALIZATION_HEALTH.md`:

- **NaN Count / %**: 0 (0.00%)
- **Inf Count / %**: 0 (0.00%)
- **Accidental All-Zero Tensors**: None
- **Standard Projections & Embedding**: $\mu \approx 0.0000, \sigma \approx 0.0200$, bounded within $[-0.10, +0.10]$.
- **Residual Projections ($W_O, W_{\text{down}}$)**: Scaled by $1 / \sqrt{2N} = 1 / \sqrt{12} \approx 0.2887$; measured $\sigma \approx 0.00577$ (target $0.00577$).
- **RMSNorm Scales**: Exactly $1.000000$ (identity transformation at initialization).

---

## 8. Numerical Stability

Precision evaluation conducted on CPU across FP32, BF16, and FP16 in `docs/STEP_04_NUMERICAL_STABILITY.md`:

- **FP32**: Production baseline. Zero NaN/Inf, logit range $[-1.26, +2.96]$, cross-entropy loss $8.3386$ (matches theoretical uniform entropy $\ln(4096) = 8.3178$).
- **BF16**: Executable on CPU with AVX2/oneDNN emulation. Zero NaN/Inf, loss $8.3384$.
- **FP16**: Executable on CPU. Zero NaN/Inf, loss $8.3387$.
- **Recommendation**: Default training and CPU baseline inference strictly frozen at **FP32**. Low-precision inference deferred to post-training quantization phases.

---

## 9. CPU Benchmark

Empirically measured using 4 CPU threads on Intel i9-13900H (`docs/STEP_04_CPU_BASELINE.md`):

- **Model Initialization**: $25.97\text{ ms}$
- **Static Parameter Memory (FP32)**: $13.13\text{ MiB}$
- **Base Process RAM**: $207.1\text{ MB}$
- **Peak Process RAM**: $278.2\text{ MB}$

### Steady-State Latency & Throughput ($B=1$ and $B=2$):

| Batch ($B$) | Context ($T$) | Cold Run (ms) | Warm Median (ms) | Warm p95 (ms) | ms / Token | Throughput (tok/s) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | **16** | $3.52\text{ ms}$ | **$2.19\text{ ms}$** | $2.45\text{ ms}$ | $0.1370\text{ ms}$ | **$7,297.3\text{ tok/s}$** |
| **1** | **64** | $3.99\text{ ms}$ | **$3.88\text{ ms}$** | $4.21\text{ ms}$ | $0.0606\text{ ms}$ | **$16,488.5\text{ tok/s}$** |
| **1** | **128** | $6.42\text{ ms}$ | **$5.88\text{ ms}$** | $6.45\text{ ms}$ | $0.0459\text{ ms}$ | **$21,776.5\text{ tok/s}$** |
| **1** | **256** | $10.53\text{ ms}$ | **$10.02\text{ ms}$** | $11.10\text{ ms}$ | $0.0391\text{ ms}$ | **$25,549.9\text{ tok/s}$** |
| **1** | **512** | $21.55\text{ ms}$ | **$20.26\text{ ms}$** | $21.90\text{ ms}$ | $0.0396\text{ ms}$ | **$25,265.7\text{ tok/s}$** |
| **2** | **16** | $3.42\text{ ms}$ | **$3.16\text{ ms}$** | $3.55\text{ ms}$ | $0.0988\text{ ms}$ | **$10,123.1\text{ tok/s}$** |
| **2** | **64** | $5.71\text{ ms}$ | **$6.48\text{ ms}$** | $7.15\text{ ms}$ | $0.0506\text{ ms}$ | **$19,768.3\text{ tok/s}$** |
| **2** | **128** | $10.95\text{ ms}$ | **$9.51\text{ ms}$** | $10.80\text{ ms}$ | $0.0372\text{ ms}$ | **$26,905.2\text{ tok/s}$** |
| **2** | **256** | $16.53\text{ ms}$ | **$16.73\text{ ms}$** | $18.40\text{ ms}$ | $0.0327\text{ ms}$ | **$30,601.5\text{ tok/s}$** |
| **2** | **512** | $49.55\text{ ms}$ | **$41.16\text{ ms}$** | $44.80\text{ ms}$ | $0.0402\text{ ms}$ | **$24,877.0\text{ tok/s}$** |

---

## 10. Memory Analysis

Detailed theoretical analysis presented in `docs/STEP_04_MEMORY_MODEL.md`:

- **Static Weights**:
  - FP32: $13.13\text{ MiB}$
  - FP16 / BF16: $6.57\text{ MiB}$
  - INT8: $3.28\text{ MiB}$
  - INT4: $1.64\text{ MiB}$
- **KV Cache Footprint**:
  - Elements: $2 \times N \times H_{kv} \times d_{\text{head}} \times B \times T = 2,304 \times B \times T$
  - At $T=512, B=1$: $4.50\text{ MiB}$ (FP32), $2.25\text{ MiB}$ (FP16), $1.13\text{ MiB}$ (INT8).
- **Activations (Inference)**: Peak layer workspace is $< 8.5\text{ MiB}$ at full context $T=512$.
- **Standalone Embedded C Runtime (Projected)**: Total footprint $\approx 26.1\text{ MiB}$ (FP32) or $< 7\text{ MiB}$ (INT8).

---

## 11. Synthetic Learnability

Associative recall experiment verified in `docs/STEP_04_SYNTHETIC_LEARNABILITY.md` and automated in `tests/test_brain_synthetic_learnability.py`:

- **Task**: Recall value $V$ mapped to queried key $K$ across three in-context pairs.
- **Initial Loss**: $8.4409$ (Accuracy $0.0\%$)
- **Final Loss (40 steps AdamW)**: **$0.0077$** (Accuracy **$100.0\%$**)
- **Elapsed Training Time**: $1.05\text{ s}$ on CPU.
- **Finding**: Pre-RMSNorm, RoPE, MHA, SwiGLU, and tied output projections route gradients smoothly and enable associative retrieval.

---

## 12. Known Limitations

1. **Python Dynamic Overhead**: Small-batch CPU execution contains Python interpreter and object allocation overhead; compiled C/C++ runtimes will be significantly faster.
2. **Fixed Maximum Sequence Length**: Hard upper bound at $T_{\text{max}} = 512$. Long-context extrapolation (e.g. RoPE scaling / YaRN) is not implemented in v0.1.
3. **No KV Cache Reuse in Full Forward**: The current forward pass is a sequence-parallel prompt processor. Autoregressive single-token step decoding with persistent KV cache is reserved for Step 6.
4. **Hardware Validation Boundaries**: Direct benchmarking on older 28nm processors or low-end ARM chips (Cortex-A53) has not yet been physically conducted.

---

## 13. Bugs Found and Fixed

1. **Direct pip stall under headless stdout**: Streaming large PyPI wheels through background pipes caused buffer stalls; resolved by clean, verified direct wheel download and PEP 427 compliance.
2. **Causal mask softmax floating point safety**: Converted $-\infty$ to safe $-10^9$ in float32 within `CausalMask` to prevent SIMD NaN generation during softmax on edge CPU backends.
3. **Stand-alone script module import paths**: Standalone benchmark scripts failed to resolve `chakrview`; resolved by adding repository root dynamically to `sys.path`.
4. **Layerwise causality isolation interface**: Fixed test call in causality test suite to use `model.layers` with self-contained attention masking.

---

## 14. Remaining Risks

1. **Corpus Quality & Pre-Training Dynamics**: Architecture and gradient flow are verified, but language model output quality depends entirely on pre-training corpus curriculum and tokenizer entropy.
2. **Single-Thread CPU Performance**: On heavily constrained single-core systems, latency will scale inversely with clock speed and memory bandwidth.

---

## 15. Decision

Based on the complete empirical and mathematical evidence across Phases 1 through 14:
- All 200 repository tests pass (including 64 neural-core specific tests).
- Parameters ($3,443,136$), weight tying, causality, and gradient flow are $100\%$ verified.
- The model learns deterministically without gradient pathologies.

### **VERIFIED — READY FOR TRAINING**
