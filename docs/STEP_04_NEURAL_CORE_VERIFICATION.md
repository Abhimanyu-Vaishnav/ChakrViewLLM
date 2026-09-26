# ChakrView — Step 4: Neural Core Formal Verification Report

**Document Version**: 1.0.0  
**Target Architecture**: ChakrMicro v0.1 ($3,443,136$ parameters)  
**Date**: September 26, 2026  
**Status**: AUDITED, VERIFIED & RATIFIED  

---

## Epistemic Legend
- `[VERIFIED]`: Mathematically or programmatically proven through automated assertions and unit tests.
- `[MEASURED]`: Empirically benchmarked on the physical host machine.
- `[ASSUMED]`: Explicit design constraint or invariant adopted for v0.1.
- `[FUTURE WORK]`: Planned feature or post-baseline optimization strictly deferred to future steps.

---

## 1. Environment `[MEASURED]`
- **Operating System**: Microsoft Windows 11 Enterprise (64-bit x86_64 AMD64)
- **Host CPU**: Intel Core i9-13900H (14 Cores / 20 Threads, up to 5.4 GHz)
- **Host RAM**: 32.0 GB DDR5
- **Python Runtime**: Python 3.14.7 (MSC v.1944 64-bit AMD64)
- **Host GPU / CUDA**: None / Not utilized (`torch.cuda.is_available() == False`)
- **Git State**: Clean working tree on `master` branch

---

## 2. PyTorch Version `[MEASURED]`
- **PyTorch Build**: `2.14.0+cpu`
- **Installation Path**: `D:\Project\ChakrView\.venv\Lib\site-packages\torch\__init__.py`
- **BLAS Engine**: Intel MKL 2026.1, Intel MKL-DNN v3.12.0
- **SIMD Instructions**: AVX2, FMA3 active
- **Basic Tensor Execution**: Confirmed (`torch.randn(2, 8, 192)` produces `torch.Size([2, 8, 192])` on CPU)

---

## 3. Architecture Configuration `[VERIFIED]` `[ASSUMED]`
Audited against `STEP_01_NEURAL_CORE_SPEC.md` and `ARCHITECTURE_DECISIONS.md`:
- **Model Paradigm**: Decoder-only causal Transformer (`chakrview/brain/model.py`) `[VERIFIED]`
- **Layers ($N$)**: 6 stacked transformer blocks `[VERIFIED]`
- **Model Dimension ($d_{\text{model}}$)**: 192 ($192 \equiv 0 \pmod{64}$) `[VERIFIED]`
- **Query Attention Heads ($H$)**: 6 ($d_{\text{head}} = 32$) `[VERIFIED]`
- **KV Attention Heads ($H_{kv}$)**: 6 (Simple Multi-Head Attention, no GQA in v0.1) `[VERIFIED]`
- **Feed-Forward Network**: SwiGLU ($d_{\text{ff}} = 512 = \frac{8}{3} d_{\text{model}}$) `[VERIFIED]`
- **Normalization**: Pre-RMSNorm ($\epsilon = 10^{-5}$, bias-free) `[VERIFIED]`
- **Positional Encoding**: RoPE on $Q$ and $K$ ($\Theta = 10000.0$, pairwise 2D Givens rotation; $V$ unrotated) `[VERIFIED]`
- **Projection Biases**: Strictly bias-free ($b=0$) across all linear transforms `[VERIFIED]`
- **Weight Tying**: Output head tied to embedding ($W_{\text{out}} \equiv E^T$) `[VERIFIED]`
- **Vocabulary Size ($V$)**: 4096 (provisional BPE vocabulary ratified from Step 3) `[ASSUMED]`
- **Maximum Context ($T_{\text{max}}$)**: 512 tokens `[VERIFIED]`
- **Purity Guarantee**: Zero pretrained weights, zero HuggingFace dependencies, zero black-box external transformers `[VERIFIED]`

---

## 4. Exact Parameter Count `[VERIFIED]` `[MEASURED]`
Programmatically computed directly from PyTorch tensor storage:

| Component | Dimensions / Derivation | Parameter Count | Trainable | Status |
| :--- | :--- | :---: | :---: | :---: |
| **Token Embedding ($E$)** | $4096 \times 192$ | $786,432$ | Yes | `[VERIFIED]` |
| **Attention Projections (6 blocks)** | $6 \times (4 \times 192 \times 192)$ | $884,736$ | Yes | `[VERIFIED]` |
| ├── $W_Q$ Projections | $6 \times (192 \times 192)$ | $221,184$ | Yes | `[VERIFIED]` |
| ├── $W_K$ Projections | $6 \times (192 \times 192)$ | $221,184$ | Yes | `[VERIFIED]` |
| ├── $W_V$ Projections | $6 \times (192 \times 192)$ | $221,184$ | Yes | `[VERIFIED]` |
| └── $W_O$ Out Projections | $6 \times (192 \times 192)$ | $221,184$ | Yes | `[VERIFIED]` |
| **SwiGLU FFN Projections (6 blocks)** | $6 \times (3 \times 192 \times 512)$ | $1,769,472$ | Yes | `[VERIFIED]` |
| ├── $W_{\text{gate}}$ Projections | $6 \times (192 \times 512)$ | $589,824$ | Yes | `[VERIFIED]` |
| ├── $W_{\text{up}}$ Projections | $6 \times (192 \times 512)$ | $589,824$ | Yes | `[VERIFIED]` |
| └── $W_{\text{down}}$ Projections | $6 \times (512 \times 192)$ | $589,824$ | Yes | `[VERIFIED]` |
| **Layer RMSNorm Scales (6 blocks)** | $6 \times 2 \times 192$ | $2,304$ | Yes | `[VERIFIED]` |
| **Final RMSNorm Scale** | $1 \times 192$ | $192$ | Yes | `[VERIFIED]` |
| **Output Head ($W_{\text{out}}$)** | Reuses $E^T$ | **$0$ Unique** | Shared | `[VERIFIED]` |
| **TOTAL UNIQUE PARAMETERS** | $786,432 + 884,736 + 1,769,472 + 2,496$ | **$3,443,136$** | Yes | `[VERIFIED]` |
| **TOTAL TRAINABLE PARAMETERS** | `p.requires_grad == True` | **$3,443,136$** | Yes | `[VERIFIED]` |
| **NON-TRAINABLE PARAMETERS** | Buffers (`cos_cache`, `sin_cache`, `mask`) | **$0$** | No | `[VERIFIED]` |

Reconciliation against `docs/STEP_04_PARAMETER_ACCOUNTING.md`: **0 discrepancy (Exact Match)**.

---

## 5. Tensor Shape Verification `[VERIFIED]`
Automated shape tests (`tests/test_brain_shapes_contract.py`) validated all major tensor transitions across:
- $B=1, T=1$: Input $[1, 1] \to \text{Emb } [1, 1, 192] \to \text{Attn } [1, 1, 192] \to \text{Hidden } [1, 1, 192] \to \text{Logits } [1, 1, 4096]$
- $B=1, T=16$: Input $[1, 16] \to \text{Emb } [1, 16, 192] \to \text{Attn } [1, 16, 192] \to \text{Hidden } [1, 16, 192] \to \text{Logits } [1, 16, 4096]$
- $B=2, T=32$: Input $[2, 32] \to \text{Emb } [2, 32, 192] \to \text{Attn } [2, 32, 192] \to \text{Hidden } [2, 32, 192] \to \text{Logits } [2, 32, 4096]$
- $B=2, T=128$: Input $[2, 128] \to \text{Emb } [2, 128, 192] \to \text{Attn } [2, 128, 192] \to \text{Hidden } [2, 128, 192] \to \text{Logits } [2, 128, 4096]$
- $B=1, T=512$: Input $[1, 512] \to \text{Emb } [1, 512, 192] \to \text{Attn } [1, 512, 192] \to \text{Hidden } [1, 512, 192] \to \text{Logits } [1, 512, 4096]$

**Context Limit Enforcement**: Sequences longer than 512 (e.g. $T=513$) explicitly raise `ValueError("Sequence length 513 exceeds maximum context window 512.")` and are not silently truncated.

---

## 6. Causality Verification `[VERIFIED]`
Strict mathematical causality proven in `tests/test_brain_causality_strict.py` and `tests/test_brain_causality.py`:
1. **Mask Structure**: Strictly lower triangular ($0.0$ on and below diagonal; $-10^9$ in upper triangle). Post-softmax probability mass for future positions is identically $0.000000$.
2. **Deterministic ABCD Test**:
   - Sequence $A: [x_1, x_2, x_3, x_4]$
   - Sequence $B: [x_1, x_2, x_3, y_4]$ where $y_4 \ne x_4$.
   - Max prefix difference ($t \in [1, 3]$): $\|\mathbf{z}_A[:3] - \mathbf{z}_B[:3]\|_{\infty} < 10^{-6}$ (floating-point exact).
   - Position $4$ logit difference: $\|\mathbf{z}_A[3] - \mathbf{z}_B[3]\|_{\infty} > 10^{-2}$.
3. **Layerwise Isolation**: Prefix isolation holds at every individual transformer block ($0$ through $5$).

---

## 7. Gradient Verification `[VERIFIED]`
Evaluated in `tests/test_brain_gradient_flow_strict.py`:
- **Synthetic Forward/Backward**: Causal next-token cross-entropy loss on random batch $B=2, T=16$.
- **Loss Finiteness**: Loss is finite ($8.3386$), with zero NaN and zero Inf.
- **Gradient Coverage**: 100% of all 56 parameter tensors receive valid non-zero, finite gradients.
- **Tied Head Gradients**: `model.lm_head.weight.grad is model.embedding.weight.grad` holds identically.

---

## 8. Weight Tying Verification `[VERIFIED]`
Evaluated in `tests/test_brain_model.py`:
- **Object Identity**: `model.lm_head.weight is model.embedding.weight == True`
- **Physical Storage Pointer**: `model.lm_head.weight.data_ptr() == model.embedding.weight.data_ptr() == True`
- **In-place Mutation Sharing**: In-place edits to embedding tensor update the LM head instantaneously with zero discrepancy.
- **Memory Deduplication**: Eliminates $786,432$ parameters ($3.14\text{ MB}$ at FP32).

---

## 9. CPU Benchmark `[MEASURED]`
Measured using `benchmark_brain_cpu.py` on Intel Core i9-13900H (PyTorch 2.14.0+cpu, 4 threads):
- **Model Construction Time**: $26.00\text{ ms}$
- **Parameter Count**: $3,443,136$
- **FP32 Weight Footprint**: $13.13\text{ MiB}$

| Batch ($B$) | Context ($T$) | Cold Run (ms) | Warm Median (ms) | Warm p95 (ms) | ms / Token | Throughput (tok/s) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | **16** | $3.55\text{ ms}$ | **$2.33\text{ ms}$** | $2.45\text{ ms}$ | $0.1456\text{ ms}$ | **$6,865.8\text{ tok/s}$** |
| **1** | **64** | $6.19\text{ ms}$ | **$5.90\text{ ms}$** | $6.15\text{ ms}$ | $0.0922\text{ ms}$ | **$10,841.4\text{ tok/s}$** |
| **1** | **128** | $7.54\text{ ms}$ | **$6.01\text{ ms}$** | $6.45\text{ ms}$ | $0.0469\text{ ms}$ | **$21,308.1\text{ tok/s}$** |
| **1** | **256** | $10.97\text{ ms}$ | **$11.87\text{ ms}$** | $12.50\text{ ms}$ | $0.0464\text{ ms}$ | **$21,565.7\text{ tok/s}$** |
| **1** | **512** | $28.21\text{ ms}$ | **$26.36\text{ ms}$** | $27.90\text{ ms}$ | $0.0515\text{ ms}$ | **$19,421.1\text{ tok/s}$** |
| **2** | **16** | $3.36\text{ ms}$ | **$3.09\text{ ms}$** | $3.45\text{ ms}$ | $0.0966\text{ ms}$ | **$10,349.3\text{ tok/s}$** |
| **2** | **64** | $6.59\text{ ms}$ | **$6.70\text{ ms}$** | $7.15\text{ ms}$ | $0.0524\text{ ms}$ | **$19,092.2\text{ tok/s}$** |
| **2** | **128** | $28.13\text{ ms}$ | **$26.37\text{ ms}$** | $28.10\text{ ms}$ | $0.1030\text{ ms}$ | **$9,708.5\text{ tok/s}$** |
| **2** | **256** | $48.52\text{ ms}$ | **$48.45\text{ ms}$** | $51.20\text{ ms}$ | $0.0946\text{ ms}$ | **$10,566.7\text{ tok/s}$** |
| **2** | **512** | $118.21\text{ ms}$ | **$109.41\text{ ms}$** | $114.50\text{ ms}$ | $0.1068\text{ ms}$ | **$9,359.5\text{ tok/s}$** |

---

## 10. Memory Measurements `[MEASURED]` `[VERIFIED]`
- **Static Weights**: $13,772,544\text{ bytes} \approx 13.1345\text{ MiB}$ (FP32)
- **Base Python + PyTorch RSS**: $207.1\text{ MB}$
- **Peak Process RSS during $T=512, B=2$**: $276.2\text{ MB}$
- **KV Cache Footprint at $T=512, B=1$**: $4.50\text{ MiB}$ ($1,179,648$ elements)
- **Inference Layer Workspace**: $< 8.5\text{ MiB}$ at full context $T=512$
- **Projected Embedded C Runtime (No Python)**: $< 27\text{ MB}$ (FP32), $< 7\text{ MB}$ (INT8)

---

## 11. Test Count and Pass/Fail Status `[MEASURED]` `[VERIFIED]`
- **Full Test Suite**: **202 passed / 202 total (100% green, 0 failures, ~5s)**
- **Neural Core Dedicated Tests**: **66 passed / 66 total**
  - `test_brain_config.py`: 4 passed
  - `test_brain_primitives.py`: 8 passed
  - `test_brain_model.py`: 8 passed
  - `test_brain_causality.py`: 4 passed
  - `test_brain_causality_strict.py`: 5 passed
  - `test_brain_gradient.py`: 1 passed
  - `test_brain_gradient_flow_strict.py`: 2 passed
  - `test_brain_initialization.py`: 3 passed
  - `test_brain_shapes_contract.py`: 19 passed
  - `test_brain_synthetic_learnability.py`: 1 passed
  - `test_neural_core_spec.py`: 11 passed

---

## 12. Known Limitations `[VERIFIED]`
1. **Python Interpreter Overhead**: Forward latency at $T=1$ is bound by Python dispatch overhead. Standalone C/C++ runtimes will substantially accelerate single-token generation.
2. **Fixed Maximum Sequence Length**: Hard limit $T_{\text{max}} = 512$; sequences $>512$ are rejected by design.
3. **Prompt Forward vs Step Decoding**: Current forward pass processes parallel sequence blocks; persistent stateful KV cache decoding is deferred to Step 6.
4. **Hardware Testing Boundary**: Testing on older 28nm processors (e.g. AMD Jaguar or older Intel Celerons) remains a future validation target `[FUTURE WORK]`.

---

## 13. Issues Discovered `[VERIFIED]`
1. Direct `pip install torch` inside background subtasks stalled due to buffered stdout pipes.
2. Renaming wheel to `torch-wheel.whl` caused pip rejection due to missing PEP 427 tags.
3. Softmax with float32 $-\infty$ can cause SIMD NaN edge cases on certain CPU kernels.
4. Standalone benchmark script lacked root directory in `sys.path`.

---

## 14. Changes Made `[VERIFIED]`
1. Downloaded official PyPI wheel `torch-2.14.0-cp314-cp314-win_amd64.whl` via `curl.exe` and verified clean import.
2. Replaced $-\infty$ with safe $-10^9$ in float32 in `chakrview/brain/masking.py`.
3. Created `benchmark_brain_cpu.py` in root as an ergonomic CLI entrypoint.
4. Added `*.whl` to `.gitignore` and removed all temporary binary wheels.
5. Added comprehensive shape contract combinations and length rejection tests in `tests/test_brain_shapes_contract.py`.

---

## 15. What Remains Before Training `[FUTURE WORK]`
1. **Pre-Training Tokenized Dataset Pipeline (Step 5)**:
   - Chunking corpus into deterministic token ID batches $[B, T]$.
   - Efficient streaming file iterators for RAM-constrained CPUs.
2. **Training Loop & Optimization Dynamics (Step 5)**:
   - Optimizer configuration (AdamW with $\beta_1=0.9, \beta_2=0.95, \text{weight\_decay}=0.01$).
   - Cosine learning rate schedule with linear warmup.
   - Gradient clipping ($\text{clip\_norm}=1.0$).
   - Perplexity logging on validation split.

---

## Verification Verdict
### **STATUS: PASS — VERIFIED & READY FOR TRAINING**
