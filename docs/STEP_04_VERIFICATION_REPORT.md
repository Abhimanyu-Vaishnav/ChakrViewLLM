# ChakrView Step 4: Neural Core Verification & Stabilization Report

**Document Version**: 1.0.0  
**Phase**: Step 4 — Neural Core Verification & Quality Gate  
**Model Target**: Chakr-Micro v0.1 ($3,443,136$ Unique Parameters)  
**Status**: COMPLETE, VERIFIED & RATIFIED  

---

## A. Environment Verification

| Item | Expected Specification | Measured / Detected Value | Status |
| :--- | :--- | :--- | :---: |
| **Python Version** | $\ge 3.10$ | `Python 3.14.7` (MSC v.1944 64-bit AMD64) | `[PASS] [MEASURED]` |
| **PyTorch Runtime** | Official CPU build | `PyTorch 2.14.0+cpu` | `[PASS] [MEASURED]` |
| **BLAS Engine** | Intel MKL / AVX2 | Intel MKL 2026.1, MKL-DNN v3.12.0, AVX2 enabled | `[PASS] [MEASURED]` |
| **CUDA Dependency** | Zero CUDA dependency | `torch.cuda.is_available() == False` | `[PASS] [MEASURED]` |
| **NumPy Version** | Standard scientific backend | `NumPy 2.5.3` | `[PASS] [MEASURED]` |
| **PyTest Engine** | Test framework | `pytest 9.1.1` | `[PASS] [MEASURED]` |
| **Hardware** | Development machine | Intel i9-13900H (14C/20T), 32 GB DDR5, Windows 11 | `[PASS] [MEASURED]` |

---

## B. Architecture Verification

Every architectural decision was audited against `docs/STEP_01_NEURAL_CORE_SPEC.md`, `docs/ARCHITECTURE_DECISIONS.md`, and `docs/STEP_04_NEURAL_CORE_IMPLEMENTATION_PLAN.md`:

| Architectural Feature | Step 1 Specification | Implementation Module | Verification Finding | Status |
| :--- | :--- | :--- | :--- | :---: |
| **Paradigm** | Decoder-only Causal Transformer | `chakrview/brain/model.py` | Strict autoregressive causal stack | `[PASS]` |
| **Layers ($N$)** | $6$ Stacked Blocks | `chakrview/brain/model.py` | `nn.ModuleList` of 6 `TransformerBlock`s | `[PASS]` |
| **Model Dim ($d_{\text{model}}$)** | $192$ | `chakrview/brain/config.py` | $192$ ($192 \equiv 0 \pmod{64}$) | `[PASS]` |
| **Query Heads ($H$)** | $6$ | `chakrview/brain/attention.py` | $6$ parallel heads ($d_{\text{head}} = 32$) | `[PASS]` |
| **KV Heads ($H_{kv}$)** | $6$ (Simple MHA) | `chakrview/brain/attention.py` | $H_{kv} = H = 6$; No GQA in v0.1 | `[PASS]` |
| **Feed-Forward** | SwiGLU ($d_{\text{ff}} = 512$) | `chakrview/brain/feedforward.py`| $(\text{SiLU}(xW_{\text{gate}}) \odot xW_{\text{up}})W_{\text{down}}$ | `[PASS]` |
| **Normalization** | Pre-RMSNorm | `chakrview/brain/normalization.py`| Bias-free, $\epsilon = 10^{-5}$, scale $\boldsymbol{\gamma}$ | `[PASS]` |
| **Positional Encoding**| RoPE on $Q$ and $K$ | `chakrview/brain/rotary.py` | $\Theta = 10000.0$; pairwise 2D Givens | `[PASS]` |
| **Value State RoPE** | $V$ Unrotated | `chakrview/brain/attention.py` | $V$ strictly bypasses RoPE | `[PASS]` |
| **Linear Projections** | Bias-free ($b=0$) | `chakrview/brain/attention.py`, `feedforward.py` | `bias=False` everywhere | `[PASS]` |
| **Weight Tying** | $W_{\text{out}} \equiv E^T$ | `chakrview/brain/output.py` | Shares `TokenEmbedding.weight` storage | `[PASS]` |
| **Context Length** | $T_{\text{max}} = 512$ | `chakrview/brain/config.py` | Enforced; $T>512$ rejected with `ValueError` | `[PASS]` |
| **Vocabulary** | $V = 4096$ | `chakrview/brain/config.py` | Ratified from Step 3 BPE experiment | `[PASS]` |
| **Pretrained Weights** | Zero external weights | Whole repository | $100\%$ indigenous parameters | `[PASS]` |

---

## C. Tensor-Shape Verification

Audit of exact tensor dimensions across the end-to-end forward pipeline ($B=\text{batch}, T=\text{seq\_len}$):

```
input_ids:                   [B, T]                 int64, values in [0, 4095]
        │
TokenEmbedding:              [B, T, 192]            float32
        │
┌───────┴─────────────────────────────────────────────────────────────────┐
│ TransformerBlock (Repeated N = 6 Times)                                 │
│                                                                         │
│   Residual Input (x):       [B, T, 192]           float32               │
│   Pre-Attention RMSNorm:    [B, T, 192]           float32               │
│   Q, K, V Projections:      [B, T, 192] each      float32               │
│   Reshape Heads:            [B, 6, T, 32] each    float32               │
│   RoPE on Q and K:          [B, 6, T, 32] each    float32               │
│   Scaled Dot-Product:       [B, 6, T, T]          float32               │
│   Causal Additive Mask:     [1, 1, T, T]          float32 (-1e9 upper)  │
│   Softmax Probabilities:    [B, 6, T, T]          float32               │
│   Weighted Context V:       [B, 6, T, 32]         float32               │
│   Merge Heads:              [B, T, 192]           float32               │
│   Out Projection:           [B, T, 192]           float32               │
│   Residual Addition 1:      [B, T, 192]           x = x + Attn(Norm1(x))│
│                                                                         │
│   Pre-FFN RMSNorm:          [B, T, 192]           float32               │
│   W_gate, W_up:             [B, T, 512] each      float32               │
│   SwiGLU Gating:            [B, T, 512]           SiLU(Gate) * Up       │
│   W_down Projection:        [B, T, 192]           float32               │
│   Residual Addition 2:      [B, T, 192]           x = x + FFN(Norm2(x)) │
└───────┬─────────────────────────────────────────────────────────────────┘
        │
Final RMSNorm:               [B, T, 192]            float32
        │
Tied LM Head:                [B, T, 4096]           float32 (x @ E^T)
```

---

## D. Parameter Accounting Audit

The analytical mathematical derivations match the code-measured PyTorch parameter counts bit-for-bit (verified via `scripts/verify_parameter_accounting.py`):

| Component | Analytical Derivation | Mathematical Expected | Code Measured | Status |
| :--- | :--- | :---: | :---: | :---: |
| **Token Embedding ($E$)** | $V \times d_{\text{model}} = 4096 \times 192$ | $786,432$ | $786,432$ | `[PASS] [MEASURED]` |
| **Attention $W_Q$ (6 layers)** | $6 \times (192 \times 192)$ | $221,184$ | $221,184$ | `[PASS] [MEASURED]` |
| **Attention $W_K$ (6 layers)** | $6 \times (192 \times 192)$ | $221,184$ | $221,184$ | `[PASS] [MEASURED]` |
| **Attention $W_V$ (6 layers)** | $6 \times (192 \times 192)$ | $221,184$ | $221,184$ | `[PASS] [MEASURED]` |
| **Attention $W_O$ (6 layers)** | $6 \times (192 \times 192)$ | $221,184$ | $221,184$ | `[PASS] [MEASURED]` |
| **Attention Total (6 layers)** | $6 \times 147,456$ | **$884,736$** | **$884,736$** | `[PASS] [MEASURED]` |
| **SwiGLU $W_{\text{gate}}$ (6 layers)** | $6 \times (192 \times 512)$ | $589,824$ | $589,824$ | `[PASS] [MEASURED]` |
| **SwiGLU $W_{\text{up}}$ (6 layers)** | $6 \times (192 \times 512)$ | $589,824$ | $589,824$ | `[PASS] [MEASURED]` |
| **SwiGLU $W_{\text{down}}$ (6 layers)** | $6 \times (512 \times 192)$ | $589,824$ | $589,824$ | `[PASS] [MEASURED]` |
| **SwiGLU Total (6 layers)** | $6 \times 294,912$ | **$1,769,472$** | **$1,769,472$** | `[PASS] [MEASURED]` |
| **Layer RMSNorm Scales (6 layers)** | $6 \times 2 \times 192$ | $2,304$ | $2,304$ | `[PASS] [MEASURED]` |
| **Final RMSNorm Scale** | $1 \times 192$ | $192$ | $192$ | `[PASS] [MEASURED]` |
| **Total Normalization Parameters**| $2304 + 192$ | **$2,496$** | **$2,496$** | `[PASS] [MEASURED]` |
| **Tied LM Head Unique Params** | Shared with $E$ ($W_{\text{out}} = E^T$) | **$0$** | **$0$** | `[PASS] [MEASURED]` |
| **TOTAL UNIQUE PARAMETERS** | $786,432 + 884,736 + 1,769,472 + 2,496$ | **$3,443,136$** | **$3,443,136$** | `[PASS] [MEASURED]` |
| **TOTAL TRAINABLE PARAMETERS**| All unique parameters requiring grad | **$3,443,136$** | **$3,443,136$** | `[PASS] [MEASURED]` |

---

## E. Weight Tying Verification

- **Storage Sharing Test**:
  ```python
  assert model.lm_head.weight is model.embedding.weight
  assert model.lm_head.weight.data_ptr() == model.embedding.weight.data_ptr()
  ```
  **Result**: `True` (`data_ptr` identical; sharing verified).
- **Double-Counting Avoidance**:
  - `len(list(model.parameters()))`: $56$ parameter references.
  - `len(set(model.parameters()))`: $56$ unique parameter objects.
  - The tied head introduces $0$ additional parameter objects; accounting programmatically evaluates unique tensors only.

---

## F. Causality Verification

- **Experiment 1 (Multi-token perturbation)**:
  - Sequence A: `[t0, t1, t2, t3, t4, t5, t6, t7]`
  - Sequence B: `[t0, t1, t2, t3, ALTERED, ALTERED, ALTERED, ALTERED]`
  - Max prefix logit difference ($t \in [0, 3]$): $\|\mathbf{z}_A[:4] - \mathbf{z}_B[:4]\|_{\infty} = 0.000000$ (`< 1e-6`).
  - Suffix difference ($t \ge 4$): Diverges ($> 1e-1$).
- **Experiment 2 (Single-token future perturbation ABCD)**:
  - Sequence 1: `[A, B, C, D]`
  - Sequence 2: `[A, B, C, D']`
  - Difference at positions $0, 1, 2$ ($A, B, C$): $0.000000$ (`< 1e-6`).
  - Difference at position $3$ ($D$ vs $D'$): $> 0.1$.
- **Conclusion**: Causal masking, RoPE frequency mapping, Pre-RMSNorm statistics, and residual connections strictly preserve temporal causality with zero backward information leakage.

---

## G. Gradient Verification

- **Synthetic Learnability Smoke Test**:
  - Evaluated on sequence `ABABAB...` with next-token prediction cross-entropy loss.
  - Forward Loss: $8.4013$ (Finite, non-zero, $0$ NaN, $0$ Inf).
  - Backward Pass: `loss.backward()` completed cleanly.
  - Gradients: Evaluated across all $56$ parameter tensors. Every parameter tensor received a non-zero, finite gradient tensor (`assert not torch.isnan(p.grad).any()`).
  - Optimizer Step: 15 SGD steps reduced loss monotonically to $7.2140$.

---

## H. Initialization Verification

- **Determinism**: Identical random seeds produce bit-exact identical initial model parameters (`torch.equal(p1, p2) == True`).
- **Statistical Distribution**:
  - Base projections ($W_Q, W_K, W_V, W_{\text{gate}}, W_{\text{up}}$) and Embedding ($E$): $\mu \approx 0.0, \sigma \approx 0.02$.
  - Residual projections ($W_O, W_{\text{down}}$): Depth-scaled $\sigma_{\text{residual}} = \frac{0.02}{\sqrt{2N}} = \frac{0.02}{\sqrt{12}} \approx 0.00577$. Measured $\sigma \approx 0.0057$.
  - Normalization scales ($\boldsymbol{\gamma}$): Initialized to exactly $\mathbf{1.0}$.
- **Weight Tying Invariance**: Re-running `initialize_weights(model, config)` preserves physical weight sharing (`model.lm_head.weight.data_ptr() == model.embedding.weight.data_ptr()`).

---

## I. CPU Baseline Performance (Measured)

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

---

## J. Memory Observations

- **Static Weights (FP32)**: $13.135\text{ MiB}$ ($13.773\text{ MB}$).
- **Static Weights (INT8 Projection)**: $3.284\text{ MiB}$ (`[HYPOTHESIS]`, unquantized in v0.1).
- **Process Working Set (Measured)**:
  - Base Python + PyTorch runtime RSS: $206.9\text{ MB}$.
  - Post-initialization RSS: $232.3\text{ MB}$ ($\Delta \approx 25.4\text{ MB}$).
  - Peak inference RSS ($T=512$): $256.2\text{ MB}$ ($\Delta \approx 23.9\text{ MB}$).
- **Boundary Clarification**: Total runtime process memory reflects PyTorch runtime libraries, memory pools, and temporary activation buffers, and must not be conflated with the static $13.1\text{ MiB}$ weight tensor footprint.

---

## K. Test Results Summary

Full repository test suite execution (`pytest -v`):
- **178 Total Tests Passed** (0 failures, 0 skipped, 0 warnings in 3.07s).
- **151 Tokenizer & Corpus Tests**:
  - `test_tokenizer_bytes.py`: 7 tests
  - `test_tokenizer_special_tokens.py`: 4 tests
  - `test_tokenizer_bpe_engine.py`: 7 tests
  - `test_tokenizer_utf8_lossless.py`: 33 tests
  - `test_tokenizer_determinism.py`: 3 tests
  - `test_tokenizer_serialization.py`: 3 tests
  - `test_tokenizer_interface.py`: 10 tests
  - `test_corpus_*.py`: 84 tests
- **27 Neural Core Tests**:
  - `test_brain_config.py`: 4 tests
  - `test_brain_primitives.py`: 8 tests
  - `test_brain_model.py`: 8 tests
  - `test_brain_causality.py`: 3 tests
  - `test_brain_gradient.py`: 1 test
  - `test_brain_initialization.py`: 3 tests

---

## L. Known Limitations & Unfrozen Boundaries

1. **Python Dispatch Overhead**: Small-batch CPU execution contains Python interpreter overhead. Future compiled C++ / GGML inference will reduce per-token latency further.
2. **Legacy 28nm Hardware Hypothesis**: While the INT8 footprint ($3.28\text{ MiB}$) fits in L3 caches of modern CPUs (i9-13900H has 36 MB L3), execution on older 28nm CPUs lacking large L3 cache remains an unverified hypothesis (`[HYPOTHESIS]`) pending physical testing on such devices.
3. **Unfrozen Points for Step 5+**:
   - Production optimizer selection (AdamW vs Lion vs Sophia).
   - Pre-training learning rate schedules and warmup curves.
   - Post-training INT8 / INT4 quantization kernels.
   - Streaming token iterators for production pre-training corpus.

---

## M. Step 4 Exit Gate Evaluation

| Verification Gate | Requirement | Status |
| :--- | :--- | :---: |
| PyTorch Runtime | PyTorch 2.14.0+cpu functional on CPU | `[PASS]` |
| Full Test Suite | 178/178 tests passing (zero regressions) | `[PASS]` |
| Forward Pass | Functional across $T \in \{1, 8, 16, 32, 64, 128, 256, 512\}$ | `[PASS]` |
| Backward Pass | Gradients flow cleanly to all 56 parameter tensors | `[PASS]` |
| Causal Masking | Strict future-token isolation verified | `[PASS]` |
| Tensor Contracts | All intermediate dimensions conform to spec | `[PASS]` |
| Parameter Accounting | Mathematical expected equals code measured ($3,443,136$) | `[PASS]` |
| Weight Tying | `LMHead.weight is TokenEmbedding.weight` verified | `[PASS]` |
| Initialization | Deterministic, scaled residuals, finite values verified | `[PASS]` |
| CPU Inference | Baseline latency and throughput measured across 8 lengths | `[PASS]` |
| Indigenous Mandate | No pretrained weights, no HF wrappers, 100% indigenous | `[PASS]` |
| Documentation | Complete verification report and baseline docs recorded | `[PASS]` |
| Git Working Tree | Clean working tree; verified commit created | `[PASS]` |

---
*End of Verification Report — Step 4 Frozen & Ratified*
