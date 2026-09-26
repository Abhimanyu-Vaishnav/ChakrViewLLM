# Project Status: ChakrView

## Project Overview
- **Project**: ChakrView
- **Current phase**: Step 4 — Neural Core Prototype Implementation & Verification
- **Status**: Step 4 Complete & Verified (171/171 tests passing; Zero Failures; Ready for Step 5)

---

## Status Summary

### Implementation Notice
> **IMPORTANT**: Step 4 Neural Core Prototype has been successfully implemented and verified from scratch with zero external model weights, zero pretrained models, and zero wrapper frameworks. The indigenous neural brain package `chakrview/brain/` is completely operational on CPU. Step 1 delivered the initial neural core specification. Step 2 delivered the tokenizer specification and audit. Step 2.2 delivered the minimal BPE prototype. Step 2.3 delivered the initial corpus pipeline and BPE experiment. Step 2.4 froze the tokenizer-to-neural-core interface contract. Step 3 executed empirical tokenizer research, corpus engineering, multi-candidate benchmarking, and vocabulary selection. Step 4 has now implemented the complete neural core prototype `ChakrMicro v0.1` ($3,443,136$ parameters) with exact parameter accounting, strict causal masking verification, RoPE positional encoding, SwiGLU feedforward, tied LM head, gradient flow learnability verification, and CPU forward-pass benchmarking with 171/171 unit tests passing.

### Progress by Module
- `chakrview/brain/`: **Indigenous Neural Core Engine (ChakrMicro v0.1)**
  - `config.py`: `ModelConfig` dataclass with dimension divisibility and parity validation
  - `embeddings.py`: `TokenEmbedding` with parameter matrix $E \in \mathbb{R}^{4096 \times 192}$ and out-of-bounds error handling
  - `normalization.py`: `RMSNorm` (Pre-RMSNorm, bias-free, float32 rsqrt stability)
  - `rotary.py`: `RotaryEmbedding` (pairwise 2D Givens rotation, precomputed frequency caches, sequence bounds check)
  - `masking.py`: `CausalMask` (upper-triangular mask, $-1e9$ additive safe float32)
  - `attention.py`: `MultiHeadAttention` (Simple MHA, bias-free QKV/OutProj, RoPE on Q/K, scaled dot product)
  - `feedforward.py`: `SwiGLU` (bias-free gate, up, down projections with SiLU activation)
  - `block.py`: `TransformerBlock` (Pre-RMSNorm $\to$ MHA $\to$ Residual $\to$ Pre-RMSNorm $\to$ SwiGLU $\to$ Residual)
  - `output.py`: `LMHead` (Tied output head sharing `embedding.weight` memory pointer)
  - `initialization.py`: `initialize_weights` ($\mathcal{N}(0, 0.02)$ for base, depth-scaled $\frac{0.02}{\sqrt{12}}$ for residual projections $W_O, W_{\text{down}}$, ones for RMSNorm)
  - `model.py`: `ChakrMicro` (End-to-end causal language model and programmatic `count_parameters()` reporting)
  - `__init__.py`: Clean public exports of all brain primitives
- `chakrview/config.py`: **Formal Architectural Configuration & Contract Module**
  - `ChakrConfig`: Frozen hyperparameter dataclass with automatic divisibility, parity, and bounds validation
  - `calculate_parameter_breakdown()`: Analytical parameter formulas for all subcomponents (3,443,136 total parameters)
  - `calculate_memory_budget()`: Static weight memory, KV cache scaling, and dynamic activation footprint calculations
  - `calculate_flops_breakdown()`: GEMM and non-GEMM FLOP accounting for sequences $T \in \{128, 256, 512\}$
  - `get_tensor_forward_contracts()`: Bit-exact tensor shape contracts for every stage of the forward pass
- `chakrview/corpus/`: **Dedicated Corpus Engineering Pipeline**
  - `loader.py`: Category document loader and memory-conscious streaming iterator (`CorpusDocument`)
  - `validators.py`: Data quality validator checking empty documents, duplicates, control chars, null bytes, surrogates, and length bounds with automated JSON report generation
  - `cleaner.py`: Clear separation across Raw, Cleaned Training, Tokenizer Benchmark, and Adversarial stages
  - `normalizer.py`: Strict runtime identity pass-through vs optional training normalization policies
  - `splitter.py`: Category-aware deterministic 80/10/10 partitioning using SHA-256 hash scoring with strict disjointness verification
  - `statistics.py`: Character, script (Devanagari, Latin, Digits, Punctuation, Math, Emojis), codepoint, and category profiler
  - `manifest.py`: SHA-256 integrity and metadata manifest generator
- `chakrview/tokenizer/`: **Production Research Engine**
  - `special_tokens.py`: Minimal special-token set (`<BOS>`: 0, `<EOS>`: 1, `<PAD>`: 2) with literal string isolation
  - `bytes.py`: Bijective mapping for 256 fundamental byte primitives (IDs 3..258)
  - `bpe.py`: Deterministic BPE engine with triple-key tie-breaking (`-freq`, `pair[0]`, `pair[1]`)
  - `encoder.py`: Lossless UTF-8 and raw byte encoding
  - `decoder.py`: Bit-exact byte and text reconstruction
  - `tokenizer.py`: Unified `BPETokenizer` class interface
  - `trainer.py`: Streaming and in-memory BPE trainer with grapheme-aware pre-tokenization hooks and deterministic tie-breaking
  - `corpus.py`: Raw corpus validator, order-preserving line deduplication, deterministic category-aware 80/10/10 split, and chunked streaming iterators
  - `metrics.py`: Rigorous accounting for token fertility, compression ratio, latency statistics (mean, median, p95, min, max), and static/downstream memory
  - `serialization.py`: Transparent JSON serialization (`vocab.json`, `merges.json`, `config.json`) with SHA-256 checksum integrity verification
  - `benchmark.py`: Comprehensive benchmarking suite across vocabulary sizes, numerics, grapheme variants, latency, and memory
  - `interface.py`: Token ID bounds validator, sequence truncation/chunking runtime policies, batch tensor preparation, neural handoff simulation (`[seq_len, 192]`), and static memory accounting
- `data/`: **Structured Dataset Hierarchy Verified**
  - `raw/`: 19 validated files across 8 categories (Hindi, English, Hinglish, Sanskrit, Mixed, Code, Mathematics, Numbers) with `manifest.json`
  - `processed/train/`: Deterministically partitioned training split (477 lines, 48,119 bytes)
  - `validation/`: Unseen validation split (68 lines, 7,621 bytes) + `corpus_validation_report.json`
  - `processed/test/`: Unseen test split (55 lines, 4,468 bytes)
  - `manifests/`: `corpus_manifest.json` with file hashes
  - `statistics/`: `corpus_statistics.json` and `step3_tokenizer_benchmark.json`
  - `experiments/`: Trained candidate artifacts for `vocab_2048/`, `vocab_4096/`, `vocab_8192/`, `vocab_16384/`
- `tests/`: **171/171 Tests Passing** across 19 test modules (100% green, 0 failures)
  - 151 Tokenizer and Corpus pipeline tests
  - 20 Neural Core prototype tests (Config, Primitives, Model, Causality, Gradients)
- `docs/`: **Comprehensive Step 4 Documentation & Empirical Baselines Ratified**
  - `docs/STEP_04_NEURAL_CORE_IMPLEMENTATION_PLAN.md` (Formal mathematical and engineering plan)
  - `docs/STEP_04_PARAMETER_ACCOUNTING.md` (Analytical vs code parameter reconciliation: 3,443,136 parameters)
  - `docs/STEP_04_CPU_BASELINE.md` (Empirical CPU forward-pass latency and throughput benchmarks)
  - `docs/STEP_04_NEURAL_CORE_REPORT.md` (Complete Step 4 Engineering Report)
  - `docs/STEP_04_NEURAL_CORE_SPEC.md` (Formal architectural specification, parameter budget, memory, FLOPs, tensor contracts)
  - `docs/NEURAL_CORE_DECISIONS.md` (Architectural Decision Records ADR 17–ADR 23)
  - `docs/ARCHITECTURE_DECISIONS.md` (Complete repository ADR index updated with Step 4 frozen decisions)

### Step 4 Ratified Architecture & Prototype: Chakr-Micro v0.1
- **Architecture**: Decoder-only causal autoregressive transformer
- **Layers ($N$)**: 6 stacked transformer blocks
- **Model Dimension ($d_{\text{model}}$)**: 192
- **Attention Query Heads ($H$)**: 6 ($d_{\text{head}} = 32$)
- **Attention KV Heads ($H_{kv}$)**: 6 (Simple MHA)
- **SwiGLU Intermediate Dimension ($d_{\text{ff}}$)**: 512 (exact $\frac{8}{3} d_{\text{model}}$ integer, $2^9$, 32 cache lines)
- **Maximum Context Window ($T_{\text{max}}$)**: 512 tokens
- **Vocabulary Size ($V$)**: 4096 (ratified from Step 3 empirical benchmark)
- **Normalization**: Pre-RMSNorm with $\epsilon = 10^{-5}$ and scale $\boldsymbol{\gamma}$ (bias-free)
- **Positional Mechanism**: RoPE ($\Theta = 10000.0, d_{\text{rot}} = 32$) applied to $Q$ and $K$
- **Weight Tying**: Enabled and hardware-verified ($W_{\text{out}} = E^T$, data_ptr identical)
- **Projections Bias**: Strictly bias-free ($b = 0$) across all linear projections
- **Total Parameters**: **$3,443,136$** ($786,432$ embedding, $2,656,512$ transformer layers, $192$ final norm)
- **Static Weight Memory**: FP32: $13.13\text{ MiB}$ ($13.77\text{ MB}$), FP16: $6.57\text{ MiB}$ ($6.89\text{ MB}$), INT8: $3.28\text{ MiB}$ ($3.44\text{ MB}$), INT4: $1.64\text{ MiB}$ ($1.72\text{ MB}$)
- **Causality Verification**: Verified (zero future-token leakage to prefix tokens)
- **Gradient Flow**: Verified (15-step learnability smoke test with loss reduction)
- **CPU Forward Benchmark**: $T=1$: $1.31\text{ ms}$, $T=512$: $21.72\text{ ms}$ ($23,569.6\text{ tok/s}$) on Intel i9-13900H (PyTorch 2.14.0+cpu)

### Next Phase
- Step 5: Pre-Training Infrastructure & Data Loader Pipeline (Awaiting User Instructions).
