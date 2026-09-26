# Project Status: ChakrView

## Project Overview
- **Project**: ChakrView
- **Current phase**: Step 3.1 — Neural Core Architecture & Mathematical Specification
- **Status**: Step 3.1 Specification Completed (107/107 tests passing; Awaiting Step 3.2 Instructions)

---

## Status Summary

### Implementation Notice
> **IMPORTANT**: No neural core architecture code or Transformer layers have been written, and no neural weights, pretrained tokenizers, or pretrained components (such as Llama, Qwen, Mistral, Gemma, GPT, etc.) have been implemented or downloaded. ChakrView is being developed strictly incrementally from the ground up as an indigenous AI research initiative. Step 1 delivered the initial neural core specification. Step 2 delivered the tokenizer specification and audit. Step 2.2 delivered the minimal BPE prototype. Step 2.3 delivered the corpus engineering pipeline and BPE empirical benchmark. Step 2.4 froze the complete tokenizer-to-neural-core interface contract. Step 3.1 has now completed the rigorous mathematical and architectural specification of Chakr-Micro v0.1 with exact parameter, memory, FLOPs, and KV-cache accounting with 107/107 unit tests passing.

### Progress by Module
- `chakrview/tokenizer/`: **Interface Frozen & Tested**
  - `special_tokens.py`: Minimal special-token set (`<BOS>`: 0, `<EOS>`: 1, `<PAD>`: 2) with literal string isolation
  - `bytes.py`: Bijective mapping for 256 fundamental byte primitives (IDs 3..258)
  - `bpe.py`: Deterministic BPE engine with triple-key tie-breaking (`-freq`, `pair[0]`, `pair[1]`)
  - `encoder.py`: Lossless UTF-8 and raw byte encoding
  - `decoder.py`: Bit-exact byte and text reconstruction
  - `tokenizer.py`: Unified `BPETokenizer` prototype interface
  - `trainer.py`: Deterministic BPE trainer and artifact serialization (`merges.json`, `vocab.json`)
  - `benchmark.py`: Empirical evaluator and 3-way numeric strategy runner
  - `interface.py`: Token ID bounds validator, sequence truncation/chunking runtime policies, batch tensor preparation, neural handoff simulation (`[seq_len, 192]`), and static memory accounting
- `chakrview/tokenizer_corpus/`: **Implemented (Corpus Pipeline)**
  - `loader.py`: Category file loader producing immutable `CorpusItem` records
  - `validator.py`: UTF-8 integrity and non-emptiness verification
  - `normalization.py`: Training normalization policy explicitly distinguished from lossless runtime encoding
  - `dedup.py`: Duplicate detection preserving appearance order
  - `split.py`: Deterministic stratified 80/20 train/validation partitioning
  - `statistics.py`: Comprehensive character, script, byte, and category metrics
- `data/tokenizer_corpus/`: Controlled 8-category research corpus (490 docs, 42,193 UTF-8 bytes, 100% synthetic/manually authored)
- `data/tokenizer_experiments/`: Candidate vocabulary artifacts (`v2048/`, `v4096/`, `v8192/`, `v16384/`)
- `tests/`: Active test suite (107 tests passing across 10 test modules)
- `docs/`: Active documentation initialized & audited
  - `docs/PROJECT_STATUS.md`
  - `docs/STEP_01_NEURAL_CORE_SPEC.md` (Step 1 Core Specification)
  - `docs/ARCHITECTURE_DECISIONS.md` (Step 1 & 2 Architecture Decision Records, including ADR 10)
  - `docs/STEP_02_TOKENIZER_SPEC.md` (Step 2 Tokenizer Specification — Audited)
  - `docs/TOKENIZER_DECISIONS.md` (Step 2 Tokenizer Architecture Decision Records — Audited)
  - `docs/STEP_02_2_TOKENIZER_PROTOTYPE.md` (Step 2.2 Research Prototype Report)
  - `docs/STEP_02_3_CORPUS_AND_BPE_EXPERIMENT.md` (Step 2.3 Experiment Report)
  - `docs/TOKENIZER_BENCHMARK_RESULTS.md` (Step 2.3 Consolidated Benchmark Results)
  - `docs/STEP_02_4_TOKENIZER_INTERFACE.md` (Step 2.4 Tokenizer Interface Contract)
  - `docs/STEP_03_1_NEURAL_CORE_SPEC.md` (Step 3.1 Neural Core Mathematical Specification)
  - `docs/NEURAL_CORE_DECISIONS.md` (Step 3.1 Neural Core Decision Records ADR 11–ADR 15)

### Step 3.1 Specification Summary (Chakr-Micro v0.1)
- **Architecture**: Decoder-only autoregressive causal language model
- **Frozen Hyperparameters**: $L = 6, d_{\text{model}} = 192, H = 6, H_{kv} = 6, d_{\text{head}} = 32, d_{\text{ff}} = 512, T_{\text{max}} = 512, V = 4096$
- **Total Parameters**: $3,443,136$ exact ($786,432$ embeddings + $2,656,512$ transformer blocks + $192$ final norm + $0$ tied head)
- **Static Parameter Memory**: $\text{FP32} = 13.13\text{ MiB}, \text{FP16} = 6.57\text{ MiB}, \text{INT8} = 3.28\text{ MiB}, \text{INT4} = 1.64\text{ MiB}$ (strictly excludes runtime buffers)
- **KV-Cache Footprint ($B=1$)**: $T=128 \to 0.56\text{ MB (FP16)}, T=256 \to 1.13\text{ MB (FP16)}, T=512 \to 2.25\text{ MB (FP16)}$
- **Forward Pass FLOPs**: $\mathbf{FLOPs}(T) = 6,912,768 \cdot T + 4,788 \cdot T^2$ ($4.794\text{ GFLOPs}$ at $T=512$, $9.36\text{ MFLOPs/token}$)
- **Invariants Verified**: $d_{\text{head}} = 192 / 6 = 32$ exact integer; $d_{\text{ff}} = \frac{8}{3} \times 192 = 512$ exact integer; weight tying $W_{\text{out}} = E^T$; Pre-RMSNorm $\epsilon=10^{-5}$; RoPE $\Theta=10000.0$; bias-free linear projections.

### Next Phase
- Step 3.2: Neural Core Implementation & Diagnostic Learnability Tests (Awaiting User Instructions).





