# Project Status: ChakrView

## Project Overview
- **Project**: ChakrView
- **Current phase**: Step 2.4 — Tokenizer-to-Neural-Core Interface Contract
- **Status**: Interface Contract Frozen & Verified (107/107 tests passing; Awaiting Step 3 Instructions)

---

## Status Summary

### Implementation Notice
> **IMPORTANT**: No neural core architecture code or Transformer layers have been written, and no neural weights, pretrained tokenizers, or pretrained components (such as Llama, Qwen, Mistral, Gemma, GPT, etc.) have been implemented or downloaded. ChakrView is being developed strictly incrementally from the ground up as an indigenous AI research initiative. Step 1 delivered the neural core specification. Step 2 delivered the tokenizer specification and audit. Step 2.2 delivered the minimal BPE prototype. Step 2.3 delivered the corpus engineering pipeline and BPE empirical benchmark. Step 2.4 froze the complete tokenizer-to-neural-core interface contract, batching format, and static memory accounting with 107/107 unit tests passing.

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

### Step 2.4 Interface Contract Summary
- **Provisional Vocabulary**: $V = 4096$ experimentally selected from Step 2.3 benchmark corpus (separable and replaceable by $V = 8192$ without altering neural layers).
- **Special Token Contract**: $\langle\text{BOS}\rangle=0, \langle\text{EOS}\rangle=1, \langle\text{PAD}\rangle=2$. Raw bytes: $3 \dots 258$. Merges: $259 \dots 4095$. Valid range strictly $[0, 4095]$.
- **Context Contract**: $\text{MAX\_CONTEXT} = 512$ is a neural model context window, not a tokenizer limit. Tokenizer supports arbitrary sequence length with explicit truncation/chunking policies.
- **Batch Contract**: `input_ids` shape $[B, T]$, `attention_mask` shape $[B, T]$ ($1$ for valid token, $0$ for padding).
- **Embedding & Weight Tying**: $E \in \mathbb{R}^{4096 \times 192}$, $W_{\text{out}} = E^T \in \mathbb{R}^{192 \times 4096}$ adds 0 additional unique parameters.
- **Static Parameter Storage**: 786,432 parameters ($\text{FP32} = 3.00\text{ MB}, \text{FP16} = 1.50\text{ MB}, \text{INT8} = 0.75\text{ MB}, \text{INT4} = 0.375\text{ MB}$). Excludes dynamic activations, allocator overhead, temporary tensors, and KV cache.
- **Neural Handoff**: Token IDs $\to [B, T, 192]$ validated by simulation unit test.

### Next Phase
- Step 3: Neural Core Implementation & Training Pipeline.




