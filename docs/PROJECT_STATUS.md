# Project Status: ChakrView

## Project Overview
- **Project**: ChakrView
- **Current phase**: Step 2.3 — Corpus Engineering & BPE Vocabulary Experiment
- **Status**: Completed & Verified (97/97 tests passing; V=2048, 4096, 8192, 16384 benchmarked; Awaiting Step 3 Instructions)

---

## Status Summary

### Implementation Notice
> **IMPORTANT**: No neural core architecture code or Transformer layers have been written, and no neural weights, pretrained tokenizers, or pretrained components (such as Llama, Qwen, Mistral, Gemma, GPT, etc.) have been implemented or downloaded. ChakrView is being developed strictly incrementally from the ground up as an indigenous AI research initiative. Step 1 delivered the neural core specification. Step 2 delivered the tokenizer specification and audit. Step 2.2 delivered the minimal BPE prototype. Step 2.3 delivered the corpus engineering pipeline, deterministic BPE trainer, and empirical benchmark across candidate vocabulary sizes ($V \in \{2048, 4096, 8192, 16384\}$) using only standard library and existing pytest.

### Progress by Module
- `chakrview/tokenizer/`: **Implemented & Benchmarked**
  - `special_tokens.py`: Minimal special-token set (`<BOS>`: 0, `<EOS>`: 1, `<PAD>`: 2) with literal string isolation
  - `bytes.py`: Bijective mapping for 256 fundamental byte primitives (IDs 3..258)
  - `bpe.py`: Deterministic BPE engine with triple-key tie-breaking (`-freq`, `pair[0]`, `pair[1]`)
  - `encoder.py`: Lossless UTF-8 and raw byte encoding
  - `decoder.py`: Bit-exact byte and text reconstruction
  - `tokenizer.py`: Unified `BPETokenizer` prototype interface
  - `trainer.py`: Deterministic BPE trainer and artifact serialization (`merges.json`, `vocab.json`)
  - `benchmark.py`: Empirical evaluator and 3-way numeric strategy runner
- `chakrview/tokenizer_corpus/`: **Implemented (Corpus Pipeline)**
  - `loader.py`: Category file loader producing immutable `CorpusItem` records
  - `validator.py`: UTF-8 integrity and non-emptiness verification
  - `normalization.py`: Training normalization policy explicitly distinguished from lossless runtime encoding
  - `dedup.py`: Duplicate detection preserving appearance order
  - `split.py`: Deterministic stratified 80/20 train/validation partitioning
  - `statistics.py`: Comprehensive character, script, byte, and category metrics
- `data/tokenizer_corpus/`: Controlled 8-category research corpus (490 docs, 42,193 UTF-8 bytes, 100% synthetic/manually authored)
- `data/tokenizer_experiments/`: Candidate vocabulary artifacts (`v2048/`, `v4096/`, `v8192/`, `v16384/`)
- `tests/`: Active test suite (97 tests passing across 9 test modules)
- `docs/`: Active documentation initialized & audited
  - `docs/PROJECT_STATUS.md`
  - `docs/STEP_01_NEURAL_CORE_SPEC.md` (Step 1 Core Specification)
  - `docs/ARCHITECTURE_DECISIONS.md` (Step 1 Architecture Decision Records)
  - `docs/STEP_02_TOKENIZER_SPEC.md` (Step 2 Tokenizer Specification — Audited)
  - `docs/TOKENIZER_DECISIONS.md` (Step 2 Tokenizer Architecture Decision Records — Audited)
  - `docs/STEP_02_2_TOKENIZER_PROTOTYPE.md` (Step 2.2 Research Prototype Report)
  - `docs/STEP_02_3_CORPUS_AND_BPE_EXPERIMENT.md` (Step 2.3 Experiment Report)
  - `docs/TOKENIZER_BENCHMARK_RESULTS.md` (Step 2.3 Consolidated Benchmark Results)

### Step 2.3 Empirical Findings
- **Reconstruction Standard**: 100% bit-exact lossless round-trip verified for all vocabulary candidates ($V = 2048, 4096, 8192, 16384$) across the entire validation corpus.
- **Pareto Optimal Candidate**: $V = 4096$ achieves $2.43\text{ tok/word}$ for Hindi, $1.77$ for Hinglish, and $2.78$ for English on this benchmark corpus, utilizing $22.8\%$ of Chakr-Micro parameter capacity.
- **Diminishing Returns on $V \ge 8192$**: Expanding to $V = 16384$ yielded a negligible 3-token ($0.1\%$) validation improvement while inflating memory to $2.23\text{ MB}$ and doubling latency.
- **Numeric Strategy Insight**: Candidate B (two-digit chunks `\d{1,2}`) avoids monolithic date memorization of normal BPE while cutting sequence expansion in half compared to individual digits (`\d`).

### Next Phase
- Awaiting Step 3 instructions from research leadership.



