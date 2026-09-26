# Project Status: ChakrView

## Project Overview
- **Project**: ChakrView
- **Current phase**: Step 3 — Tokenizer Corpus, Training, Benchmarking & Empirical Selection
- **Status**: Step 3 Complete & Ratified (133/133 tests passing; Zero Failures; Ready for Review)

---

## Status Summary

### Implementation Notice
> **IMPORTANT**: No neural core architecture code, transformer layers, or training loops have been implemented, and no pretrained weights or external tokenizer frameworks have been downloaded or installed. ChakrView is being developed strictly incrementally from first principles as an indigenous AI research initiative. Step 1 delivered the initial neural core specification. Step 2 delivered the tokenizer specification and audit. Step 2.2 delivered the minimal BPE prototype. Step 2.3 delivered the initial corpus pipeline and BPE experiment. Step 2.4 froze the tokenizer-to-neural-core interface contract. Step 3 has now executed the complete empirical tokenizer research, training, benchmarking, and empirical selection phase with 133/133 unit tests passing.

### Progress by Module
- `chakrview/tokenizer/`: **Production Research Engine Implemented & Verified**
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
  - `processed/train/`: Deterministically partitioned training split (476 lines, 48,119 bytes)
  - `validation/`: Unseen validation split (72 lines, 7,671 bytes)
  - `processed/test/`: Unseen test split (53 lines, 4,292 bytes)
- `tests/`: **133/133 Tests Passing** across 12 test modules (100% green, 0 failures)
- `docs/`: **Comprehensive Step 3 Documentation Ratified**
  - `docs/STEP_03_TOKENIZER_TRAINING.md` (Training specification, memory-conscious engine, and pipeline)
  - `docs/STEP_03_BENCHMARK_RESULTS.md` (Complete empirical measurement tables across all candidates and metrics)
  - `docs/TOKENIZER_FINAL_DECISION.md` (Final ratified tokenizer architecture decision and trade-off analysis)
  - `docs/ARCHITECTURE_DECISIONS.md` (Architectural Decision Records)

### Step 3 Empirical Benchmark & Final Selection
- **Selected Vocabulary Size:** $V = 4096$ is ratified by measured evidence. Balances Hindi ($2.17\text{ tok/word}$) and English ($2.49\text{ tok/word}$) efficiency, consumes 786,432 parameters ($22.84\%$ of the 3.44M model budget), and requires only $902.4\text{ KB}$ static RAM (fitting inside CPU L2/L3 cache).
- **Selected Numeric Strategy:** Candidate C (Normal BPE) is selected as general-purpose default ($36$ tokens across test battery vs $97$ tokens for Candidate A, avoiding a $2.69\times$ context sequence expansion). Candidate A is retained as a modular adapter for dedicated arithmetic reasoning.
- **Selected Pre-tokenization Strategy:** Variant A (Raw Byte BPE) is selected over Variant B (Grapheme-aware BPE), avoiding a $+22.6\%$ sequence length expansion and $2\times$ regex CPU latency overhead.
- **Invariants Verified:** 100% lossless round-trip on Indic conjuncts, Sanskrit half-forms, ZWJ/ZWNJ, emojis, and arbitrary non-UTF-8 octets.

### Next Phase
- Step 4: Neural Core Implementation & Training Pipeline (Awaiting User Instructions).






