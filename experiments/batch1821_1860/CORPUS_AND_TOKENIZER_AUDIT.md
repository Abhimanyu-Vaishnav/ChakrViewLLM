# CORPUS AND TOKENIZER AUDIT
## Batches 1821–1860: Provenance, Domain Distributions, Devanagari Encoding & Leakage Audit

### 1. Corpus Sources & Provenance

The ChakrView training and evaluation corpus resides under `data/raw/` and is indexed via `data/raw/manifest.json`.

- **Total Files**: 27 files
- **Total Raw Bytes**: 1,573,825 bytes (~1.57 MB)
- **License**: CC0-1.0 / MIT Compatible project-authored indigenous synthetic text and public domain classical excerpts.
- **Domain Breakdown**:
  1. **Code** (4 files, 217,108 bytes): Python algorithms, data structures, configurations, and system logic.
  2. **English Prose & Technical** (4 files, 349,643 bytes): Architecture specifications, operating systems, distributed systems, and networking documentation.
  3. **Hindi** (5 files, 511,714 bytes): Native Devanagari text covering philosophy, grammar, and general knowledge.
  4. **Hinglish** (3 files, 144,796 bytes): Romanized Hindi colloquial and technical dialogues.
  5. **Mathematics** (3 files, 152,472 bytes): Linear algebra, calculus, discrete mathematics, and worked arithmetic proofs.
  6. **Reasoning & Formal Logic** (1 file, 71,220 bytes): Propositional logic, deduction syllogisms, and causal inference.
  7. **Numbers** (3 files, 24,038 bytes): Tabular sequences, prime sequences, and arithmetic tables.
  8. **Sanskrit** (2 files, 97,885 bytes): Classical linguistic verses and grammatical aphorisms.
  9. **Mixed** (2 files, 4,949 bytes): Multi-lingual conversational transitions.

---

### 2. Tokenizer Architecture & Byte-Fallback Encoding Behavior

- **Tokenizer Implementation**: Byte-Pair Encoding (BPE) with byte-level fallback.
- **Artifacts Path**: `release/chakrview-v0.5.0/data/experiments/vocab_4096`
- **Vocabulary Size**: 4,093 token entries + special tokens (`<bos>=0, <eos>=1, <pad>=2`).
- **Devanagari Token Findings**:
  - Direct inspection of `vocab.json` reveals **0 multi-character Devanagari merge tokens**.
  - All Devanagari Hindi/Sanskrit characters are encoded via **underlying UTF-8 byte tokens** (`<0xE0>` through `<0xFF>`).
  - **Roundtrip Fidelity**: Because byte-level fallback is active, UTF-8 byte sequences decode with **100% lossless fidelity** (`text == tok.decode(tok.encode(text))`).
  - **Efficiency Implication**: A Hindi word requires 2 to 3 byte tokens per character (e.g. 5 tokens for a 2-word phrase), resulting in a higher token density compared to Latin scripts. Hinglish (Romanized Hindi) achieves standard subword compression with average word-to-token ratios of 1.2.

---

### 3. Data Integrity & Leakage Controls

- **Synthetic vs. Real-World Transparency**: All project-authored technical and arithmetic data are explicitly recorded as **indigenous synthetic corpora**. None are misrepresented as natural web crawl data.
- **Document-Level Split Partitioning**: To avoid train/test contamination:
  - Held-out evaluation documents are drawn from dedicated test sections never exposed to the canonical Step 10,000 checkpoint or continued pre-training runs.
  - Test suites enforce strict deduplication: n-gram overlap between train and test documents is capped below 5%.
