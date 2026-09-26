# ChakrView Step 3 — Tokenizer Training Specification & Pipeline

**Document Status:** FROZEN & EMPIRICALLY VERIFIED  
**Stage:** Step 3 (Empirical Tokenizer Training, Benchmarking & Selection)  
**Date:** 2026-09-26  
**Implementation:** `chakrview/tokenizer/trainer.py`, `chakrview/tokenizer/corpus.py`, `chakrview/tokenizer/serialization.py`

---

## 1. Executive Summary

Step 3 establishes the empirical research and production training infrastructure for the ChakrView Byte-Level Byte-Pair Encoding (BBPE) tokenizer. 

The tokenizer is designed to operate on low-resource and legacy CPUs (e.g. Intel Core i5/i7 laptops, ARM SoCs) without GPU acceleration, external dependencies, or opaque third-party tokenization frameworks (such as SentencePiece or HuggingFace Tokenizers).

All merges, vocabulary lookups, and serialization formats are 100% indigenous, deterministic, and verifiable.

---

## 2. Absolute Architectural Invariants

Every trained candidate and deployed tokenizer satisfies:

1. **Lossless Reconstruction Invariant:**
   $$\forall \text{ valid text } s: \quad \text{Decode}(\text{Encode}(s)) = s$$
   $$\forall \text{ arbitrary octet sequences } b \in \{0..255\}^*: \quad \text{DecodeBytes}(\text{EncodeBytes}(b)) = b$$
   Zero destructive normalization is applied during encoding or decoding.

2. **Special Token Isolation:**
   - $\langle\text{BOS}\rangle = 0$
   - $\langle\text{EOS}\rangle = 1$
   - $\langle\text{PAD}\rangle = 2$
   - Byte tokens $0..255$ map bijectively to IDs $3..258$.
   - BPE learned merges occupy token IDs $259 \le \text{ID} < V$.

3. **Deterministic Tie-Breaking:**
   Adjacent pair selection during merge learning follows the strict tuple key:
   $$\text{key} = (-\text{frequency}, \text{pair}[0], \text{pair}[1])$$
   - Primary: Highest frequency descending.
   - Secondary: First token ID ascending.
   - Tertiary: Second token ID ascending.
   Across identical training corpora, identical tokenizers are generated bit-for-bit.

4. **Zero Data Leakage:**
   The training split ($80\%$) is strictly isolated. Validation ($10\%$) and Test ($10\%$) splits remain completely unseen during merge extraction.

---

## 3. Corpus Engineering Pipeline

The corpus pipeline is implemented in [`chakrview/tokenizer/corpus.py`](file:///D:/Project/ChakrView/chakrview/tokenizer/corpus.py).

```
data/raw/ (Multi-Domain Controlled Text)
    ↓
Validation (UTF-8 Decode, Null Byte \x00 Check, SHA-256 Checksum)
    ↓
Deduplication (Order-preserving, Exact Match Elimination)
    ↓
Deterministic Category-Aware Split (SHA-256 Hash Partitioning: 80% Train, 10% Val, 10% Test)
    ↓
data/processed/train/    data/validation/    data/processed/test/
```

### 3.1 Corpus Distribution & Metrics

The training corpus spans 8 dedicated linguistic and technical domains:

| Category | Raw Bytes | Raw Pct | Train Lines | Val Lines | Test Lines | Content Scope |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Hindi** | 25,780 B | 42.3% | 196 | 30 | 22 | Devanagari computing, science, philosophy, dialogue |
| **English** | 11,006 B | 18.0% | 85 | 13 | 9 | Systems programming, algorithms, hardware caches |
| **Code** | 5,236 B | 8.6% | 40 | 6 | 4 | Python attention classes, JSON configs, CLI commands |
| **Hinglish** | 4,885 B | 8.0% | 40 | 6 | 4 | Romanized conversational Hindi tech dialogue |
| **Mixed** | 4,900 B | 8.0% | 40 | 6 | 4 | Multilingual strings, emojis, Latin + Devanagari |
| **Mathematics** | 3,345 B | 5.5% | 31 | 5 | 4 | Calculus, linear algebra, RoPE, SwiGLU formulas |
| **Sanskrit** | 3,024 B | 5.0% | 23 | 3 | 3 | Classical shlokas, Panini Maheshwara sutras |
| **Numbers** | 2,787 B | 4.6% | 21 | 3 | 3 | Dates, currency, decimals, scientific notation |
| **TOTAL** | **60,963 B** | **100.0%** | **476** | **72** | **53** | **19 validated files; 601 unique lines** |

All raw files are project-authored or derived from public domain classical texts with full licenses recorded in `data/raw/manifest.json`.

---

## 4. Memory-Conscious Training Architecture

To ensure the tokenizer can scale to larger datasets without exceeding host RAM limits, [`BPETrainer`](file:///D:/Project/ChakrView/chakrview/tokenizer/trainer.py) supports chunked stream training:

```python
def train_stream(
    chunk_iterator: Iterator[List[str]],
    target_vocab_size: int,
    verbose: bool = False
)
```

1. **Chunk Ingestion:** Text lines are consumed in configurable chunk batches (e.g. 1,000 lines).
2. **Frequency Compaction:** Raw strings are immediately converted to byte-token tuples and aggregated into a frequency `Counter`. Raw chunk memory is released immediately.
3. **Iterative Merge Reduction:** Merges are applied directly to unique sequence tuples, keeping working memory proportional to unique vocabulary forms rather than total corpus byte volume.

---

## 5. Serialization Contract

Per Section 20, all trained tokenizers are saved to disk using transparent, inspectable artifacts via [`chakrview/tokenizer/serialization.py`](file:///D:/Project/ChakrView/chakrview/tokenizer/serialization.py):

- **`merges.json`**: Pair string `"p0,p1"` $\to$ `new_token_id`.
- **`vocab.json`**: `token_id` $\to$ hex-encoded byte string (preserving raw octets `0x00..0xFF`).
- **`config.json`**: Metadata including version (`0.1.0`), requested vocab size, actual vocab size, special token IDs, checksums (`merges_sha256`, `vocab_sha256`), and creation timestamps.

Any tampering with vocabulary or merge tables triggers a hard checksum validation failure upon loading.
