# ChakrView Step 2.3 — Corpus Engineering & BPE Vocabulary Experiment Report

**Document Version**: 0.1.0  
**Phase**: Step 2.3 — Corpus Pipeline & BPE Empirical Benchmark  
**Status**: EXPERIMENT COMPLETED & VERIFIED  
**Date**: September 26, 2026  

---

> ### SCIENTIFIC DISCLAIMER & POLICY COMPLIANCE
> - **Zero Web Scraping / Zero Downloads**: All text in this corpus was either manually authored, synthesized from engineering test cases, or generated using permissive procedural examples. No external datasets, copyrighted books, or pretrained tokenizers were downloaded or used.
> - **Corpus Limitations**: This corpus does **not** represent the global real-world distribution of Hindi, English, Hinglish, or programming languages. It is an isolated, controlled benchmark rig created for tokenizer engineering.
> - **Empirical Framing**: All evaluative conclusions in this document apply strictly **"on this benchmark corpus"** and must not be cited as universal or final.
> - **Correctness Invariant**:
>   $$\text{Decode}(\text{Encode}(\text{text})) \equiv \text{text} \quad \forall \text{ inputs}$$
>   *"Tokenizer correctness takes precedence over compression."*

---

## 1. Corpus Description & Categories

The research corpus is organized in `data/tokenizer_corpus/` across eight isolated, category-specific text files:

```
data/tokenizer_corpus/
├── hindi.txt      (Devanagari script: aksharas, matras, conjuncts, literature, philosophy, science)
├── english.txt    (Standard English: computer systems, CPU/GPU, OS kernels, algorithms)
├── hinglish.txt   (Romanized colloquial Hindi in Latin script: chat, technical discussions)
├── mixed.txt      (Bilingual code-switching: intra-sentence Hindi-English alternation)
├── code.txt       (Source code in Python, C, Bash, SQL with multi-space and tab indentation)
├── numbers.txt    (Integers, floats, dates, timestamps, Indian currency ₹, scientific notation)
├── math.txt       (Mathematical logic, linear algebra, calculus, set theory, Greek symbols)
├── unicode.txt    (Complex emojis, ZWJ sequences, Vedic accents, regional Indic scripts, runes)
└── README.md      (Corpus provenance, license, limitations, and usage constraints)
```

---

## 2. Corpus Statistics

Analyzed using the `chakrview.tokenizer_corpus.statistics` module:

| Metric | Measured Value |
| :--- | :--- |
| **Total Documents / Lines** | 490 |
| **Unique Documents** | 485 |
| **Duplicate Lines Removed** | 5 |
| **Total Characters** | 30,583 |
| **Total UTF-8 Bytes** | 42,193 |
| **Total Whitespace Characters** | 4,941 |
| **Longest Sample** | 160 characters (`math.txt`: LaTeX Greek letters) |
| **Shortest Sample** | 1 character (`code.txt`: block delimiter `}`) |

### Category Breakdown
- **Hindi (Devanagari)**: 74 docs, 15,227 bytes (**36.09%** byte share)
- **English**: 73 docs, 7,412 bytes (**17.57%** byte share)
- **Extended Unicode & Emojis**: 58 docs, 4,405 bytes (**10.44%** byte share)
- **Source Code**: 109 docs, 3,692 bytes (**8.75%** byte share)
- **Mixed / Code-Switched**: 37 docs, 3,420 bytes (**8.11%** byte share)
- **Hinglish**: 47 docs, 3,345 bytes (**7.93%** byte share)
- **Mathematics & Logic**: 42 docs, 2,602 bytes (**6.17%** byte share)
- **Numbers & Currency**: 50 docs, 2,090 bytes (**4.95%** byte share)

---

## 3. Corpus Pipeline Architecture

The pipeline is implemented as an isolated, standard-library-only package in `chakrview/tokenizer_corpus/`:

```
Raw Corpus (.txt files)
         │
         ▼
     [loader.py] ────────► Load CorpusItem(text, category, file, line)
         │
         ▼
    [validator.py] ──────► Verify UTF-8, non-emptiness, reject embedded NULs
         │
         ▼
  [normalization.py] ───► Apply Training Normalization (NFC, strip trailing WS)
         │
         ▼
      [dedup.py] ────────► Deduplicate lines while preserving sequence order
         │
         ▼
    [statistics.py] ─────► Extract script, character, byte, and category distributions
         │
         ▼
      [split.py] ────────► Stratified 80/20 train/val split using deterministic hashing
         │
         ├───► Train Split: 383 items (33,648 bytes)
         └───► Val Split:   107 items (8,545 bytes)
```

### Normalization Policy Distinction
- **Training Normalization**: Applies Unicode NFC canonical composition and strips trailing line whitespace to stabilize merge frequency counting during vocabulary generation.
- **Runtime Input Encoding**: **STRICTLY LOSSLESS**. No normalization is performed during inference. Both CRLF and LF, uncomposed characters, and raw bytes are tokenized directly to guarantee $\text{Decode}(\text{Encode}(S)) \equiv S$.

---

## 4. BPE Training Procedure

The research BPE trainer (`chakrview.tokenizer.trainer.BPETrainer`) implements:
1. **Base Primitive Initialization**: Starts with the 3 special tokens (`<BOS>`: 0, `<EOS>`: 1, `<PAD>`: 2) and the 256 fundamental byte tokens (`0x00`..`0xFF` mapped to IDs 3..258).
2. **Frequency Counting**: Iterates across token sequences to count adjacent token pairs.
3. **Deterministic Tie-Breaking Rule**:
   $$\text{Best Pair} = \arg\min_{\text{pair}} \Big( -\text{frequency}(\text{pair}), \; \text{pair}[0], \; \text{pair}[1] \Big)$$
   Primary: Frequency descending. Secondary: First token ID ascending. Tertiary: Second token ID ascending.
4. **Merge Application**: Replaces non-overlapping occurrences from left to right.
5. **Artifact Output**: Saves `merges.json`, `vocab.json`, `train_stats.json`, and `metadata.json` under `data/tokenizer_experiments/v{V}/`.

---

## 5. Candidate Vocabulary Experiments

Four independent candidates were trained:

1. **Candidate $V = 2,048$**:
   - Target merges: 1,789. Performed: 1,789 in 4.37s. Actual Vocab: 2,048.
2. **Candidate $V = 4,096$**:
   - Target merges: 3,837. Performed: 3,837 in 8.21s. Actual Vocab: 4,096.
3. **Candidate $V = 8,192$**:
   - Target merges: 7,933. Performed: 7,933 in 12.49s. Actual Vocab: 8,192.
4. **Candidate $V = 16,384$**:
   - Target merges: 16,125. Performed: 9,788 in 13.25s. Actual Vocab: 10,047.
   - *Corpus Merge Exhaustion*: After 9,788 merges, all adjacent pair combinations in the training set were exhausted.

---

## 6. Empirical Benchmark Results

Evaluated on the identical validation split (107 items, 8,545 bytes, 6,475 characters, 1,028 words):

| Candidate | Actual Vocab | Val Tokens | Compression (Bytes/Token) | Tokens / Word | Lossless Test | Encode ($\mu\text{s}$) | Decode ($\mu\text{s}$) | Vocab RAM |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$V = 2048$** | 2,048 | 3,243 | 2.63 | 3.15 | **PASS** | 402.2 | 3.14 | 435.5 KB |
| **$V = 4096$** | 4,096 | 2,949 | 2.90 | 2.87 | **PASS** | 513.1 | 2.99 | 898.1 KB |
| **$V = 8192$** | 8,192 | 2,777 | 3.08 | 2.70 | **PASS** | 763.0 | 3.39 | 1,852.5 KB |
| **$V = 16384$** | 10,047 | 2,774 | 3.08 | 2.70 | **PASS** | 877.6 | 3.50 | 2,228.1 KB |

---

## 7. Category Efficiency Analysis (Tokens per Word)

| Category | $V = 2048$ | $V = 4096$ | $V = 8192$ | $V = 16384$ |
| :--- | :---: | :---: | :---: | :---: |
| **Hindi (Devanagari)** | 2.63 | 2.43 | 2.23 | 2.23 |
| **English** | 3.10 | 2.78 | 2.57 | 2.56 |
| **Hinglish** | 2.06 | 1.77 | 1.57 | 1.57 |
| **Code** | 3.02 | 2.60 | 2.52 | 2.52 |
| **Numbers & Currency** | 3.44 | 3.20 | 3.11 | 3.11 |
| **Mathematics & Logic** | 3.40 | 3.00 | 2.85 | 2.85 |
| **Extended Unicode & Emojis**| 5.16 | 4.90 | 4.78 | 4.78 |
| **Mixed / Code-Switched** | 2.25 | 2.00 | 1.84 | 1.84 |

---

## 8. Numeric Strategy Experiment

Evaluated on candidates $V = 4096$ and $V = 8192$:
- **Candidate A: Individual Digits (`\d`)**:
  - Encodes numbers strictly digit-by-digit.
  - `1234567890` produces 10 tokens. Code snippet expands from 26 to 29 tokens.
  - **Advantage**: Aligns directly with column-wise arithmetic algorithms; guarantees zero out-of-distribution numeric combinations.
- **Candidate B: Two-Digit Chunks (`\d{1,2}`)**:
  - Groups pairs of digits (`00`..`99`).
  - `1234567890` produces 5 tokens. Code snippet produces 26 tokens (matching normal BPE).
  - **Advantage**: Substantially reduces sequence length for timestamps, dates, and large numbers while bounding subword explosion.
- **Candidate C: Normal BPE**:
  - Digits merged based on training frequency.
  - Causes memorization of arbitrary specific dates (e.g. `2026-09-26` merges into 1 monolithic token).
  - **Risk**: Fails to generalize to unseen arithmetic and creates inconsistent token boundaries for identical digits.

---

## 9. Failure Analysis & Invariants

- **Lossless Verification**: Every vocabulary candidate ($V = 2048, 4096, 8192, 16384$) achieved **0 reconstruction failures** across the 107 validation items:
  $$\text{Decode}(\text{Encode}(\text{text})) == \text{text} \quad (100.0\% \text{ pass rate})$$
- **Special Token Safety**: Literal user text containing `"<BOS>"`, `"<EOS>"`, or `"<PAD>"` remained 100% isolated.
- **Merge Exhaustion Finding**: $V = 16384$ could not reach its target merge count because small training corpora naturally run out of frequent n-grams, demonstrating that large vocabulary targets require proportionally scaled datasets.

---

## 10. Technical Interpretation & Recommendation

1. **$V = 16,384$ is Empirically Disqualified for Micro Tier**:
   Improving from $V = 8192$ to $V = 16384$ reduced validation tokens by only 3 tokens ($0.1\%$) while increasing encoding latency by $15\%$ and exhausting corpus combinations. In Chakr-Micro, $V = 16384$ would consume $54.2\%$ of total parameter budget in embeddings ($3.15\text{M}$ params), starving the neural attention layers.
2. **$V = 4,096$ Achieves the Optimal Pareto Frontier on This Benchmark**:
   - Reduces validation tokens by $9.1\%$ compared to $V = 2048$.
   - Achieves $2.43$ tokens/word for Hindi, $1.77$ for Hinglish, and $2.78$ for English on this benchmark corpus.
   - Requires only $898\text{ KB}$ of vocab table RAM and executes in $513\,\mu\text{s}$ per item on edge CPU.
   - Consumes exactly $22.8\%$ of Chakr-Micro parameter capacity ($0.79\text{M}$ parameters), leaving $77.2\%$ for reasoning blocks.
3. **Formal Recommendation**:
   **$V = 4,096$ is the recommended vocabulary size on this research benchmark.**  
   $V = 8,192$ remains a secondary research candidate for larger model tiers (such as Chakr-Small, $d=384$).

---

## 11. Test Suite Verification

Executed across 97 unit and integration tests:
```
============================= test session starts =============================
platform win32 -- Python 3.14.7, pytest-9.1.1, pluggy-1.6.0
rootdir: D:\Project\ChakrView
configfile: pytest.ini
testpaths: tests
collected 97 items

tests\test_bpe_trainer.py .....                                          [  5%]
tests\test_corpus_pipeline.py .......                                    [ 12%]
tests\test_tokenizer_adversarial.py ...............................      [ 44%]
tests\test_tokenizer_baseline.py .                                       [ 45%]
tests\test_tokenizer_bpe_engine.py ......                                [ 51%]
tests\test_tokenizer_bytes.py .......                                    [ 58%]
tests\test_tokenizer_determinism.py ...                                  [ 61%]
tests\test_tokenizer_special_tokens.py ....                              [ 65%]
tests\test_tokenizer_utf8_lossless.py .................................  [100%]

============================= 97 passed in 0.16s ==============================
```
