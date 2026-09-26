# ChakrView Step 3 — Empirical Tokenizer Benchmark Results

**Document Status:** MEASURED EMPIRICAL EVIDENCE  
**Execution Date:** 2026-09-26  
**Benchmarking Engine:** `chakrview/tokenizer/benchmark.py`, `chakrview/tokenizer/metrics.py`  
**Artifact Hash Source:** `experiments/tokenizer/reports/step3_benchmark_results.json`

---

## 1. Benchmark Execution Environment

All benchmarks were executed locally on the designated host environment under controlled, identical thermal and process conditions:

- **OS:** Windows 11 AMD64 (10.0.26100)
- **CPU:** Intel64 Family 6 Model 186 Stepping 2 (Intel Core i9-13900H / RTX A1000 platform)
- **Python Version:** 3.14.7
- **Corpus Version:** ChakrView Controlled Corpus v0.1 (SHA-256 Manifest Verified)
- **Random Seed:** 42 (Deterministic partitioning and merge ordering)

---

## 2. Multi-Candidate Vocabulary Benchmark Table

Evaluated on the completely unseen **Validation Split** (72 documents, 7,599 UTF-8 bytes, 5,014 chars, 772 words):

| Metric | Candidate V=2048 | Candidate V=4096 | Candidate V=8192 | Candidate V=16384 |
| :--- | :---: | :---: | :---: | :---: |
| **Requested Vocab Size** | 2048 | 4096 | 8192 | 16384 |
| **Actual Vocab Size** | 2048 | 4096 | 8192 | **12018 (Exhausted)** |
| **Merges Learned** | 1,789 | 3,837 | 7,933 | 11,759 |
| **Training Time (sec)** | 5.62 s | 10.40 s | 16.94 s | 19.28 s |
| **Total Validation Tokens** | 2,181 | 1,879 | 1,697 | 1,685 |
| **Tokens / Character** | 0.4350 | 0.3748 | 0.3385 | 0.3361 |
| **Tokens / Word** | 2.8251 | 2.4339 | 2.1982 | 2.1826 |
| **Bytes / Token (Compression)** | 3.4842 | 4.0442 | 4.4779 | 4.5098 |
| **Hindi Tokens / Word** | 2.3955 | 2.1682 | 1.8864 | 1.8773 |
| **Hindi Bytes / Token** | 5.9677 | 6.5933 | 7.5783 | 7.6150 |
| **English Tokens / Word** | 2.8815 | 2.4882 | 2.2986 | 2.2938 |
| **English Bytes / Token** | 2.7007 | 3.1276 | 3.3856 | 3.3926 |
| **Hinglish Tokens / Word** | 2.0411 | 1.6027 | 1.3562 | 1.3288 |
| **Sanskrit Tokens / Word** | 4.8696 | 4.4348 | 4.1087 | 4.1087 |
| **Code Tokens / Word** | 2.8143 | 2.2143 | 2.0143 | 2.0000 |
| **Mathematics Tokens / Word** | 3.5814 | 2.9767 | 2.7674 | 2.7442 |
| **Numbers Tokens / Word** | 3.8696 | 3.1957 | 3.1522 | 3.1522 |
| **Encode Median Latency** | 542.5 µs | 695.1 µs | 939.4 µs | 1,212.2 µs |
| **Encode P95 Latency** | 575.4 µs | 715.7 µs | 1,033.8 µs | 1,285.0 µs |
| **Decode Median Latency** | 3.39 µs | 2.84 µs | 2.59 µs | 2.65 µs |
| **Tokenizer Static RAM** | 436.5 KB | 902.4 KB | 1,849.9 KB | 3,168.3 KB |
| **Embedding Parameters ($V \times 192$)** | 393,216 | 786,432 | 1,572,864 | 2,307,456 |
| **Embedding FP16 Memory** | 786.4 KB | 1.57 MB | 3.15 MB | 4.61 MB |
| **% of Model Parameters (3.44M)** | **11.42%** | **22.84%** | **45.67%** | **67.00%** |
| **Lossless Reconstruction** | **100% PASS** | **100% PASS** | **100% PASS** | **100% PASS** |

---

## 3. Generalization on Unseen Test Split

Evaluated on the completely unseen **Test Split** (53 documents, 4,239 UTF-8 bytes, 3,048 chars, 483 words):

| Metric | V=2048 | V=4096 | V=8192 | V=16384 |
| :--- | :---: | :---: | :---: | :---: |
| **Test Tokens / Word** | 2.7992 | 2.4327 | 2.1760 | 2.1690 |
| **Test Bytes / Token** | 3.1354 | 3.6077 | 4.0333 | 4.0410 |
| **Hindi Test Tok / Word** | 2.7982 | 2.5872 | 2.2752 | 2.2752 |
| **English Test Tok / Word** | 2.9857 | 2.6143 | 2.2429 | 2.2429 |
| **Test Lossless Invariant** | **100% PASS** | **100% PASS** | **100% PASS** | **100% PASS** |

---

## 4. Numeric Tokenization Experiment

Evaluated across 14 representative numeric structures:
- **Arithmetic:** `123 + 456 = 579`, `9999 * 8888 = 88871112`
- **Long integers:** `1234567890`, `42`, `0`
- **Decimals:** `3.1415926`
- **Negatives:** `-98765`
- **Currency:** `₹50000`
- **Percentages:** `13.56%`
- **Network addresses:** `192.168.1.1`
- **Dates & Times:** `2026-09-26`, `12:45:31`
- **Scientific notation:** `1.23e-10`
- **Semantic versions:** `v0.1.0`

### Measured Results:

| Strategy | Total Tokens Across Test Battery | Sequence Expansion vs Normal BPE | Lossless Reconstruction |
| :--- | :---: | :---: | :---: |
| **Candidate A (Single Digits `\d`)** | **97** | **2.69× (+169.4%)** | 100% PASS |
| **Candidate B (Two-Digit Chunks `\d{1,2}`)** | **73** | **2.03× (+102.8%)** | 100% PASS |
| **Candidate C (Normal Frequency BPE)** | **36** | **1.00× (Baseline)** | 100% PASS |

### Engineering Findings:
- While Candidate A decomposes numbers into isolated digit tokens, it inflates context occupancy by **$2.69\times$**. In a 512-token context micro model, a simple financial balance or log entry would occupy disproportionate sequence budget.
- Candidate C achieves optimal compression (36 tokens total), representing common years (`2026`), IP blocks (`192`), and round amounts efficiently while guaranteeing 100% exact round-trip reconstruction.
- Candidate C is selected as the default general-purpose representation. Candidate A is retained as an optional pre-tokenization adapter for arithmetic reasoning tasks.

---

## 5. Grapheme-Aware Pre-Tokenization Experiment

Compared:
- **Variant A:** Raw byte-level BPE (unrestricted byte pair merges).
- **Variant B:** Grapheme-aware pre-tokenization + byte BPE (merges constrained within Devanagari grapheme clusters, words, and whitespace).

### Measured Results:

| Metric | Variant A (Raw Byte BPE) | Variant B (Grapheme-Aware BPE) | Delta / Impact |
| :--- | :---: | :---: | :---: |
| **Total Tokens** | 2,181 | 2,675 | **+22.6% token expansion** |
| **Tokens / Word** | 2.8251 | 3.4650 | **+22.6% sequence length** |
| **Encode Latency** | 577.7 µs / doc | 1,132.5 µs / doc | **+96.0% slower (2× CPU latency)** |
| **Reconstruction** | 100% Lossless | 100% Lossless | Both strictly lossless |

### Engineering Findings:
- Constraining BPE to prevent cross-boundary merges prevents the tokenizer from learning frequent multi-word or word-punctuation collocations.
- The regex pre-tokenization overhead on CPU adds significant execution time ($1.13\text{ ms}$ vs $0.58\text{ ms}$).
- Therefore, **Variant A (Raw Byte BPE)** is strictly superior in both sequence efficiency and CPU execution speed.

---

## 6. Adversarial and Unicode Invariant Results

Tested against the exhaustive adversarial test suite:
- **Devanagari Complex Conjuncts:** कौशाम्बी, त्र्यंबकेश्वर, कुंडलियाँ, पूँछ, अँधेरा, ऋग्वेद, ॐ $\to$ **PASS (Exact match)**
- **Classical Sanskrit Shlokas:** सत्त्व, दग्ध, बुद्ध, उष्ट्र, कार्त्तिकेय, वाग्देवी $\to$ **PASS (Exact match)**
- **Half-Forms with ZWJ / ZWNJ:** क्\u200Dष vs क्\u200Cष $\to$ **PASS (Zero bytes altered)**
- **Multi-Codepoint Emojis:** 👨👩👧👦, 👩💻, 👍🏽, 🇮🇳 $\to$ **PASS (Exact match)**
- **Mathematical Symbols:** $\forall x \in A$, $\int f(x) dx$, $E=mc^2$ $\to$ **PASS (Exact match)**
- **Arbitrary Binary / Invalid UTF-8:** `b"\x00"`, `b"\xff"`, truncated lead bytes $\to$ **PASS (`decode_bytes(encode_bytes(b)) == b`)**
