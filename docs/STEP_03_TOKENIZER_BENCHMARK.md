# ChakrView Step 3 — Empirical Tokenizer Benchmark & Finalization Report

**Document Version**: 1.0.0  
**Project Phase**: Step 3 — Empirical Tokenizer Benchmark & Finalization  
**Status**: APPROVED & EMPIRICALLY RATIFIED  
**Evaluation Date**: 2026-09-26  
**Hardware Environment**: Intel Core i9-13900H (14 cores / 20 threads), 32 GB RAM, Windows 11, Python 3.14.7  

---

## 1. Objective

The objective of Step 3 is to replace provisional tokenizer hypotheses with **rigorous empirical evidence** measured on a dedicated, version-controlled multilingual benchmark corpus. Specifically, this phase evaluates:
1. **Vocabulary Candidate Sizing**: Systematic comparison of $V \in \{2048, 4096, 8192, 16384\}$ across 20 distinct linguistic, structural, memory, and performance metrics.
2. **Numeric Tokenization Strategies**: Empirical evaluation of Candidate A (single digits), Candidate B (2-digit chunks), and Candidate C (unconstrained frequency-based BPE).
3. **Stress Testing**: Adversarial evaluation across complex Devanagari conjuncts, classical Sanskrit, ZWJ/ZWNJ half-forms, multi-codepoint emojis, combining characters, and non-UTF-8 arbitrary raw byte streams.
4. **Hardware Latency & Memory Footprint**: Cold and warm latency profiling across input scales ($32\text{ B}$ to $128\text{ KB}$) and separation of static data from runtime process memory.
5. **Architectural Recommendation**: Empirical determination of whether $V = 4096$ remains appropriate for the Chakr-Micro v0.1 neural core.

---

## 2. Experimental Setup

- **Random Seed**: Fixed to `42` across all training, random byte generation, and benchmark executions.
- **Isolation Principle**: All benchmark scripts, candidates, fixtures, and reports reside in `experiments/tokenizer/` completely decoupled from production runtime code.
- **Engine**: Indigenous Byte-Level BPE engine (`chakrview/tokenizer/`) adhering to strict deterministic tie-breaking:
  $$\text{key} = (-\text{frequency}, \text{pair}_0, \text{pair}_1)$$
- **Test Corpus**: 748 curated items comprising $61,644\text{ UTF-8 bytes}$ spanning 23 distinct linguistic and technical categories.
- **Automated Verification**: Test suite expanded to 122 automated tests (107 baseline + 15 Step 3 benchmark tests), passing at $100\%$.

---

## 3. Corpus Design: 23-Category Control Corpus

A small, high-density, deterministic control corpus was constructed in `experiments/tokenizer/corpus/` covering 23 categories (A through W):

| Category Key | Category Filename | Description & Linguistic Coverage | Lines | UTF-8 Bytes |
| :--- | :--- | :--- | :---: | :---: |
| **A** | `english.txt` | Expository computer science, memory bandwidth, SIMD, AI systems | 15 | 1,842 |
| **B** | `hindi.txt` | Standard Devanagari Hindi text on science, technology, history | 12 | 1,863 |
| **C** | `sanskrit.txt` | Classical Sanskrit shlokas, Panini sutras, Upanishadic mantras | 10 | 1,515 |
| **D** | `hinglish.txt` | Colloquial Romanized Hindi-English digital code-switching | 11 | 973 |
| **E** | `indian_names.txt` | Indian personal, cultural, and geographical names (EN + HI) | 10 | 1,326 |
| **F** | `technical.txt` | Microprocessor architecture, AVX2, AVX-512, cache hierarchies | 10 | 1,180 |
| **G** | `python_code.txt` | Realistic Python classes, decorators, math functions, typings | 27 | 785 |
| **H** | `json_data.txt` | Structured nested JSON configuration manifests and schemas | 32 | 609 |
| **I** | `urls.txt` | Web endpoints, query parameters, ports, Devanagari path components | 10 | 604 |
| **J** | `windows_paths.txt`| Windows drive paths, UNC shares, spaces, system file extensions | 10 | 599 |
| **K** | `linux_paths.txt` | POSIX paths, hidden system directories, mount points, device paths | 11 | 400 |
| **L** | `mathematics.txt` | Mathematical equations, LaTeX formulas, Gaussian distributions | 12 | 674 |
| **M** | `numbers.txt` | Integers, counters, decimals, phone numbers, timestamps | 10 | 382 |
| **N** | `currencies.txt` | ₹ INR, $ USD, € EUR, £ GBP, Indian numbering (Lakhs, Crores) | 8 | 511 |
| **O** | `dates.txt` | ISO 8601 timestamps, European, Indian, historical date strings | 8 | 409 |
| **P** | `scientific_notation.txt` | Exponents, physical constants (Planck, Boltzmann, light speed) | 5 | 322 |
| **Q** | `emojis.txt` | Simple, composite, skin tones, flags, tech emojis, ZWJ sequences | 9 | 509 |
| **R** | `mixed_unicode.txt`| Polyglot sentences interweaving Hindi, Sanskrit, Latin, Greek | 8 | 772 |
| **S** | `devanagari_conjuncts.txt`| Complex consonant clusters, halant half-forms, ligatures | 7 | 812 |
| **T** | `zwj_zwnj.txt` | Explicit zero-width joiner (U+200D) and non-joiner (U+200C) tokens | 8 | 495 |
| **U** | `whitespace.txt` | Single/multiple spaces, tabs, mixed indents, leading/trailing | 12 | 315 |
| **V** | `line_endings.txt` | Windows CRLF (\r\n), Unix LF (\n), mixed newline sequences | 5 | 165 |
| **W** | `raw_bytes.txt` | Foundational byte documentation and boundary specifications | 9 | 661 |
| **TOTAL** | **23 Categories** | Combined with Step 2.3 validated corpus items | **748** | **61,644** |

---

## 4. Vocabulary Candidates Comparison (Phases 3.4 & 3.5)

The 20 mandatory Phase 3.5 metrics were computed across all four vocabulary candidates:

| # | Evaluation Dimension | V = 2048 | V = 4096 (Provisional) | V = 8192 | V = 16384 |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **1** | **Total Token Count** | $18,159$ | **$13,397$** | $8,907$ | $748$ |
| **2** | **Total UTF-8 Byte Count** | $61,644$ | **$61,644$** | $61,644$ | $61,644$ |
| **3** | **Tokens / Character (Overall)** | $0.405$ | **$0.299$** | $0.199$ | $0.017$ |
| **4** | **Tokens / Word (Overall)** | $2.678$ | **$1.976$** | $1.313$ | $0.110$ |
| **5** | **Bytes / Token (Compression)**| $3.395$ | **$4.601$** | $6.921$ | $82.412$ |
| **6** | **Compression Ratio** | $3.395$ | **$4.601$** | $6.921$ | $82.412$ |
| **7** | **Hindi (Tokens / Word)** | $2.46$ | **$1.76$** | $1.20$ | $0.07$ |
| **8** | **English (Tokens / Word)** | $2.57$ | **$1.89$** | $1.29$ | $0.07$ |
| **9** | **Hinglish (Tokens / Word)** | $1.87$ | **$1.30$** | $0.98$ | $0.08$ |
| **10**| **Sanskrit (Tokens / Word)** | $3.87$ | **$3.07$** | $2.01$ | $0.08$ |
| **11**| **Code (Tokens / Word)** | $2.56$ | **$1.98$** | $1.19$ | $0.15$ |
| **12**| **Mathematics (Tokens / Word)**| $2.75$ | **$2.09$** | $1.41$ | $0.14$ |
| **13**| **Numeric (Tokens / Word)** | $3.03$ | **$2.32$** | $1.56$ | $0.15$ |
| **14**| **Unicode (Tokens / Word)** | $2.64$ | **$1.98$** | $1.35$ | $0.13$ |
| **15**| **Emoji (Tokens / Word)** | $2.15$ | **$1.65$** | $1.10$ | $0.12$ |
| **16**| **Encode Latency (µs/doc)** | $429.8\text{ µs}$ | **$578.2\text{ µs}$** | $857.6\text{ µs}$ | $1,391.5\text{ µs}$ |
| **17**| **Decode Latency (µs/doc)** | $3.1\text{ µs}$ | **$2.6\text{ µs}$** | $2.0\text{ µs}$ | $0.5\text{ µs}$ |
| **18**| **Process Memory Footprint** | $435.7\text{ KB}$ | **$900.3\text{ KB}$** | $1,832.2\text{ KB}$ | $3,875.8\text{ KB}$ |
| **19**| **Merge Table Static Bytes** | $446,160\text{ B}$ | **$921,952\text{ B}$** | $1,876,192\text{ B}$ | $3,968,784\text{ B}$ |
| **20**| **Learned Merges Count** | $1,789$ | **$3,837$** | $7,933$ | $16,092$ (Exhausted) |
| — | **Actual Vocabulary Size** | $2,048$ | **$4,096$** | $8,192$ | $16,351$ |
| — | **Lossless Reconstruction** | **PASS (100%)** | **PASS (100%)** | **PASS (100%)** | **PASS (100%)** |

---

## 5. Numeric Tokenization Experiment (Phase 3.6)

Evaluated across the 25 dedicated arithmetic expressions, large numbers, decimals, scientific notations, and currencies:

| Numeric Strategy | Formulation | Total Tokens | Avg Tok / Item | Sequence Expansion vs Normal BPE | Lossless Check |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Candidate A** | Single Digits (`\d`) | $154$ | $6.16$ | **$2.11\times$** ($+111\%$) | **PASS (100%)** |
| **Candidate B** | 2-Digit Chunks (`\d{1,2}`) | $107$ | $4.28$ | **$1.47\times$** ($+47\%$) | **PASS (100%)** |
| **Candidate C** | Normal BPE (Unconstrained) | $73$ | $2.92$ | **$1.00\times$** (Baseline) | **PASS (100%)** |

### Detailed Analysis of Numeric Strategies:
1. **Candidate A (Single Digits)**:
   - *Advantage*: Guarantees absolute token boundary stability. Every digit $0\dots9$ is an independent, isolated token. Eliminates arbitrary multi-digit concatenation (e.g. `123` is always `[1] [2] [3]`). Substantially simplifies positional alignment for arithmetic heads.
   - *Disadvantage*: Inflates sequence length by $2.11\times$, consuming double the context window for numeric tables and mathematical expressions.
2. **Candidate B (2-Digit Chunks)**:
   - *Advantage*: Bounded vocabulary expansion; limits digit tokens to at most $100$ combinations ($00\dots99$).
   - *Disadvantage*: Odd-length numbers split unevenly depending on left-to-right grouping (e.g. `123` $\to$ `12` `3` vs `1` `23`), introducing slight tokenization jitter.
3. **Candidate C (Normal BPE)**:
   - *Advantage*: Maximal sequence compression ($2.92\text{ tok/item}$).
   - *Disadvantage*: Merges numbers based on training frequency, leading to inconsistent representations (e.g. year `2026` becomes a single token while `2027` is split into `20` `27`).

---

## 6. Unicode, Indic & Raw Byte Stress Tests (Phases 3.7 & 3.8)

### Phase 3.7: Unicode & Indic Adversarial Suite (41 Items)
All items were encoded, decoded, and asserted bit-for-bit identical:
- **Hindi Challenging Lexemes**: `कौशाम्बी`, `त्र्यंबकेश्वर`, `कुंडलियाँ`, `पूँछ`, `अँधेरा`, `ऋग्वेद`, `ॐ` $\to$ **$100\%$ PASS**.
- **Classical Sanskrit Clusters**: `सत्त्व`, `दग्ध`, `बुद्ध`, `उष्ट्र`, `कार्त्तिकेय`, `वाग्देवी` $\to$ **$100\%$ PASS**.
- **Zero-Width Joiner (ZWJ U+200D) & Non-Joiner (ZWNJ U+200C)**:
  - Half-forms: `क्` + ZWJ + `ष` $\to$ `क्‍ष` $\to$ **$100\%$ PASS**.
  - Explicit virama: `क्` + ZWNJ + `ष` $\to$ `क्‌ष` $\to$ **$100\%$ PASS**.
  - Isolated sequences: `\u200D\u200C\u200D\u200C` $\to$ **$100\%$ PASS**.
- **Complex Emoji Sequences**:
  - Family combination: `👨👩👧👦` (multiple codepoints + ZWJ) $\to$ **$100\%$ PASS**.
  - Profession modifier: `👩💻` (woman + ZWJ + laptop) $\to$ **$100\%$ PASS**.
  - Skin-tone modifier: `👍🏽` (thumbs up + medium skin tone) $\to$ **$100\%$ PASS**.
  - Flag indicator: `🇮🇳` (regional indicator symbols IN) $\to$ **$100\%$ PASS**.
- **Combining Marks & Variation Selectors**:
  - `e\u0301` (e with acute), `a\u0308\u0304` (a with diaeresis and macron) $\to$ **$100\%$ PASS**.
  - `\u2764\ufe0f` (emoji presentation), `\u2600\ufe0e` (text presentation) $\to$ **$100\%$ PASS**.
  - Zero-width non-breaking space `\ufeff`, word joiner `\u2060` $\to$ **$100\%$ PASS**.

### Phase 3.8: Raw Byte Arbitrary Sequences (11 Test Cases)
The raw byte contract `decode_bytes(encode_bytes(b)) == b` was verified across:
1. `b"\x00"` (single NULL byte) $\to$ Token ID `3` $\to$ Decoded: `b"\x00"` (**PASS**)
2. `b"\xff"` (single 0xFF byte) $\to$ Token ID `258` $\to$ Decoded: `b"\xff"` (**PASS**)
3. `b"\x00\xff\x00"` $\to$ **PASS**
4. All 256 byte octets ascending ($0\dots255$) $\to$ **PASS**
5. All 256 byte octets descending ($255\dots0$) $\to$ **PASS**
6. Random 256 bytes (deterministic seed 42) $\to$ **PASS**
7. Random 1024 bytes (deterministic seed 42) $\to$ **PASS**
8. Invalid isolated UTF-8 high bytes (`b"\x80\x81\x82\xfe\xff\xc0\xaf"`) $\to$ **PASS**
9. Truncated Hindi Devanagari 2-byte prefix (`b"\xe0\xa4"`) $\to$ **PASS**
10. Truncated Emoji 3-byte prefix (`b"\xf0\x9f\x98"`) $\to$ **PASS**
11. Mixed binary/text payload (`b"ChakrView\x80\xff\x00AI\xe0\xa4Core\xfe"`) $\to$ **PASS**

---

## 7. Performance & Latency Measurements (Phase 3.9)

Evaluated on the development machine across 7 payload sizes with 15 warm iterations per size:

| Input Payload Size | Token Count | Cold Encode | Warm Encode (Median) | Warm Decode (Median) | Encode Throughput | Decode Throughput |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **32 Bytes** | $6$ | $285.4\text{ µs}$ | **$270.2\text{ µs}$** | **$1.2\text{ µs}$** | $0.12\text{ MB/s}$ | $27.74\text{ MB/s}$ |
| **128 Bytes** | $17$ | $1,050.2\text{ µs}$| **$1,018.4\text{ µs}$** | **$2.4\text{ µs}$** | $0.11\text{ MB/s}$ | $53.07\text{ MB/s}$ |
| **512 Bytes** | $120$ | $9,210.1\text{ µs}$| **$9,056.9\text{ µs}$** | **$13.5\text{ µs}$** | $0.05\text{ MB/s}$ | $36.85\text{ MB/s}$ |
| **2 KB** | $538$ | $36,120.4\text{ µs}$| **$35,509.5\text{ µs}$** | **$59.4\text{ µs}$** | $0.06\text{ MB/s}$ | $32.88\text{ MB/s}$ |
| **8 KB** | $2,221$ | $142,500.1\text{ µs}$| **$140,044.7\text{ µs}$**| **$228.4\text{ µs}$** | $0.05\text{ MB/s}$ | $35.46\text{ MB/s}$ |
| **32 KB** | $8,883$ | $552,100.0\text{ µs}$| **$548,415.6\text{ µs}$**| **$781.8\text{ µs}$** | $0.06\text{ MB/s}$ | $39.38\text{ MB/s}$ |
| **128 KB** | $35,547$ | $2,210,400.0\text{ µs}$| **$2,186,984.2\text{ µs}$**| **$3,628.8\text{ µs}$** | $0.06\text{ MB/s}$ | $40.10\text{ MB/s}$ |

### Performance Observations:
- **Asymmetric Latency**: Decoding is **$100\times\text{--}600\times$ faster** than encoding ($1.2\text{ µs}$ for 32B decode vs $270.2\text{ µs}$ for encode).
  - *Reason*: Decoding is a simple direct array lookup $O(N)$ mapping token ID $\to$ bytes. Encoding requires sequential iterative merge scanning over token pairs in pure Python.
- **Inference Decode Throughput**: Autoregressive next-token decoding throughput exceeds $30\text{ MB/s}$, which corresponds to over $7,000,000\text{ tokens/second}$ during inference text regeneration.
- **Pure Python Prototype Bottleneck**: Python BPE pair iteration imposes an encode ceiling of $\approx 0.06\text{--}0.12\text{ MB/s}$. When promoted to C/Rust in future runtime phases, encoding throughput will scale by $50\times\text{--}100\times$.

---

## 8. Memory Profiling (Phase 3.10)

| Vocabulary Candidate | Merges Count | Vocab Table RAM | Merge Table RAM | Total Static Table Memory | Model Embedding Impact ($d=192$) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **V = 2048** | $1,789$ | $198.2\text{ KB}$ | $237.5\text{ KB}$ | **$435.7\text{ KB}$ ($0.43\text{ MB}$)** | $393,216$ params ($1.50\text{ MB}$ FP32) |
| **V = 4096** | $3,837$ | $410.5\text{ KB}$ | $489.8\text{ KB}$ | **$900.3\text{ KB}$ ($0.88\text{ MB}$)** | $786,432$ params ($3.00\text{ MB}$ FP32) |
| **V = 8192** | $7,933$ | $835.1\text{ KB}$ | $997.1\text{ KB}$ | **$1,832.2\text{ KB}$ ($1.79\text{ MB}$)**| $1,572,864$ params ($6.00\text{ MB}$ FP32) |
| **V = 16384** | $16,092$ | $1,765.4\text{ KB}$| $2,110.4\text{ KB}$| **$3,875.8\text{ KB}$ ($3.79\text{ MB}$)**| $3,145,728$ params ($12.00\text{ MB}$ FP32) |

### Memory Boundary Separation:
- **Static Table RAM**: Memory strictly required by `tokenizer.vocab` and `tokenizer.merges` in Python.
- **Runtime Working Memory**: Additional scratch list allocations during merge loops ($\approx 2\text{--}8\text{ KB}$ per encoded document).
- **Zero Memory Leaks**: Deterministic garbage collection cycles confirm zero memory drift across repeated encode/decode loops.

---

## 9. Determinism Results (Phase 3.11)

- **Test Condition**: Repeated encoding and decoding of 30 distinct multilingual sample texts across 5 independent iterations under fixed random seed 42.
- **Result**: **$0$ discrepancies detected**. Bit-exact token IDs, bit-exact reconstructed strings, and bit-exact merge ranks across all repetitions.
- **Conclusion**: The triple-key tie-breaking rule `(-frequency, pair[0], pair[1])` guarantees $100\%$ algorithmic determinism across runs.

---

## 10. Failure Cases & Boundary Observations

1. **Vocabulary Exhaustion at $V = 16384$**:
   - The trainer was instructed to generate $16,125$ merges to reach $V = 16384$.
   - At merge step $16,092$, pair frequency dropped to $1$, and all adjacent pairs across the entire corpus were completely consumed. The vocabulary reached an actual size of $16,351$ and halted.
   - *Observation*: Without a multi-megabyte corpus, attempting to train $V \ge 16384$ overfits to the point where entire sentences become individual tokens.
2. **Numeric Multi-Token Jitter in Candidate C**:
   - In Candidate C, `123456789` split into `[123, 45, 67, 89]` while `987654321` split into `[98, 76, 54, 321]`.
   - *Observation*: Normal BPE creates arbitrary subword boundaries in arithmetic strings, whereas Candidate A maintains uniform single-digit boundaries.

---

## 11. Engineering Interpretation & Decision Table

| Evaluation Criterion | Weight | V = 2048 | V = 4096 (Provisional) | V = 8192 | V = 16384 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Token Fertility (Hindi)** | High | $2.46\text{ tok/word}$ (Too long) | **$1.76\text{ tok/word}$ (Balanced)** | $1.20\text{ tok/word}$ (Dense) | $0.07\text{ tok/word}$ (Overfit) |
| **Token Fertility (English)** | High | $2.57\text{ tok/word}$ | **$1.89\text{ tok/word}$** | $1.29\text{ tok/word}$ | $0.07\text{ tok/word}$ |
| **Token Fertility (Sanskrit)** | Medium | $3.87\text{ tok/word}$ | **$3.07\text{ tok/word}$** | $2.01\text{ tok/word}$ | $0.08\text{ tok/word}$ |
| **Chakr-Micro Embedding Params**| High | $393,216$ ($12.9\%$) | **$786,432$ ($22.8\%$)** | $1,572,864$ ($37.2\%$) | $3,139,392$ ($54.2\%$) |
| **Static Tokenizer RAM** | Medium | $0.43\text{ MB}$ | **$0.88\text{ MB}$** | $1.79\text{ MB}$ | $3.79\text{ MB}$ |
| **Encode Latency** | Medium | $429.8\text{ µs}$ | **$578.2\text{ µs}$** | $857.6\text{ µs}$ | $1,391.5\text{ µs}$ |
| **Generalization Safety** | High | High (under-merges) | **Optimal (balanced)** | Risk of memorization | Severe memorization |
| **Lossless Round-Trip** | Critical | $100\%$ PASS | **$100\%$ PASS** | $100\%$ PASS | $100\%$ PASS |

---

## 12. Final Recommendation

### 1. Final Recommendation on Vocabulary Size: **RATIFY $V = 4096$ FOR v0.1**
- **Evidence-Based Rationale**:
  1. $V = 4096$ delivers a healthy token fertility of $1.76\text{ tok/word}$ for Hindi and $1.89\text{ tok/word}$ for English, representing a $26.2\%$ token compression improvement over $V = 2048$.
  2. In Chakr-Micro ($d_{\text{model}} = 192$), $V = 4096$ consumes exactly $786,432$ parameters ($22.8\%$ of total model capacity), leaving $77.2\%$ for relational Transformer reasoning layers. In contrast, $V = 8192$ inflates embedding weights to $37.2\%$ of the model, which over-allocates parameter capacity to memory lookup rather than attention reasoning.
  3. Static table memory is under $1\text{ MB}$ ($900.3\text{ KB}$), easily meeting low-spec CPU constraints.
  4. Candidate $V = 16384$ overfits the training distribution and exhausts merge candidates.

### 2. Final Recommendation on Numeric Tokenization: **RETAIN CANDIDATE C AS DEFAULT; PROVIDE CANDIDATE A HOOK**
- For generic language and conversational modeling, **Candidate C (normal BPE)** achieves optimal compression ($2.92\text{ tok/item}$).
- For arithmetic-heavy evaluation tasks, **Candidate A (single digits)** expands context by $2.11\times$ but guarantees strict single-digit boundary stability. The adapter in `experiments/tokenizer/candidates/numeric_adapters.py` provides this configurable capability without altering core tokenizer contracts.

---

## 13. Limitations & What Remains Provisional

1. **Corpus Scale Limitation**: The benchmark control corpus ($61,644\text{ bytes}$) is a high-density curated research corpus, not a multi-gigabyte internet crawl. While it proves correctness, fertility trends, and numerical stability, large-scale pre-training will require scaling corpus data.
2. **Pre-Tokenization Regex**: Current splitting utilizes whitespace and punctuation boundaries. Advanced language-specific regex pre-tokenization (e.g. Devanagari grapheme cluster boundaries) remains an unfrozen area for future tokenizer optimization.
3. **Execution Language**: The current prototype is implemented in pure Python. While decode throughput is fast ($>30\text{ MB/s}$), encode throughput ($0.06\text{--}0.12\text{ MB/s}$) will require a compiled C/Rust backend before multi-gigabyte dataset tokenization.

---
*End of Report — ChakrView Research Team*
