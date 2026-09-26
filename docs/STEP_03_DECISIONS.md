# ChakrView Step 3 — Architectural Decision Record & Empirical Ratification

**Document Status:** FROZEN & RATIFIED BY EMPIRICAL EVIDENCE  
**Stage:** Step 3 (Tokenizer Corpus, Training, Benchmarking & Selection)  
**Date:** 2026-09-26  
**Benchmarking Report:** [`docs/STEP_03_TOKENIZER_BENCHMARK_REPORT.md`](file:///D:/Project/ChakrView/docs/STEP_03_TOKENIZER_BENCHMARK_REPORT.md)

---

## 1. Scope of Investigation

Step 3 investigated three core empirical questions:
1. **Vocabulary Size Selection:** Which vocabulary size candidate ($V \in \{2048, 4096, 8192, 16384\}$) provides the optimal trade-off between sequence compression, language balance, latency, static memory, and model parameter budget?
2. **Numeric Tokenization Strategy:** Does digit-level decomposition (Candidate A), two-digit chunking (Candidate B), or normal frequency-driven BPE (Candidate C) best balance sequence length and numerical integrity?
3. **Pre-tokenization Strategy:** Does regex grapheme-cluster isolation (Variant B) improve Indic representation sufficiently to justify its latency and token expansion costs over raw byte-level BPE (Variant A)?

---

## 2. Summary of Measured Evidence

1. **Vocabulary Scaling:**
   - $V = 2048$ under-segments text ($2.40\text{ tok/word}$ Hindi, $2.88$ English), expanding sequence lengths and increasing the quadratic attention compute of the transformer ($O(L^2)$).
   - $V = 4096$ delivers **$2.17\text{ tok/word}$ on Hindi** and **$2.49$ on English**, requires **$902.3\text{ KB}$ static RAM** (fitting inside CPU L2 cache), and allocates **$786,432$ parameters ($22.84\%$)** to the embedding table, leaving $77.16\%$ for core transformer reasoning layers.
   - $V = 8192$ improves Hindi fertility marginally to $1.89\text{ tok/word}$, but allocates **$45.67\%$ of the entire model parameter budget** to static lookup tables and increases CPU encode latency by $+41\%$.
   - $V = 16384$ completely exhausts unique merge candidates on the research corpus at 12,077 merges, memorizing entire sentences (`'chalo ab re'`, `'error de toh '`), and consumes **$67.33\%$** of the model budget.

2. **Numeric Representation:**
   - Candidate A (Single Digits `\d`) inflates sequence lengths by **$2.64\times$** ($+164.4\%$), severely crowding the 512-token context window of Chakr-Micro.
   - Candidate C (Normal BPE) represents numbers in 45 tokens across the benchmark battery with 100% exact lossless reconstruction.

3. **Pre-tokenization:**
   - Variant B (Grapheme-aware BPE) causes a **$+23.6\%$ sequence expansion** and **$89.5\%$ CPU latency penalty** compared to Variant A (Raw Byte BPE).

---

## 3. Frozen Decisions

The following architectural choices are formally ratified based on empirical measurements:

1. **Vocabulary Size:** $V = 4096$ is **FROZEN** for Chakr-Micro v0.1.
2. **Base Tokenizer Architecture:** Byte-Level BPE (BBPE) with 256 foundational byte primitives ($3 \le \text{ID} \le 258$) and 3 special tokens ($\langle\text{BOS}\rangle=0, \langle\text{EOS}\rangle=1, \langle\text{PAD}\rangle=2$) is **FROZEN**.
3. **Lossless Invariant:** Strict $\text{Decode}(\text{Encode}(S)) \equiv S$ without destructive runtime Unicode normalization is **FROZEN**.
4. **Pre-tokenization Strategy:** Variant A (Raw Byte BPE) is **FROZEN** as the default runtime engine.
5. **Default Numeric Strategy:** Candidate C (Normal frequency-driven BPE) is **FROZEN** for general language pre-training.

---

## 4. Decisions Kept Experimental / Unfrozen

The following decisions remain provisional pending neural-core training evidence in future steps:

1. **Task-Specific Numeric Adapter:** Candidate A (Single Digits) remains preserved as an optional pre-tokenization adapter for dedicated calculator / arithmetic tasks where single-digit alignment outweighs context occupancy.
2. **Dynamic Vocabulary Expansion:** Vocabulary scale $V = 8192$ remains an architectural extension point for future Chakr-Base / Chakr-Large models ($d_{\text{model}} \ge 512$) where the embedding table does not exceed $25\%$ of total parameters.
3. **C-Accelerated Inference:** Implementing native C/Rust extensions for the BPE merge loop is deferred to post-baseline optimization.

---

## 5. What Remains Uncertain / Risks Identified

1. **Sanskrit Conjunct Granularity:** Sanskrit text requires $4.43\text{ tok/word}$ due to long syllabic consonant clusters. Additional classical texts will be required if Sanskrit pre-training is prioritized.
2. **True CPU Cache Locality:** While static tables ($902.3\text{ KB}$) fit inside L2 cache, the interaction between quantized neural weights (3.44 MB) and tokenizer memory inside L3 cache must be benchmarked under concurrent execution in Step 4.

---

## 6. Ratification

This record completes Step 3. All decisions are documented, reproducible, and supported by 133 automated unit tests.
