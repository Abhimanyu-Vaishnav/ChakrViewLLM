# ChakrView Step 3 — Final Tokenizer Architecture Decision

**Document Status:** FROZEN & RATIFIED BY EMPIRICAL EVIDENCE  
**Decision Date:** 2026-09-26  
**Selected Configuration:** Byte-Level BPE, $V = 4096$, Raw Byte Pre-tokenization (Variant A), Normal Frequency BPE Numerics (Candidate C)

---

## 1. Selected Configuration Summary

Based on rigorous experimental measurement across 4 vocabulary scales, 3 numeric strategies, and 2 pre-tokenization variants on unseen validation and test corpora, the recommended tokenizer configuration for **ChakrView v0.1 (Chakr-Micro)** is:

| Parameter | Selected Value | Justification / Basis |
| :--- | :---: | :--- |
| **Tokenizer Architecture** | **Byte-Level BPE (BBPE)** | Guaranteed out-of-vocabulary immunity and 100% lossless byte reconstruction |
| **Vocabulary Size ($V$)** | **4,096** | Pareto-optimal balance between sequence compression, embedding parameter weight, and CPU cache locality |
| **Base Byte Primitives** | **256 tokens** ($3 \le \text{ID} \le 258$) | Full coverage of raw octet values `0x00..0xFF` |
| **Special Tokens** | $\langle\text{BOS}\rangle=0, \langle\text{EOS}\rangle=1, \langle\text{PAD}\rangle=2$ | Fixed frozen neural-core interface contract |
| **Learned BPE Merges** | **3,837 merges** ($259 \le \text{ID} < 4096$) | Learned from training split with deterministic tie-breaking |
| **Pre-tokenization** | **Variant A (Raw Byte BPE)** | Avoids +22.6% token expansion and 2× regex latency overhead of Variant B |
| **Default Numeric Strategy** | **Candidate C (Normal BPE)** | Avoids 2.69× sequence expansion of Candidate A in a 512-token context window |
| **Downstream Embedding Params** | **786,432 parameters** | Consumes exactly 22.84% of total model capacity (3.44M params) |
| **Static Tokenizer RAM** | **902.4 KB** | Fits entirely inside modern L2 / L3 processor caches |
| **Encode Latency (Median)** | **695.1 µs / doc** | Real-time interactive CPU execution without acceleration |
| **Decode Latency (Median)** | **2.84 µs / doc** | Ultra-low overhead autoregressive decoding on single thread |

---

## 2. Why $V = 4096$ Was Selected

### The 22.84% Parameter Sweet Spot
In a micro-scale language model like Chakr-Micro ($d_{\text{model}} = 192$, $N_{\text{layers}} = 6$, $N_{\text{heads}} = 6$, total parameters $\approx 3,443,712$), the vocabulary size directly determines the size of the embedding matrix ($V \times d_{\text{model}}$) and the tied output projection matrix ($d_{\text{model}} \times V$).

- At $V = 4096$, the embedding table consumes **786,432 parameters (22.84%)**, leaving **77.16%** of parameter capacity for multi-head attention and SwiGLU feed-forward networks (the core compute and reasoning layers).
- At $V = 8192$, the embedding table explodes to **1,572,864 parameters (45.67%)**. Nearly half of the entire neural model would be static lookup tables rather than trainable computational layers.
- At $V = 16384$, the embedding table consumes **67.00%** of the entire model parameter budget, crippling the depth and capacity of the transformer.

### Cache Locality & Static Footprint
The entire vocabulary lookup and merge dictionary for $V = 4096$ occupies **902.4 KB of RAM**. This fits comfortably into the L2/L3 cache of modern and legacy Intel, AMD, and ARM processors, preventing constant main memory page walks during tokenization.

---

## 3. What Was Sacrificed

Choosing $V = 4096$ over $V = 8192$ involves deliberate, measured trade-offs:

1. **Marginal Token Fertility:**
   - Hindi words require **2.17 tokens/word** at $V = 4096$ compared to **1.89 tokens/word** at $V = 8192$ (a sacrifice of ~0.28 tokens/word, or +14.8% sequence length).
   - English words require **2.49 tokens/word** at $V = 4096$ compared to **2.30 tokens/word** at $V = 8192$ (+8.2% sequence length).
2. **Context Window Occupancy:**
   - In a 512-token context window, $V = 4096$ can represent approximately 236 Hindi words or 205 English words, whereas $V = 8192$ would represent ~270 Hindi words.
   - However, this 14% context gain at $V = 8192$ would cost 786,432 neural parameters—parameters that are essential for grammatical reasoning, coherence, and associative recall.

---

## 4. Why Alternative Candidates Were Rejected

### Candidate $V = 2048$ — REJECTED
- **Under-segmentation:** Yields poor token fertility (Hindi: 2.40 tok/word; English: 2.88 tok/word; Numbers: 3.87 tok/word).
- **Attention Compute Explosion:** Longer sequence lengths directly increase the quadratic compute cost of attention ($O(L^2)$) by approximately $+35\%$, negating CPU inference savings.

### Candidate $V = 8192$ — REJECTED
- **Excessive Parameter Drain:** Requires 1.57M parameters (45.67% of the model).
- **Latency Penalty:** CPU encoding latency is +35.1% slower than $V = 4096$ ($939.4\,\mu\text{s}$ vs $695.1\,\mu\text{s}$).
- **Static RAM:** Doubles memory footprint to 1.85 MB.

### Candidate $V = 16384$ — REJECTED
- **Candidate Exhaustion:** On controlled datasets, unique adjacent pairs are exhausted at 12,018 merges. The tokenizer cannot reach 16,384 merges without memorizing whole sentences.
- **Diminishing Returns:** Compression improves by only +0.032 bytes/token over $V = 8192$.
- **Parameter Cannibalization:** Consumes 67.0% of the entire model budget.

---

## 5. Decision on Numeric and Pre-tokenization Strategies

### Numeric Strategy: Candidate C (Normal BPE) as Default
- Candidate A (single digits `\d`) inflates sequence lengths by **$2.69\times$**, consuming 97 tokens across test cases compared to 36 tokens for Candidate C. In a 512-token context window, this sequence tax is unacceptable for general text.
- Candidate C is selected as the default general-purpose representation.
- Candidate A is preserved as a modular, pluggable adapter for dedicated arithmetic reasoning modules.

### Pre-tokenization Strategy: Variant A (Raw Byte BPE)
- Variant B (Grapheme-aware pre-tokenization) causes a **+22.6% token sequence expansion** because BPE cannot merge across segment boundaries.
- Variant B doubles CPU encoding latency ($1132.5\,\mu\text{s}$ vs $577.7\,\mu\text{s}$) due to regex parsing overhead.
- Variant A is selected for optimal compression, minimal latency, and native simplicity.

---

## 6. Ratification

This decision is backed by empirical data recorded in `docs/STEP_03_BENCHMARK_RESULTS.md` and verified by 133 automated unit and integration tests.

The provisional contract defined in Step 2.4 is hereby **FORMALLY CONFIRMED AND FROZEN**.
