# ChakrView Multilingual Tokenizer Audit — Batches 2021–2060
**Milestone:** Batches 2021–2060  
**Branch:** `dev/v0.6-v0.7-validation`  
**Status:** Complete  
**Date:** October 2026  

---

## 1. Audit Overview & Objectives

ChakrView utilizes a frozen 4,096-token byte-level Byte-Pair Encoding (BPE) tokenizer located at:
`release/chakrview-v0.5.0/data/experiments/vocab_4096`

This audit evaluates the behavior and fertility of this tokenizer on diverse natural text across English, Hindi (Devanagari), Conversational Hinglish, Classical Sanskrit, Mathematics, and Python Source Code.

### Non-Negotiable Architectural Rule:
The canonical neural checkpoint (`training_checkpoint_step_10000.pt`, SHA-256 `96357705...`) has an embedding table strictly sized to `vocab_size = 4096`. Modifying the vocabulary, token IDs, or merges immediately breaks checkpoint compatibility and invalidates all pretraining weights. Therefore, any proposed tokenizer modification must be scientifically justified and maintained as a separate branch.

---

## 2. Quantitative Fertility & Fragmentation Measurements

We evaluated representative natural text samples from Corpus v2 across all supported languages and domains.

### Summary Metrics Table

| Language / Domain | Sample Chars | Sample Bytes | Tokens | Tokens / Word | Tokens / Char | Tokens / Byte | Roundtrip Fidelity |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Conversational Hinglish** | 719 | 719 | 170 | **1.405** | 0.236 | 0.236 | **100.0% Exact** |
| **Hindi Devanagari Prose** | 632 | 1,846 | 233 | **2.198** | 0.369 | 0.126 | **100.0% Exact** |
| **English Natural Prose** | 567 | 567 | 162 | **2.314** | 0.286 | 0.286 | **100.0% Exact** |
| **Mathematical Derivation** | 586 | 588 | 254 | **2.268** | 0.433 | 0.432 | **100.0% Exact** |
| **Python Code** | 703 | 703 | 285 | **3.239** | 0.405 | 0.405 | **100.0% Exact** |
| **Classical Sanskrit** | 733 | 2,168 | 425 | **4.670** | 0.580 | 0.196 | **100.0% Exact** |

---

## 3. In-Depth Linguistic Findings

### 3.1 Conversational Hinglish Efficiency
- **Finding:** Conversational Hinglish exhibits the highest tokenization efficiency in words per token (**1.405 tokens/word**).
- **Mechanism:** Hinglish predominantly uses Latin characters with common grammatical roots (e.g., *"karna"*, *"hai"*, *"aaj"*, *"weather"*, *"reminder"*). Because the byte-level BPE vocabulary has strong Latin subword merges, Hinglish suffers minimal fragmentation.

### 3.2 Hindi Devanagari Behavior
- **Finding:** Hindi Devanagari achieves **2.198 tokens/word** and **0.126 tokens/byte**.
- **Mechanism:** Devanagari characters in UTF-8 are 3 bytes each. A Hindi word like *"रसायन"* (Chemistry) decomposes into constituent syllable grams. While words take ~2.2 tokens, on a byte basis the tokenizer compresses 8 bytes into 1 token.
- **Matra & Halant Integrity:** Consonant conjuncts and vowel signs (matras) roundtrip with 100% fidelity without dangling surrogate bytes or UTF-8 corruption.

### 3.3 Classical Sanskrit Fragmentation
- **Finding:** Classical Sanskrit exhibits the highest tokens/word ratio (**4.670 tokens/word**).
- **Explanation:** Classical Sanskrit employs heavy sandhi (euphonic vowel/consonant fusion) and complex agglutinative compound words (samasa). For instance, long compound words like *"धर्मक्षेत्रे"* or *"कुरुक्षेत्रे"* are broken down into smaller subword morphemes.
- **Fidelity:** Despite the higher token-to-word ratio, roundtrip decoding is **100% byte-exact**.

### 3.4 Mathematics & Code Fragmentation
- **Python Code:** Achieves **3.239 tokens/word**. Code contains indentation whitespace, punctuation delimiters (`:`, `(`, `)`, `[`, `]`), and variable names with snake_case, leading to expected subword tokenization.
- **Mathematics:** Formulas containing Greek letters or operational symbols split into individual symbolic tokens, providing the autoregressive model with granular character-level access to algebraic variables.

---

## 4. Byte-Level Baseline & Roundtrip Verification

All samples were decoded back to strings and compared byte-for-byte against the original source inputs:
$$\text{Fidelity} = \frac{\sum \mathbb{I}(\text{decode}(\text{encode}(x)) == x)}{\text{Total Samples}} = 100.0\%$$

Zero decode exceptions, replacement characters (`\ufffd`), or length discrepancies were detected across any language.

---

## 5. Tokenizer Modification Decision & Recommendation

### Evaluation Against Improvement Threshold:
1. **Fidelity:** Roundtrip decoding is 100% lossless across all 6 evaluated domains.
2. **Devanagari Viability:** Hindi and Sanskrit tokenize cleanly without byte fallback artifacts.
3. **Compatibility:** Retaining `vocab_4096` preserves the pretraining weights of the canonical 10,000-step checkpoint.
4. **Context Consumption:** At 1.4 to 4.6 tokens/word, a standard 256-token context window can accommodate 55 to 180 words, which is well-matched to short-form instructions and reasoning probes.

### Verdict:
**MAINTAIN CANONICAL TOKENIZER `vocab_4096`.**  
No vocabulary expansion or re-training is required for Phase 5 continued pretraining. Expanding the vocabulary to 8k or 16k is scientifically rejected for this milestone because it would force a full cold-start pretraining run, throwing away the existing 10,000-step canonical weights without demonstrated need.
