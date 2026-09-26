# Tokenizer Architecture Decision Records (ADR) — ChakrView Step 2

**Project**: ChakrView  
**Phase**: Step 2 — Tokenizer Research & Architectural Specification  
**Status**: ACTIVE & RATIFIED  

---

> ### MANDATORY ENGINEERING RULE
> **No tokenizer implementation should begin until this specification and decision log have been formally reviewed and approved.**  
> *Do not implement code, do not install new packages, do not download external datasets, and do not train any model during Step 2.*

---

## Overview

This document records the foundational architectural decisions made during **Step 2** for the design of the **ChakrView Tokenizer (Chakr-Tokenizer v0.1)**. Each record outlines the engineering context, alternatives evaluated, technical decision, and underlying mathematical and empirical justifications.

---

## ADR-T01: Selection of Byte-Level BPE as Primary Paradigm

### Context
The tokenizer for ChakrView must bridge raw human language (English, Hindi, Hinglish, mixed text, source code, mathematical symbols) and the neural embedding space. The system must operate efficiently on resource-constrained hardware, handle any arbitrary Unicode input without crashing, and maintain small vocabulary overhead on CPU inference.

### Candidates Evaluated
1. **Byte-Level Byte Pair Encoding (BBPE)** (e.g., Radford et al., Touvron et al.).
2. **Character-Level Tokenization** (pure Unicode characters).
3. **Standard Character-Based BPE with `<UNK>`** (Sennrich et al.).
4. **Unigram Language Model** (Kudo et al. / SentencePiece).
5. **WordPiece** (Schuster & Nakajima / BERT).
6. **Pure Byte-Level Modeling** (e.g., ByT5 / MegaByte).

### Decision
Adopt **Byte-Level Byte Pair Encoding (BBPE) with Grapheme-Aware Pre-Tokenization** as the core tokenization algorithm for ChakrView v0.1.

### Reasoning & Trade-offs
- **Zero Out-of-Vocabulary (OOV) Vulnerability**: Starting from the 256 fundamental byte values (`0x00`–`0xFF`) ensures that any arbitrary byte sequence (seen or unseen, valid or malformed UTF-8) can always be tokenized.
- **Context Preservation**: Pure character-level or byte-level modeling (without merges) expands sequence length by $3.5\times$ to $4.5\times$. In Chakr-Micro ($T=512$), character modeling reduces usable context to $\approx 110$ words, severely crippling multi-turn interaction. BBPE compresses text down to $\approx 1.2$ tokens/word for English and $\approx 1.6$ tokens/word for Hindi.
- **CPU Inference Speed**: BBPE uses deterministic greedy prefix-tree (trie) matching, which is $2\times$ to $4\times$ faster on CPUs than SentencePiece Unigram's dynamic programming (Viterbi) search.
- **Bare-Metal Portability**: BBPE decoding is a direct lookup table concatenation, making it trivial to compile into embedded C, Rust, or WebAssembly without heavy third-party runtime dependencies.

---

## ADR-T02: Total Elimination of the `<UNK>` (Unknown) Token

### Context
Traditional NLP tokenizers emit an out-of-vocabulary (`<UNK>`) token whenever they encounter a character not present in their pre-built dictionary.

### Decision
**Completely forbid and eliminate the `<UNK>` token** from ChakrView's vocabulary.

### Reasoning & Trade-offs
- **Information Conservation**: The moment a tokenizer emits `<UNK>`, the underlying semantic content is permanently annihilated. Downstream neural layers receive zero signal regarding the original input.
- **Lossless Reconstruction Guarantee**: By maintaining the 256 raw byte tokens as the foundational base, any rare script, emoji, technical glyph, or corrupted byte string decomposes gracefully into its constituent UTF-8 bytes.
- **Mathematical Reversibility**: Eliminating `<UNK>` ensures that $\text{Decode}(\text{Encode}(S)) \equiv S$ for 100% of Unicode strings.

---

## ADR-T03: Devanagari Grapheme Preservation in Pre-Tokenization

### Context
In the Devanagari script (`U+0900`–`U+097F`), an orthographic unit (*akshara*) consists of a consonant combined with vowel signs (matras), halants/viramas, anusvara, or nuktas. Standard Western tokenizers often treat combining marks as isolated units or split words across multi-byte boundaries, breaking the visual and semantic integrity of Hindi words.

### Decision
Implement **Grapheme-Aware Pre-Tokenization** that keeps Devanagari character clusters unified within regex boundaries (`[\u0900-\u097F]+`), preventing combining marks from being orphaned from their base glyphs.

### Reasoning & Trade-offs
- **Linguistic Integrity**: A matra (such as `ा`, `ि`, `ी`) has no independent linguistic existence without its base consonant. Splitting them forces the language model to learn redundant transition dynamics across separated byte fragments.
- **Equitable Compression**: Allocating dedicated merge steps for frequent Devanagari syllables lowers Hindi fertility from $> 4.5$ tokens/word (under naive Western tokenizers) to $\le 1.8$ tokens/word, removing the "linguistic tax" imposed on Indian languages.

---

## ADR-T04: Single-Digit Splitting for Numeric & Arithmetic Robustness

### Context
Language models frequently fail at basic arithmetic (addition, subtraction, multiplication) because tokenizers merge arbitrary multi-digit combinations (e.g., `1234` $\to$ `[123]`, `[4]` vs `[12]`, `[34]`), causing the model to learn inconsistent positional representations for identical numeric digits.

### Decision
Enforce **Single-Digit Tokenization** in the pre-tokenizer: every integer string is strictly split into individual digits ($0, 1, 2, 3, 4, 5, 6, 7, 8, 9$). Currency symbols and decimal points are isolated as independent tokens.

### Reasoning & Trade-offs
- **Arithmetic Generalization**: Representing numbers digit-by-digit aligns directly with column-wise addition and arithmetic algorithms, enabling language models to generalize to unseen multi-digit computations.
- **Sequence Overhead**: A 6-digit number consumes 6 tokens rather than 1 or 2 tokens. For a general language model, this is an acceptable trade-off to ensure numeric reliability and eliminate arithmetic hallucination.

---

## ADR-T05: Whitespace and Indentation Encoding for Code and Script Boundaries

### Context
Source code (especially Python) relies on exact indentation levels. Naive tokenizers collapse whitespace, strip leading/trailing spaces, or merge spaces with adjacent alphanumeric characters, causing syntax corruption in generated code and inserting phantom spaces during script transitions (e.g., between English and Devanagari).

### Decision
1. Use uniform **Leading-Space Byte Encoding** (binding space to the start of words).
2. Explicitly provide dedicated multi-space tokens for **2-space (`  `)** and **4-space (`    `)** indentation blocks.
3. Preserve raw newlines (`\n`, `0x0A`) without merging across alphanumeric boundaries.

### Reasoning & Trade-offs
- **Code Indentation Fidelity**: Python requires exact 4-space or 2-space indentation. Dedicated multi-space tokens preserve indentation structure without inflating sequence length.
- **Cross-Script Cleanliness**: Leading-space binding ensures that transitions between Latin and Devanagari (e.g., *"Python mein एक function"*) do not insert double spaces or collapse word boundaries.

---

## ADR-T06: Minimalist Special Token Set for Causal Decoder

### Context
Many tokenization frameworks define dozens of special tokens (`<MASK>`, `<CLS>`, `<SEP>`, `<SENT_START>`, `<NL>`), which add unnecessary complexity to pure autoregressive causal language models.

### Decision
Limit the special token set in ChakrView v0.1 to **exactly three tokens**:
1. `<BOS>` (Token ID 0): Beginning of Sequence.
2. `<EOS>` (Token ID 1): End of Sequence.
3. `<PAD>` (Token ID 2): Padding token for batch alignment.

### Reasoning & Trade-offs
- **Simplicity**: Pure causal language models generate text autoregressively from left to right. They do not require masked language modeling tokens (`<MASK>`) or cross-encoder separators (`<SEP>`).
- **Zero Redundancy**: Newlines are handled natively by the raw byte `0x0A` and `<UNK>` is eliminated by byte-level fallback.

---

## ADR-T07: Vocabulary Budget — Adopting Provisional $V = 4096$ for Chakr-Micro

### Context
In ChakrView, the input embeddings and output projection are weight-tied ($W_{\text{head}} = W_E^T$). The vocabulary size $V$ directly scales the embedding parameter count:
$$P_{\text{embed}} = V \times d = V \times 192$$
For Chakr-Micro (fixed non-embedding layer parameters $P_{\text{layers}} \approx 2.66\text{M}$):
- $V = 2048 \implies P_{\text{embed}} = 0.39\text{M}$ ($12.9\%$ of model)
- $V = 4096 \implies P_{\text{embed}} = 0.79\text{M}$ ($22.8\%$ of model)
- $V = 8192 \implies P_{\text{embed}} = 1.57\text{M}$ ($37.2\%$ of model)
- $V = 16384 \implies P_{\text{embed}} = 3.15\text{M}$ ($54.2\%$ of model)

### Decision
Confirm **$V = 4096$ as the recommended provisional vocabulary size** for Chakr-Micro, while retaining $V = 8192$ as an unfrozen experimental candidate for empirical testing.

### Reasoning & Trade-offs
- **Pareto Optimal Parameter Ratio**: At $V = 4096$, embeddings consume $22.8\%$ of model capacity, leaving $77.2\%$ for transformer layer reasoning. At $V = 16384$, embeddings would consume $54.2\%$ of the model, starving the actual reasoning blocks.
- **Sufficient Subword Capacity**: $4096$ tokens provide 256 base bytes, 3 special tokens, $\approx 1,200$ Devanagari subwords, $\approx 1,800$ English/Hinglish subwords, and $\approx 400$ code/math tokens. This achieves an estimated Hindi compression of $1.5\text{--}1.8$ tokens/word, allowing $\approx 300$ Hindi words to fit inside a 512-token context.
- **CPU Softmax Speed**: Computing cross-entropy loss and logits over 4,096 classes on a CPU takes half the time and memory compared to 8,192 classes.

---

## ADR-T08: Tokenizer Training Corpus Distribution Specification

### Context
A tokenizer learns subword merges based on frequency statistics in its training corpus. If the corpus is dominated by English, Devanagari subwords will be severely under-allocated.

### Decision
Mandate the following proportional distribution for the future tokenizer training corpus:
- **Hindi (Devanagari)**: 35.0%
- **English (General & Technical)**: 25.0%
- **Hinglish (Romanized Hindi)**: 15.0%
- **Mixed Hindi-English**: 10.0%
- **Source Code (Python, C, etc.)**: 10.0%
- **Numeric & Mathematical**: 3.0%
- **URLs, Paths & Technical**: 2.0%

### Reasoning & Trade-offs
- **Equitable Representation**: Allocating $60\%$ of total corpus representation to Indian linguistic expressions (Devanagari + Hinglish + Mixed) guarantees that Hindi and Hinglish receive sufficient merge budget to achieve optimal compression.
- **Code & Numeric Competence**: Reserving $15\%$ for code, math, and technical paths prevents punctuation and indentation degradation.

---

## ADR-T09: Absolute Reversibility & Lossless Decode Standard

### Context
Many real-world tokenizers suffer from subtle decoding bugs: trailing whitespace stripping, normalization drift (e.g., converting full-width punctuation or changing NFKD forms), or dropping unassigned bytes.

### Decision
Enforce a **Zero-Tolerance Lossless Reconstruction Standard**:
$$\text{Decode}(\text{Encode}(S)) \equiv S, \quad \forall S$$
Any tokenizer implementation that alters a single character or byte during round-trip encoding/decoding fails validation.

---

## Step 2 Frozen vs Unfrozen Decisions

### Frozen for Step 2:
- **Tokenization Paradigm**: Byte-Level Byte Pair Encoding (BBPE) with 256 base byte tokens.
- **No `<UNK>` Token**: Absolute elimination of out-of-vocabulary fallback; raw byte representation for unknown characters.
- **Special Token Set**: Exactly 3 tokens: `<BOS>` (0), `<EOS>` (1), `<PAD>` (2).
- **Numeric Strategy**: Single-digit splitting for all integers.
- **Devanagari Strategy**: Grapheme cluster preservation in pre-tokenization regex.
- **Whitespace Contract**: Lossless whitespace with dedicated indentation tokens (2-space, 4-space, raw `\n`).
- **Reversibility Standard**: 100% bit-exact round-trip accuracy.

### Unfrozen (Awaiting Empirical Validation in Step 3):
- **Final Vocabulary Size $V$**: $V = 4096$ vs $V = 8192$ (to be finalized after empirical compression benchmarks on the training corpus).
- **Exact Regex Specification**: Specific lookahead/lookbehind patterns for complex Indic conjunct ligatures.
- **Corpus Source Manifest**: Exact file sources and raw datasets for the tokenizer training corpus.
- **Merge Table Generation**: Specific merge hierarchy generated during offline training.

---

## Implementation Prerequisite Rule

> **MANDATORY RULE**: No tokenizer implementation should begin until this specification and decision record have been formally reviewed and approved by project leadership.

---
*End of Tokenizer Architecture Decision Records*
