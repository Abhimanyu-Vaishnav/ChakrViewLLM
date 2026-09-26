# Tokenizer Architecture Decision Records (ADR) — ChakrView Step 2

**Project**: ChakrView  
**Phase**: Step 2 — Tokenizer Research & Architectural Specification  
**Status**: ACTIVE & RATIFIED  

---

> ### MANDATORY ENGINEERING RULE
> **No tokenizer implementation should begin until this specification and decision log have been formally reviewed and approved.**  
> *Do not implement code, do not install new packages, do not download external datasets, and do not train any model during Step 2.*

> ### CRITICAL ARCHITECTURAL INVARIANT
> **"Tokenizer correctness takes precedence over compression."**  
> *The tokenizer must never trade away exact reconstruction merely to reduce token count. Preprocessing and pre-tokenization must remain strictly lossless.*

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
Adopt **Byte-Level Byte Pair Encoding (BBPE) with Grapheme-Aware Pre-Tokenization** as the provisional core tokenization algorithm for ChakrView v0.1.  
> **STATUS**: **BBPE is provisionally recommended and requires empirical validation before implementation is frozen.**

### Reasoning & Trade-offs
- **Zero Out-of-Vocabulary (OOV) Vulnerability**: Starting from the 256 fundamental byte values (`0x00`–`0xFF`) ensures that any arbitrary byte sequence (seen or unseen, valid or malformed UTF-8) can always be tokenized.
- **Context Preservation**: Pure character-level or byte-level modeling (without merges) expands sequence length by $3.5\times$ to $4.5\times$. In Chakr-Micro ($T=512$), character modeling reduces usable context to $\approx 110$ words, severely crippling multi-turn interaction. Under planning hypotheses, BBPE aims to compress English to $\approx 1.2$ tokens/word and Hindi to $\le 1.8$ tokens/word (to be validated empirically on the trained tokenizer).
- **CPU Inference Speed [Literature-Derived Expectation / Unverified Engineering Hypothesis]**: BBPE uses deterministic greedy prefix-tree (trie) matching, which is expected based on published literature to be faster on CPUs than SentencePiece Unigram's dynamic programming (Viterbi) search. This represents an unverified hypothesis until measured on a ChakrView benchmark.
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
Implement **Lossless Grapheme-Aware Pre-Tokenization** that keeps Devanagari character clusters unified within regex boundaries (`[\u0900-\u097F]+`), preventing combining marks from being orphaned from their base glyphs.

**Lossless Invariant**: Preprocessing must be completely lossless: `decode(encode(text)) == text` for all supported inputs. Do NOT perform irreversible Unicode transformations merely for normalization convenience. Explicit test cases must include:
- Combining marks
- Virama/halant
- Zero-Width Joiner (ZWJ, `U+200D`)
- Zero-Width Non-Joiner (ZWNJ, `U+200C`)
- Indic conjuncts (complex ligatures)
- Variation selectors (emoji and script selectors `U+FE00`–`U+FE0F`).

### Reasoning & Trade-offs
- **Linguistic Integrity**: A matra (such as `ा`, `ि`, `ी`) has no independent linguistic existence without its base consonant. Splitting them forces the language model to learn redundant transition dynamics across separated byte fragments.
- **Equitable Compression [Target Planning Hypothesis]**: Allocating dedicated merge steps for frequent Devanagari syllables is targeted to lower Hindi fertility from $> 4.5$ tokens/word (under naive Western tokenizers) to $\le 1.8$ tokens/word. This represents a target hypothesis to be validated empirically, not a measured ChakrView result.

---

## ADR-T04: Numeric Tokenization Strategy (UNFROZEN)

### Context
Language models frequently struggle with multi-digit arithmetic. Tokenizing numbers into multi-digit chunks or single digits presents trade-offs between arithmetic reasoning fidelity and sequence length expansion.

### Decision
**Numeric tokenization strategy is NOT frozen** and will be determined empirically through benchmark testing in Step 3.

#### Proposed Numeric Tokenization Experiment:
Evaluate three candidate strategies:
- **Candidate A: Individual Digits (`\d`)**: Strict single-digit splitting. Maximizes arithmetic generalization and positional alignment; sequence length increases linearly with digit count.
- **Candidate B: Common Two-Digit Chunks (`\d{1,2}`)**: Allows 2-digit pairs (e.g., `00`–`99`). Reduces sequence length for timestamps, years, and large numbers while bounding subword explosion.
- **Candidate C: Normal BPE Handling**: Digits are merged according to frequency statistics in the corpus without special digit-splitting constraints.

The final decision must be strictly benchmark-driven, evaluating arithmetic accuracy vs sequence length trade-off.

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
Retain **2048, 4096, 8192, and 16384 as candidate vocabulary sizes**, with **$V = 4096$ as the provisional planning candidate for Chakr-Micro**, subject to empirical benchmarking.

The final selection among candidate sizes will be determined strictly by measured empirical metrics:
- **tokens/character**
- **tokens/word**
- **bytes/token**
- **corpus compression ratio**
- **Hindi efficiency**
- **English efficiency**
- **Hinglish efficiency**
- **code efficiency**
- **number efficiency**
- **vocabulary memory overhead**
- **tokenizer encode/decode latency on CPU**

### Reasoning & Planning Hypotheses
- **Pareto Optimal Parameter Ratio**: At $V = 4096$, embeddings consume $22.8\%$ of model capacity, leaving $77.2\%$ for transformer layer reasoning. At $V = 16384$, embeddings would consume $54.2\%$ of the model, starving the actual reasoning blocks.
- **Hypothesized Subword Capacity**: $4096$ tokens may provide room for $\approx 1,200$ Devanagari subwords, $\approx 1,800$ English/Hinglish subwords, and $\approx 400$ code/math tokens alongside the 256 base bytes. **Notice**: This breakdown is a planning hypothesis only; the actual merge vocabulary will be learned empirically from the corpus and audited post-hoc.
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
> **CRITICAL ARCHITECTURAL INVARIANT**:
> **"Tokenizer correctness takes precedence over compression."**  
> *The tokenizer must never trade away exact reconstruction merely to reduce token count.*

Any tokenizer implementation that alters a single character or byte during round-trip encoding/decoding fails validation.

---

## Step 2 Frozen vs Unfrozen Decisions

### FROZEN:
- **byte-level fallback** (all 256 fundamental byte primitives present)
- **no `<UNK>`** (elimination of out-of-vocabulary fallback; raw byte representation)
- **lossless round-trip requirement** ($\text{Decode}(\text{Encode}(S)) \equiv S$)
- **causal tokenizer output** (strict deterministic integer token stream for causal LM)
- **special-token IDs** (`<BOS>`: 0, `<EOS>`: 1, `<PAD>`: 2)

### UNFROZEN:
- **final vocabulary size** (candidate sizes 2048, 4096, 8192, 16384 evaluated via empirical benchmark)
- **numeric strategy** (Candidate A: individual digits vs Candidate B: two-digit chunks vs Candidate C: normal BPE)
- **exact grapheme pre-tokenization rules** (fine-tuned lookahead/lookbehind patterns for complex Indic conjuncts and variation selectors)
- **corpus composition** (manifest of raw datasets and domain balance)
- **merge ranking** (specific merge hierarchy produced during offline training)
- **tokenizer implementation strategy** (custom lightweight trie vs native C wrapper)
- **optimization/trie implementation** (memory-mapped lookup vs flat array vs hash table)


---

## Implementation Prerequisite Rule

> **MANDATORY RULE**: No tokenizer implementation should begin until this specification and decision record have been formally reviewed and approved by project leadership.

---
*End of Tokenizer Architecture Decision Records*
