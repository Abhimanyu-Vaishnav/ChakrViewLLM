# ChakrView Tokenizer Specification (Chakr-Tokenizer v0.1)

**Document Version**: 0.1.0  
**Project Phase**: Step 2 — Tokenizer Research & Architectural Specification  
**Status**: APPROVED RESEARCH SPECIFICATION (IMPLEMENTATION FROZEN)  
**Author**: ChakrView Core Research & Architecture Team  

---

> ### MANDATORY ENGINEERING RULE
> **No tokenizer implementation should begin until this specification has been formally reviewed and approved.**  
> *Do not implement code, do not install new packages, do not download external datasets, and do not train any model during Step 2.*

> ### CRITICAL ARCHITECTURAL INVARIANT
> **"Tokenizer correctness takes precedence over compression."**  
> *The tokenizer must never trade away exact reconstruction merely to reduce token count. Preprocessing and pre-tokenization must remain strictly lossless.*

---

## 1. Goals & Strategic Vision

The tokenizer sits at the boundary between raw multimodal human thought (text, symbols, code) and the numeric vector space of the **ChakrView** neural core. 

ChakrView is an indigenous AI system designed for extreme efficiency on modern as well as low-resource hardware, older CPUs, and edge microprocessors. In small-scale models with constrained context windows (such as Chakr-Micro with context length $T = 512$), tokenizer design directly determines:
1. **Effective Context Capacity**: Shorter token sequences mean more semantic content fits within the 512-token context.
2. **Computational FLOPs per Word**: Generating fewer tokens per thought reduces total inference latency on CPU.
3. **Parameter Allocation**: Because embeddings and the unembedding head are weight-tied ($W_{\text{head}} = W_E^T$), vocabulary size $V$ linearly dictates embedding memory ($V \times d_{model}$). An oversized vocabulary consumes parameter capacity that should belong to cognitive depth.
4. **Indigenous Linguistic Equality**: Standard Western tokenizers severely penalize non-Latin scripts (e.g., Devanagari), splitting a single Hindi word into 6 to 10 tokens. ChakrView rejects this linguistic tax.

---

## 2. Comprehensive Requirements

The tokenizer must robustly, losslessly, and efficiently tokenize:

1. **Standard English**: High compression, preservation of contractions, morphology, and idioms.
2. **Hindi in Devanagari Script**: Unicode block `U+0900`–`U+097F`, preserving consonant clusters, vowel matras, halants/viramas, anusvara, and nuktas without orthographic mutilation.
3. **Hinglish / Romanized Hindi**: Phonetically variable colloquial Hindi written in Latin script (e.g., *"mera naam abhimanyu hai"*, *"mera naam abhimanyu h"*, *"mujhe coding seekhni hai"*).
4. **Code-Switched & Mixed Text**: Seamless intra-sentence switching between Devanagari and Latin (e.g., *"mujhe Python mein एक function banana hai"*).
5. **Broader Indic Languages**: Structural readiness for Bengali, Tamil, Telugu, Marathi, Gujarati, Punjabi, Kannada, Malayalam, and Odia.
6. **Numbers & Arithmetic**: Mathematical numbers, decimals, dates, currency, and alphanumeric model codes (e.g., `123456`, `₹50000`, `3.14159`, `2026-09-26`, `13th`, `13900H`).
7. **Punctuation & Symbols**: Ascii and Unicode punctuation, mathematical operators ($\approx, \sum, \int, \le, \ge, \pm, \pi$), and currency glyphs (₹, $, €, £).
8. **URLs, Web Addresses, & File Paths**: `https://...`, `d:\Project\ChakrView`, `/usr/local/bin`, query params, and directory delimiters.
9. **Programming Languages & Source Code**: Syntactically sensitive languages (Python, C, Rust, JavaScript, Bash) requiring exact indentation (2-space, 4-space, tabs), punctuation, and casing preservation (`snake_case`, `camelCase`, `PascalCase`).
10. **Arbitrary Unknown Unicode**: Emojis, rare scripts, and corrupted bytes without crashing or producing out-of-vocabulary (`<UNK>`) failure.
11. **Deterministic & Reversible**: $\text{Decode}(\text{Encode}(\text{text})) \equiv \text{text}$ bit-exact for 100% of Unicode input strings.
12. **Extreme Runtime Efficiency**: Low-overhead tokenization in pure Python/C with minimal memory allocation on resource-constrained CPUs.

---

## 3. Methodological Disambiguation

To preserve scientific rigor throughout this specification, all technical statements are partitioned into four explicit categories:

- **[KNOWN FACT]**: Mathematically proven properties, verified Unicode standards, or established algorithmic facts.
- **[ENGINEERING HYPOTHESIS]**: Plausible technical deductions based on theory, requiring empirical benchmark confirmation in Step 3.
- **[ASSUMPTION]**: Stated baseline assumptions regarding target hardware, user behavior, or operating conditions.
- **[PROPOSED EXPERIMENT]**: Specific, empirical test protocols designed to validate hypotheses before freezing code.

---

## 4. Evaluation of Candidate Tokenization Approaches

Seven candidate tokenization paradigms were evaluated against ChakrView's requirements:

| Evaluation Dimension | A. Byte-Level BPE (BBPE) | B. Character-Level Tokenization | C. Standard BPE (with UNK) | D. Unigram LM (SentencePiece) | E. WordPiece | F. Subword + Byte Fallback | G. Grapheme-Aware Akshara BPE |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Vocabulary Efficiency** | High (compact, no wasted rare chars) | Very High (minimal vocab) | Moderate | High (optimal entropy) | Moderate | High | High (linguistically compact) |
| **Tokens / Sentence (Compression)** | Excellent (tunable by merges) | **Extremely Poor** (4–5x longer sequences) | Excellent | Superior | Good | Excellent | Superior for Indic scripts |
| **Hindi Devanagari Efficiency** | Good (with targeted merges) | Poor (3–4 tokens per character) | Poor on tail words | Excellent | Moderate | Good | **Optimal** (respects akshara boundaries) |
| **Romanized Hindi (Hinglish)** | Excellent (captures phonetic roots) | Very Poor | Good | Good | Good | Good | Good |
| **English Efficiency** | Excellent | Very Poor | Excellent | Excellent | Excellent | Excellent | Excellent |
| **Mixed Hindi-English Efficiency** | Excellent (script-agnostic base) | Poor | Moderate | Good | Moderate | Excellent | Superior |
| **Programming Code Efficiency** | Excellent (preserves exact bytes) | Very Poor | Moderate | Good | Moderate | Good | Good |
| **Unicode Coverage** | **100% Complete** (all 256 bytes) | Incomplete (bounded set) | Incomplete | High | Incomplete | **100% Complete** | **100% Complete** |
| **Unknown Token Behavior** | **Zero `<UNK>`** (pure byte fallback) | Catastrophic `<UNK>` on tail | Emits `<UNK>` | Emits `<UNK>` or byte-fallback | Emits `<UNK>` | **Zero `<UNK>`** | **Zero `<UNK>`** |
| **Tokenizer Memory (Runtime)** | Very Low (< 5 MB trie/hash) | Negligible (< 100 KB) | Low (< 5 MB) | Moderate (Viterbi arrays) | Low | Low (< 5 MB) | Low (< 6 MB) |
| **Implementation Complexity** | Low to Moderate | **Trivial** | Low | High (DP Viterbi search) | Moderate | Moderate | Moderate |
| **Training Complexity** | Moderate ($O(N \log V)$) | Negligible | Moderate | High (EM pruning iterations) | Moderate | Moderate | Moderate |
| **Inference Speed (CPU)** | Fast [Literature expectation; unverified by ChakrView benchmark] | Fast | Fast | Slower (Viterbi search) [Hypothesis] | Fast | Fast | Fast |
| **Edge / Embedded Suitability** | **Exceptional** (bare-metal C friendly) | High | High | Low (complex dependencies) | Moderate | High | High |
| **Reversibility (Lossless)** | **100% Bit-Exact** | Lossy on unseen chars | Lossy on `<UNK>` | Lossy unless byte-fallback | Lossy on `<UNK>` | **100% Bit-Exact** | **100% Bit-Exact** |
| **Extensibility** | High | Low | Low | Moderate | Low | High | High |

---

## 5. Candidate Analysis & Architectural Recommendation

### Detailed Paradigm Assessment

1. **Character-Level Tokenization (Approach B)**:
   - *[KNOWN FACT]*: While offering a tiny vocabulary ($V \approx 1000$), character-level tokenization expands English text by $\approx 4.5\times$ and Hindi Devanagari text by $\approx 3.5\times$.
   - *[KNOWN FACT]*: In Chakr-Micro with context length $T = 512$, a character-level model can only fit $\approx 110$ words in context. This cripples multi-turn conversation, reasoning chains, and code comprehension.
   - *Conclusion*: **Rejected** for primary language modeling.

2. **Standard BPE with `<UNK>` Fallback (Approach C) & WordPiece (Approach E)**:
   - *[KNOWN FACT]*: Any tokenizer that lacks a byte-level base vocabulary must emit out-of-vocabulary (`<UNK>`) tokens when encountering unseen Unicode characters (rare regional scripts, math symbols, technical glyphs, or novel emojis).
   - *[KNOWN FACT]*: Once an `<UNK>` is emitted, semantic information is permanently destroyed, and lossless round-trip decoding is mathematically impossible.
   - *Conclusion*: **Rejected**. ChakrView mandates zero information loss.

3. **Unigram Language Model (Approach D)**:
   - *[KNOWN FACT]*: SentencePiece Unigram provides mathematically optimal compression and supports subword regularization during training.
   - *[ENGINEERING HYPOTHESIS / LITERATURE-DERIVED EXPECTATION]*: Unigram tokenization requires dynamic programming (Viterbi search) at inference time, which introduces runtime overhead on low-power CPUs compared to greedy matching. This represents an unverified expectation until measured on a ChakrView benchmark.
   - *Conclusion*: **Deferred to future research steps**; suboptimal for a lightweight indigenous v0.1 core.

4. **Byte-Level BPE with Grapheme-Aware Pre-Tokenization (Approaches A, F & G)**:
   - *[KNOWN FACT]*: Byte-Level BPE (BBPE) begins with 256 fundamental byte tokens (`0x00` through `0xFF`), ensuring that any arbitrary string of bytes can be represented without `<UNK>`.
   - *[ENGINEERING HYPOTHESIS]*: Standard Western BBPE pre-tokenizers (such as GPT-2/GPT-4 regexes) split words into arbitrary byte chunks, which can shatter Devanagari combining marks (matras, viramas) away from their base consonants before merges can form.
   - *[ENGINEERING HYPOTHESIS]*: Enhancing Byte-Level BPE with an **Indigenous Grapheme-Aware Pre-Tokenizer** prevents unnatural script splitting, ensures clean whitespace and code indentation handling, and guarantees 100% lossless byte-level fallback for unseen Unicode.

### Formal Recommendation for ChakrView v0.1
**Adopt: Chakr-BPETokenizer v0.1 (Indigenous Byte-Level BPE with Grapheme-Aware Pre-Tokenization)**.  
> **STATUS**: **BBPE is provisionally recommended and requires empirical validation before implementation is frozen.**

---

## 6. Detailed Domain Strategies

### 6.1 Unicode Strategy & Byte-Level Guarantee
1. **Base Alphabet Guarantee**:
   The vocabulary reserves the first 256 IDs strictly for raw byte values:
   $$\text{Token ID } b = b, \quad \forall b \in \{0x00, 0x01, \dots, 0xFF\}$$
2. **Zero Out-of-Vocabulary (`<UNK>`) Guarantee**:
   *[KNOWN FACT]*: Every valid or invalid Unicode string is representable as a sequence of UTF-8 bytes. Because all 256 byte values exist in the base vocabulary, **the tokenizer can encode any sequence of bytes without exception**.
3. **Lossless Preprocessing & Round-Trip Reversibility**:
   $$\text{Decode}(\text{Encode}(S)) \equiv S, \quad \forall S \in \text{Unicode Strings}$$
   **Strict Invariant**: Preprocessing must be completely lossless. Do NOT perform irreversible Unicode transformations merely for normalization convenience (e.g., stripping accents, lossy NFKD decompositions without exact reversibility).
   Explicit test cases must include:
   - Combining marks
   - Virama/halant
   - Zero-Width Joiner (ZWJ, `U+200D`)
   - Zero-Width Non-Joiner (ZWNJ, `U+200C`)
   - Indic conjuncts (complex ligatures)
   - Variation selectors (emoji and script selectors `U+FE00`–`U+FE0F`).

### 6.2 Hindi & Devanagari Strategy
Devanagari characters (`U+0900` to `U+097F`) represent orthographic syllables (*akshara*). In UTF-8, each Devanagari codepoint requires exactly 3 bytes (e.g., `क` = `0xE0 0xA4 0x95`).

1. **Grapheme Preservation in Pre-Tokenization**:
   Devanagari combines consonants with combining vowel signs (matras, e.g., `ा`, `ि`, `ी`), halants/viramas (`्`), and modifier signs (anusvara `ं`, visarga `ः`, nukta `़`).
   - *Rule*: The pre-tokenizer regex must prevent a combining mark from being detached from its base consonant.
   - Pattern match: `[\u0900-\u097F]+` matches contiguous Devanagari sequences as unified pre-tokens, preventing cross-script pollution.
2. **Corpus Merge Allocation**:
   - The tokenizer merge table must be trained on a corpus containing substantial high-quality Hindi text.
   - Frequent Devanagari characters (3 bytes) and high-frequency full syllables (6 to 9 bytes, e.g., `का`, `की`, `के`, `में`, `है`, `कर`) must be learned as single subword tokens.
   - *Planning Target Metric [Hypothesis; not a measured ChakrView result]*: Hindi Devanagari compression ratio is targeted at $\le 1.8$ tokens per word on standard Hindi text. This value is an engineering target to be validated on the trained tokenizer, not an established measurement.

### 6.3 Hinglish (Romanized Hindi) Strategy
Hinglish is characterized by phonetic spellings of Hindi grammar and vocabulary using the Latin alphabet (e.g., *"mera naam abhimanyu hai"*, *"mujhe coding seekhni hai"*).

1. **Challenges**:
   - High spelling variance: *"hai"* vs *"h"*, *"kya"* vs *"kyaa"*, *"zaroorat"* vs *"jarurat"*.
   - Frequent morphological suffixes: *-kar*, *-raha*, *-rahi*, *-enge*, *-gaya*, *-wali*.
2. **Strategy**:
   - The pre-tokenizer treats Latin words uniformly using case-aware subword segmenting.
   - High-frequency Hinglish functional words (*"hai"*, *"mein"*, *"nahi"*, *"karna"*, *"raha"*, *"aap"*, *"mera"*, *"hum"*) must be present as whole subwords.
   - Subword stems and morphological suffixes (*"seekh"*, *"ni"*, *"samajh"*) naturally combine to represent phonetic variants with minimal token consumption.

### 6.4 Mixed & Code-Switched Text Strategy
Example: *"mujhe Python mein एक function banana hai"*

1. **Script Transition Invariance**:
   The pre-tokenizer segments text along script boundaries without consuming or mangling inter-script whitespace.
   - Segment 1 (Latin): `mujhe`
   - Segment 2 (Latin): ` Python`
   - Segment 3 (Latin): ` mein`
   - Segment 4 (Devanagari): ` एक`
   - Segment 5 (Latin): ` function`
   - Segment 6 (Latin): ` banana`
   - Segment 7 (Latin): ` hai`
2. **Whitespace Attachment**:
   Leading space is systematically bound to the beginning of each word regardless of script:
   `[Ġmujhe] [ĠPython] [Ġmein] [Ġएक] [Ġfunction] [Ġbanana] [Ġhai]`
   This guarantees that script transitions do not insert phantom spaces or concatenate cross-script words.

### 6.5 Source Code & Programming Language Strategy
Example:
```python
def calculate_sum(a, b):
    return a + b
```

1. **Whitespace & Indentation Preservation**:
   - Spaces must never be collapsed or stripped.
   - Pre-tokenization encodes multi-space sequences with dedicated indentation tokens:
     - 2-space token: `  ` (two consecutive spaces)
     - 4-space token: `    ` (four consecutive spaces)
   - This drastically compresses Python indentation levels while preserving exact column alignment.
2. **Newline Preservation**:
   - Newline `\n` is explicitly represented as byte `0x0A` and not merged across semantic words.
   - Multiple newlines (`\n\n`) can merge to represent paragraph and code block separations.
3. **Identifier Casing Preservation**:
   - The pre-tokenizer splits identifiers along casing boundaries:
     - `snake_case`: `calculate_sum` $\to$ `[calculate]`, `[_sum]`
     - `camelCase`: `calculateSum` $\to$ `[calculate]`, `[Sum]`
     - `PascalCase`: `CalculateSum` $\to$ `[Calculate]`, `[Sum]`
   - This allows code vocabulary to share subword representations with standard natural language tokens.

### 6.6 Numbers & Arithmetic Strategy (UNFROZEN)
Examples: `123456`, `₹50000`, `3.14159`, `2026-09-26`, `13th`, `13900H`

> **DECISION STATUS: UNFROZEN**. Numeric tokenization strategy is NOT frozen and must be decided based on empirical benchmark results in Step 3.

#### Proposed Numeric Tokenization Experiment:
An empirical experiment will be conducted during tokenizer training to evaluate three candidate strategies:
- **Candidate A: Individual Digits (`\d`)**: Strict single-digit splitting. Maximizes arithmetic generalization and positional alignment; sequence length increases linearly with digit count.
- **Candidate B: Common Two-Digit Chunks (`\d{1,2}`)**: Allows 2-digit pairs (e.g., `00`–`99`). Reduces sequence length for timestamps, years, and large numbers while bounding subword explosion.
- **Candidate C: Normal BPE Handling**: Digits are merged according to frequency statistics in the corpus without special digit-splitting constraints.

The final decision must be strictly benchmark-driven, evaluating arithmetic accuracy vs sequence length trade-off.

#### General Punctuation Decoupling:
- Currency symbols (e.g., `₹`, `$`) and decimal points (`.`) are isolated from digits:
  `₹50000` $\to$ `[₹]`, followed by segmented digits
  `3.14159` $\to$ `[3]`, `[.]`, followed by fractional digits
- Suffixes like `th` in `13th` or `H` in `13900H` are separated:
  `13th` $\to$ digits + `[th]`
  `13900H` $\to$ digits + `[H]`

### 6.7 URLs, Web Addresses & System Paths Strategy
Examples:
- URL: `https://github.com/ChakrView/core?query=1`
- Windows Path: `d:\Project\ChakrView\brain`
- Unix Path: `/usr/local/bin/python`

1. **Punctuation Segregation**:
   - URL and path delimiters (`/`, `\`, `:`, `.`, `?`, `=`, `&`) are treated as independent punctuation tokens.
   - They are never glued across directory names or domain components:
     `https://` $\to$ `[https]`, `[:]`, `[/]`, `[/]`
     `d:\Project` $\to$ `[d]`, `[:]`, `[\\]`, `[Project]`
2. **Subword Path Reusability**:
   - Path components like `Project`, `ChakrView`, `bin`, `local` match standard dictionary subwords, preventing vocabulary explosion on file paths.

### 6.8 Special-Token Specification
A causal language model requires only a strictly minimal set of special tokens. Including redundant tokens introduces optimization noise and complicates masking logic.

| Token | Name | Token ID | Purpose in Causal LM | Included in v0.1? | Justification |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `<BOS>` | Beginning of Sequence | 0 | Conditions generation; marks sequence start | **YES** | Mandatory for prompt demarcation |
| `<EOS>` | End of Sequence | 1 | Signals termination of autoregressive generation | **YES** | Mandatory to stop infinite generation loops |
| `<PAD>` | Padding Token | 2 | Pads unequal sequences within a batch | **YES** | Mandatory for batched CPU/GPU training |
| `<UNK>` | Unknown Token | — | Replaces unknown inputs | **NO** | **Forbidden**: Byte fallback renders `<UNK>` obsolete |
| `<SEP>` | Separator Token | — | Separates cross-encoder segments | **NO** | Unnecessary for pure autoregressive decoder |
| `<NL>` | Newline Token | — | Encodes line breaks | **NO** | Redundant; raw `\n` (`0x0A`) natively handles line breaks |

**Total Special Tokens in v0.1**: Exactly 3 (`<BOS>`, `<EOS>`, `<PAD>`).

---

## 7. Vocabulary Size Analysis & Parameter Impact

### 7.1 Mathematical Model of Vocabulary Impact on Chakr-Micro
In Chakr-Micro (Step 1 Specification):
- Layers: $L = 6$
- Hidden Dimension: $d = 192$
- Weight Tying: **Enabled** ($W_{\text{head}} = W_E^T$)
- Fixed Layer Parameters: $P_{\text{layers}} = \mathbf{2,656,704}$ parameters (~2.66M)
- Embedding / Unembedding Parameters: $P_{\text{embed}} = V \cdot d = V \cdot 192$
- Total Model Parameters: $P_{\text{total}} = P_{\text{layers}} + (V \cdot 192)$
- Final Output Logits Projection Compute: $2 \cdot B \cdot T \cdot (d \cdot V)$ FLOPs per step.

### 7.2 Quantitative Trade-Off Matrix

> **Note on Estimates**: All fertility values (tokens/word) and context capacities listed below are **[Planning Hypotheses / Pre-Benchmark Estimates]**. They do NOT represent measured ChakrView results and must be validated on an actual corpus in Step 3.

| Candidate Vocab Size ($V$) | Embedding Params ($V \times 192$) | Total Micro Params | Embedding % of Model | FP16 Weight RAM | INT8 Weight RAM | Output Logits FLOPs ($T=512$) | Estimated Hindi Tokens/Word | Context Capacity (Hindi Words in $T=512$) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **$V = 2,048$** | $393,216$ (~0.39M) | **$3.05\text{ M}$** | **$12.9\%$** | $6.10\text{ MB}$ | $3.05\text{ MB}$ | $0.40\text{ GFLOPs}$ | $\approx 2.4 - 2.8$ | $\approx 180 - 210$ words |
| **$V = 4,096$ (Provisional)**| $786,432$ (~0.79M) | **$3.44\text{ M}$** | **$22.8\%$** | $6.89\text{ MB}$ | $3.44\text{ MB}$ | $0.81\text{ GFLOPs}$ | $\approx 1.5 - 1.8$ | $\approx 280 - 340$ words |
| **$V = 8,192$** | $1,572,864$ (~1.57M) | **$4.23\text{ M}$** | **$37.2\%$** | $8.46\text{ MB}$ | $4.23\text{ MB}$ | $1.61\text{ GFLOPs}$ | $\approx 1.2 - 1.4$ | $\approx 360 - 420$ words |
| **$V = 16,384$** | $3,145,728$ (~3.15M) | **$5.80\text{ M}$** | **$54.2\%$** | $11.60\text{ MB}$ | $5.80\text{ MB}$ | $3.22\text{ GFLOPs}$ | $\approx 1.1 - 1.2$ | $\approx 420 - 460$ words |

### 7.3 Trade-Off Deductions & Vocabulary Planning Hypotheses
1. **$V = 2,048$**:
   - Small parameter count ($3.05\text{M}$); fast final projection.
   - High sequence fragmentation on Hindi and code under theoretical subword distributions.
2. **$V = 4,096$**:
   - *Current Planning Hypothesis*: Embeddings occupy $22.8\%$ of parameter budget.
   - *Hypothesized Subword Capacity*: May provide room for $\approx 1,200$ Devanagari subwords, $\approx 1,800$ English/Hinglish subwords, and $\approx 400$ code/syntax tokens alongside the 256 byte primitives. **Notice**: This breakdown is a planning hypothesis only; the actual merge vocabulary will be learned empirically from the corpus and audited post-hoc.
3. **$V = 8,192$**:
   - Retained as an unfrozen candidate for empirical testing.
4. **$V = 16,384$**:
   - Disproportionate parameter allocation ($54.2\%$ in embeddings) for a 3.44M parameter model.

### 7.4 Final Vocabulary Selection Protocol
The final selection among candidate sizes (2048, 4096, 8192, 16384) will be determined strictly by measured empirical metrics on the actual ChakrView tokenizer and corpus:
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

---

## 8. Proposed Tokenizer Training Corpus Composition

> **RULE**: Do NOT download or create this corpus in Step 2. This section defines the architectural proportion specification for subsequent phases.

To ensure balanced representation across all target domains, the tokenizer training corpus must be partitioned according to the following deliberate distribution:

```
┌────────────────────────────────────────────────────────┐
│      ChakrView Tokenizer Training Corpus (100%)        │
├───────────────────────────────┬────────────────────────┤
│ Domain / Category             │ Allocation Percentage  │
├───────────────────────────────┼────────────────────────┤
│ 1. Hindi in Devanagari        │ 35.0%                  │
│ 2. English (General & Tech)   │ 25.0%                  │
│ 3. Hinglish / Romanized Hindi │ 15.0%                  │
│ 4. Mixed Hindi-English        │ 10.0%                  │
│ 5. Source Code & Logic        │ 10.0%                  │
│ 6. Numeric & Mathematical     │ 3.0%                   │
│ 7. URLs, Paths & Technical    │ 2.0%                   │
└───────────────────────────────┴────────────────────────┘
```

### Detailed Category Sub-Allocations:
1. **Hindi Devanagari (35%)**: Formal prose, conversational dialogue, encyclopedic text, news, technical Hindi, and classic literature.
2. **English (25%)**: High-quality general prose, scientific articles, educational textbooks, and instruction datasets.
3. **Hinglish (15%)**: Colloquial chat, social discourse, phonetic variations, and mixed-mode conversational text.
4. **Mixed Hindi-English (10%)**: Intra-sentence code-switched technical tutorials, Indian educational notes, and technical forums.
5. **Source Code (10%)**: Clean, idiomatic Python, C, Rust, JavaScript, and Bash scripts emphasizing proper indentation, naming conventions, and docstrings.
6. **Numeric & Mathematical (3%)**: Arithmetic equations, tables, financial ledgers, date formats, scientific notation, and matrix listings.
7. **URLs, Paths & Technical (2%)**: Terminal commands, file directory listings, URLs, API endpoints, and configuration files (JSON, YAML).

---

## 9. Tokenizer Evaluation Suite (Benchmarking Dataset)

The following explicit test cases form the mandatory evaluation benchmark suite to evaluate any candidate tokenizer:

### Benchmark Test Cases

#### Category A: English Prose & Contractions
- `TC-ENG-01`: `"The quick brown fox jumps over the lazy dog."`
- `TC-ENG-02`: `"We wouldn't have couldn't shouldn't don't, it's 100% fine."`
- `TC-ENG-03`: `"Hardware-aware execution enables low-resource language modeling."`

#### Category B: Hindi in Devanagari
- `TC-HIN-01`: `"नमस्ते दुनिया, आप कैसे हैं?"`
- `TC-HIN-02`: `"चक्रव्यूह एक स्वदेशी, कम संसाधन वाला कृत्रिम बुद्धिमत्ता मॉडल है।"`
- `TC-HIN-03`: `"संयुक्त वर्ण: क्ष, त्र, ज्ञ, श्र, द्व, द्ध, ष्ट, ण्ड, न्म, प्र।"`

#### Category C: Hinglish / Romanized Hindi
- `TC-HNG-01`: `"mera naam abhimanyu hai"`
- `TC-HNG-02`: `"mera naam abhimanyu h"`
- `TC-HNG-03`: `"mujhe coding seekhni hai aur AI model banana hai"`
- `TC-HNG-04`: `"kya aap meri madad kar sakte ho bhai?"`

#### Category D: Mixed Hindi-English (Code-Switching)
- `TC-MIX-01`: `"mujhe Python mein एक function banana hai"`
- `TC-MIX-02`: `"Yeh machine learning model CPU par bohot fast run karta hai."`
- `TC-MIX-03`: `"Aaj ka topic hai 'Linear Algebra' aur uske applications."`

#### Category E: Programming Code & Indentation
- `TC-COD-01`:
```python
def calculate_sum(a, b):
    # Sum of two numbers
    result = a + b
    return result
```
- `TC-COD-02`:
```c
#include <stdio.h>
int main() {
    printf("ChakrView Core\n");
    return 0;
}
```
- `TC-COD-03`: `for i in range(10): print(f"idx: {i*2}")`

#### Category F: Numbers, Currencies & Formats
- `TC-NUM-01`: `"123456"`
- `TC-NUM-02`: `"₹50000 and $1250.75"`
- `TC-NUM-03`: `"The constant pi is approximately 3.1415926535."`
- `TC-NUM-04`: `"Date: 2026-09-26, CPU: 13th Gen i9-13900H with 32GB RAM."`

#### Category G: URLs, File Paths & Commands
- `TC-PTH-01`: `"https://github.com/ChakrView/ChakrView?branch=main#readme"`
- `TC-PTH-02`: `"d:\Project\ChakrView\brain\core.py"`
- `TC-PTH-03`: `"/usr/local/bin/python3 -m venv .venv"`

#### Category H: Rare Unicode, Emojis & Mathematical Notation
- `TC-UNI-01`: `"Emojis: 🚀 🧠 🇮🇳 ⚡ 💻"`
- `TC-UNI-02`: `"Math: ∀x ∈ ℝ, ∑_{i=1}^n x_i ≤ √n ‖x‖_2"`
- `TC-UNI-03`: `"Tamil: வணக்கம், Bengali: নমস্কার, Gujarati: નમસ્તે"`

#### Category I: Explicit Adversarial & Stress Testing Suite
- `TC-ADV-01 (Hindi Combining Marks)`: `"कौशाम्बी, त्र्यंबकेश्वर, कुंडलियाँ, पूँछ, अँधेरा, ऋग्वेद, ॐ"`
- `TC-ADV-02 (Sanskrit Conjuncts)`: `"सत्त्व, दग्ध, बुद्ध, उष्ट्र, स्त्रोत, कार्त्तिकेय, वाग्देवी"`
- `TC-ADV-03 (ZWJ & ZWNJ)`: `"क्‍ + य = क्‍य (ZWJ: U+200D), क + ् + ‌ + य = क्‌य (ZWNJ: U+200C)"`
- `TC-ADV-04 (Complex Emojis & ZWJ Sequences)`: `"Family: 👨‍👩‍👧‍👦, Flag: 🇮🇳, Modifiers: 👍🏽, 👩‍💻"`
- `TC-ADV-05 (Uncommon Unicode)`: `"Runes: ᚠᚢᚦᚨᚱᚲ, Tibetan: ཨོཾ་མ་ཎི་པདྨེ་ཧཱུྃ, Coptic: ⲁⲃⲅⲇ"`
- `TC-ADV-06 (Mixed Scripts Intertwined)`: `"ChakrView-चक्रव्यूह_core-v0.1"`
- `TC-ADV-07 (Hindi + English + Numbers Intertwined)`: `"model-2026 mein ₹50000 ka 13th version, latency < 15.4ms"`
- `TC-ADV-08 (Code Indentation Stress)`: `"    def f():\n        if True:\n            return 1"`
- `TC-ADV-09 (Tabs & Mixed Whitespace)`: `"\tdef foo():\n\t    x = 10\n  \t  y = 20"`
- `TC-ADV-10 (CRLF and LF Alternation)`: `"Line1\r\nLine2\nLine3\r\nLine4"`
- `TC-ADV-11 (Complex URLs & Query Strings)`: `"https://api.chakrview.ai/v1/search?q=%E0%A4%AD%E0%A4%BE%E0%A4%B0%E0%A4%A4&limit=10&auth=true#top"`
- `TC-ADV-12 (Windows Paths)`: `"d:\\Project\\ChakrView\\brain\\..\\data\\raw\\corpus.jsonl"`
- `TC-ADV-13 (Linux Paths)`: `"/var/log/syslog.1.gz; rm -rf /tmp/test*"`
- `TC-ADV-14 (Mathematical Symbols)`: `"∀x ∈ A: ∃y s.t. x ⊕ y = ∅ ∧ x ≢ y"`
- `TC-ADV-15 (Malformed / Truncated UTF-8 Byte Sequences)`: Raw invalid UTF-8 byte sequences (e.g. orphan continuation byte `0x80` or truncated 3-byte prefix `0xE0 0xA4`) safely representable as raw byte tokens without throwing runtime exceptions.

---

## 10. Quantitative Tokenizer Metrics

To objectively evaluate candidate tokenizers, the following metrics are formally defined:

### 1. Tokens Per Character (TPC)
$$\text{TPC} = \frac{\text{Total Tokens Generated}}{\text{Total Unicode Characters in Input}}$$
- Lower is better for compression; target for English: $\approx 0.22\text{--}0.28$; target for Hindi Devanagari: $\le 0.55$.

### 2. Tokens Per Word (TPW / Fertility)
$$\text{TPW} = \frac{\text{Total Tokens Generated}}{\text{Total Whitespace-Delimited Words}}$$
- Target for English: $\approx 1.15\text{--}1.30$
- Target for Hindi: $\le 1.80$
- Target for Hinglish: $\le 1.45$

### 3. Compression Ratio (CR)
$$\text{CR} = \frac{\text{Total UTF-8 Bytes in Raw Input}}{\text{Total Tokens Generated}}$$
- Higher is better (more bytes compressed per token). Target: $> 3.2$ bytes/token.

### 4. Round-Trip Reconstruction Accuracy (RTRA)
$$\text{RTRA} = \frac{\sum_{i=1}^N \mathbb{I}(\text{Decode}(\text{Encode}(S_i)) == S_i)}{N} \times 100\%$$
- **Passing Threshold: Exactly 100.00%**. Zero byte discrepancies permitted.

### 5. Unicode Coverage Ratio
$$\text{UCR} = \frac{\text{Count of Successfully Encoded Unicode Characters Without Error}}{\text{Total Unicode Characters Tested}} \times 100\%$$
- **Passing Threshold: Exactly 100.00%**. Zero `<UNK>` substitutions.

### 6. Whitespace Preservation Metric
$$\text{WPM} = \mathbb{I}(\text{CountLeadingSpaces}(\text{Decoded}) == \text{CountLeadingSpaces}(\text{Input}))$$
- **Passing Threshold: 100%**. Exact column indentation preserved across all code samples.

---

## 11. Exact Acceptance Criteria for Step 2

A tokenizer design satisfies Step 2 if and only if it complies with the following conditions:

1. **Complete Zero-Loss Round-Trip**: Bit-exact reversibility for all test strings in the evaluation suite.
2. **Zero Out-of-Vocabulary Tokens**: Absolute absence of `<UNK>` token emission on any valid or invalid UTF-8 sequence.
3. **Equitable Hindi Fertility**: Hindi text achieves fertility $\le 1.8$ tokens/word on standard Devanagari evaluation text.
4. **Exact Code & Whitespace Preservation**: No collapsing, stripping, or mutation of indentation spaces and newlines.
5. **Single-Digit Numeric Integrity**: All integer strings are segmented into individual digit tokens during pre-tokenization.
6. **Hardware-Aligned Vocabulary Budget**: Recommended vocabulary fits within the memory and FLOP constraints of Chakr-Micro without overwhelming embedding capacity.
7. **Strict Boundary Decoupling**: Tokenizer logic is self-contained with no dependency on neural network tensor operations.

---

## 12. Failure Modes & Mitigations

| Failure Mode | Root Cause | Impact | Mitigation Strategy |
| :--- | :--- | :--- | :--- |
| **Devanagari Grapheme Mutilation** | Naive whitespace regex splitting combining vowel matras or viramas from base consonants. | Model generates visually invalid glyphs or fails semantic comprehension. | Enforce atomic Devanagari regex blocks `[\u0900-\u097F]+` in pre-tokenization. |
| **Arithmetic Degradation** | Merging multi-digit numbers into arbitrary subword tokens (e.g., `1234` as single token). | Poor arithmetic generalization and failure on multi-digit addition/multiplication. | Enforce single-digit tokenization rule `\d` in pre-tokenizer regex. |
| **Python Indentation Destruction** | Merging spaces with adjacent alphanumeric characters or collapsing multiple spaces. | Fatal `IndentationError` in generated Python code. | Isolate whitespace into dedicated single, double, and quad space tokens (` `, `  `, `    `). |
| **Embedding Memory Bloat** | Selecting vocabulary $V \ge 16384$ for Chakr-Micro. | Over 50% of model parameters consumed by embedding lookup; slow CPU softmax. | Cap Chakr-Micro vocabulary at $V = 4096$ (provisional recommendation). |
| **Phantom Spaces on Script Switch** | Inconsistent whitespace prefixing across Latin and Devanagari tokens. | Generated text inserts double spaces or drops spaces between English and Hindi words. | Implement uniform leading-space prefix byte encoding (e.g., `Ġ` / `0x20` binding). |

---

## 13. Step 2 Frozen vs Unfrozen Decisions

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
*End of Tokenizer Specification — ChakrView Research Team*
