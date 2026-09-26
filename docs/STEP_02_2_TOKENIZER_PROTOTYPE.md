# ChakrView Step 2.2 — Tokenizer Research Prototype Report

**Document Version**: 0.1.0  
**Phase**: Step 2.2 — Tokenizer Research Prototype Implementation  
**Status**: COMPLETE & VERIFIED (RESEARCH PROTOTYPE ONLY)  
**Target Invariant**: $\text{Decode}(\text{Encode}(\text{text})) \equiv \text{text} \quad \forall \text{ supported inputs}$  

---

> ### CRITICAL DISCLAIMER & NOTICE
> **This is a RESEARCH PROTOTYPE, NOT the final production ChakrView tokenizer.**  
> - No large vocabulary has been trained.
> - No external datasets were downloaded.
> - No external frameworks (`tokenizers`, `sentencepiece`, `transformers`, `torch`) were installed or used.
> - Implementation uses strictly the **Python 3.14 Standard Library** and existing **pytest**.
> - The sole purpose of Step 2.2 is to implement the smallest possible Byte-Level BPE engine that mathematically proves the foundational architectural invariants established in Step 2 and Step 2.1.
> - **Core Invariant**: *"Tokenizer correctness takes precedence over compression."*

---

## 1. Package Architecture & Module Layout

The tokenizer prototype is encapsulated in an isolated package under `chakrview/tokenizer/`, completely decoupled from the neural core architecture and runtime:

```
chakrview/
├── __init__.py
└── tokenizer/
    ├── __init__.py           # Unified public API exports
    ├── special_tokens.py     # Special token definitions and safety predicates (<BOS>, <EOS>, <PAD>)
    ├── bytes.py              # Foundational 256 byte primitives & bijective mapping
    ├── bpe.py                # Deterministic BPE counting, tie-breaking, merging, and training
    ├── encoder.py            # Lossless text & raw byte encoding with special-token isolation
    ├── decoder.py            # Exact lossless reconstruction to bytes and Unicode text
    └── tokenizer.py          # Unified BPETokenizer model class
```

---

## 2. Token ID Mapping & Disambiguation

To prevent common tokenization bugs, the system strictly formalizes four distinct layers of data representation:

| Layer | Definition | Example (`'क'`) | Example (`'A'`) |
| :--- | :--- | :--- | :--- |
| **1. Unicode Character** | Abstract typographic element or grapheme cluster | `'क'` | `'A'` |
| **2. Unicode Code Point** | Scalar integer in Unicode space | `U+0915` (`2325`) | `U+0041` (`65`) |
| **3. UTF-8 Binary Bytes** | Variable-length 8-bit byte representation (1–4 bytes) | `b'\xe0\xa4\x95'` (3 bytes) | `b'A'` (`0x41`, 1 byte) |
| **4. Tokenizer Token ID** | Discrete integer index in the ChakrView vocabulary | `[227, 167, 152]` (base bytes) | `[68]` (base byte) |

### 2.1 Exact Token ID Allocation
The vocabulary budget is partitioned to ensure zero collision between special tokens, byte primitives, and merged subwords:

$$\begin{aligned}
\text{Special Tokens} &\in [0, 2] \quad &&(3 \text{ tokens: } \langle\text{BOS}\rangle=0, \langle\text{EOS}\rangle=1, \langle\text{PAD}\rangle=2) \\
\text{Foundational Bytes} &\in [3, 258] \quad &&(256 \text{ tokens for bytes } \mathtt{0x00} \dots \mathtt{0xFF}) \\
\text{Learned Merges} &\in [259, \infty) \quad &&(\text{Subwords generated via deterministic BPE})
\end{aligned}$$

- **Byte Offset**: $\text{token\_id} = b + 3 \quad \forall b \in \{0, 1, \dots, 255\}$.
- **Zero Collision Guarantee**: Special token IDs $0, 1, 2$ cannot be emitted by raw bytes. Base byte values cannot collide with special tokens.

---

## 3. Special Token Safety & Isolation (Phase G)

A common vulnerability in tokenizers is accidental interpretation of user text as special tokens (e.g., prompt injection or text containing `"<BOS>"` causing sequence boundary errors).

### Safety Invariants Enforced:
1. **No `<UNK>`, `<MASK>`, `<SEP>`, `<NL>`**:
   Byte-level fallback ensures no unknown tokens ever exist. Special tokens are strictly minimal (3 tokens).
2. **Literal Text Isolation**:
   When user text contains literal strings like `"<BOS>"`, `"<EOS>"`, or `"<PAD>"`, the characters are encoded strictly into their UTF-8 byte tokens:
   $$\text{"<BOS>"} \implies [63, 69, 82, 86, 65]$$
   It **never** produces the special token integer `0`.
3. **Explicit API Control**:
   Special tokens can only be introduced via explicit parameter flags:
   `encode_text(text, add_bos=True, add_eos=True)`
4. **Clean Reconstruction**:
   `decode_tokens(tokens, skip_special_tokens=True)` strips IDs $0, 1, 2$ while perfectly preserving all literal strings.

---

## 4. Minimal BPE Engine & Deterministic Rules (Phases E & F)

The BPE engine implements deterministic pair-frequency counting, merge selection, and iterative merge replacement.

### 4.1 Strict Deterministic Tie-Breaking
When multiple candidate pairs share the exact same maximum frequency during training, ties are resolved deterministically using a triple-key tuple:
$$\text{Best Pair} = \arg\min_{\text{pair} \in \text{Candidates}} \Big( -\text{count}(\text{pair}), \; \text{pair}[0], \; \text{pair}[1] \Big)$$

- **Primary**: Frequency descending.
- **Secondary**: First token ID ascending (numerical lexicographic order).
- **Tertiary**: Second token ID ascending.

This eliminates all dependence on hash seed randomization, dictionary iteration order, or operating system architecture.

### 4.2 Encoding with Fixed Merges
Given a precomputed merge table with ranks $\text{rank}((p_0, p_1))$, encoding scans the sequence for adjacent pairs present in the merge table and replaces the pair with the lowest rank (highest priority). This process repeats until no further eligible pairs remain.

---

## 5. Dual Lossless API: Text and Raw Bytes (Phases C & D)

To ensure robust handling without forcing invalid binary sequences into Unicode string APIs:

1. **Text-Level API**:
   - `encode_text(text: str, add_bos=False, add_eos=False) -> list[int]`
   - `decode_tokens(tokens: list[int], skip_special_tokens=True) -> str`
   - Guarantees: $\text{decode\_tokens}(\text{encode\_text}(S)) \equiv S$ for all valid Unicode strings without normalization drift.
2. **Byte-Level API**:
   - `encode_bytes(data: bytes) -> list[int]`
   - `decode_bytes(tokens: list[int], skip_special_tokens=True) -> bytes`
   - Guarantees: $\text{decode\_bytes}(\text{encode\_bytes}(B)) \equiv B$ for all arbitrary binary sequences, including non-UTF-8 bytes (`0x80`, `0xFF`, overlong encodings, truncated byte sequences).

---

## 6. Comprehensive Test Suite & Verification (Phases H & I)

The prototype test suite contains **85 unit and adversarial tests** across 7 dedicated test modules, executed using `pytest`:

```
============================= test session starts =============================
platform win32 -- Python 3.14.7, pytest-9.1.1, pluggy-1.6.0
rootdir: D:\Project\ChakrView
configfile: pytest.ini
testpaths: tests
collected 85 items

tests\test_tokenizer_adversarial.py ...............................      [ 36%]
tests\test_tokenizer_baseline.py .                                       [ 37%]
tests\test_tokenizer_bpe_engine.py ......                                [ 44%]
tests\test_tokenizer_bytes.py .......                                    [ 52%]
tests\test_tokenizer_determinism.py ...                                  [ 56%]
tests\test_tokenizer_special_tokens.py ....                              [ 61%]
tests\test_tokenizer_utf8_lossless.py .................................  [100%]

============================= 85 passed in 0.09s ==============================
```

### 6.1 Test Coverage Summary

| Suite Module | Tests | Focus Areas |
| :--- | :--- | :--- |
| `test_tokenizer_bytes.py` | 7 | All 256 bytes (`0x00`..`0xFF`), boundary bytes, non-UTF-8 binary blobs, bounds validation, zero collisions. |
| `test_tokenizer_utf8_lossless.py` | 33 | Functional and object-oriented round-trip for English, Hindi, Hinglish, Mixed, Sanskrit, Emojis, Mathematics, URLs, Paths, Code, Whitespace, CRLF/LF, Numbers, Long text. |
| `test_tokenizer_bpe_engine.py` | 6 | Pair counting, deterministic tie-breaking, single-pass non-overlapping merge, toy BPE training, Devanagari multi-byte merges, token reduction. |
| `test_tokenizer_determinism.py` | 3 | Repeated encoding (50 iterations), repeated decoding (50 iterations), identical merge table generation across independent training runs. |
| `test_tokenizer_special_tokens.py` | 4 | Frozen ID checks, literal `"<BOS>"` string isolation, explicit `add_bos`/`add_eos` generation, `skip_special_tokens` behavior. |
| `test_tokenizer_adversarial.py` | 31 | Full Category I stress tests: combining marks, Sanskrit conjuncts, ZWJ/ZWNJ half-forms, variation selectors (`\uFE0E`, `\uFE0F`), emoji sequences, malformed/truncated byte sequences (`0x80`, truncated prefixes). Zero normalization alterations. |
| `test_tokenizer_baseline.py` | 1 | Unoptimized performance and throughput metrics verification. |

---

## 7. Performance Baseline Measurements (Phase J)

*Notice: This is a diagnostic research baseline for an unoptimized pure-Python prototype, not an optimized production benchmark. No performance claims are made.*

Hardware: Intel Core i9-13900H (Windows 11, Python 3.14.7, single-threaded pure Python).

| Sample Benchmark Text | UTF-8 Bytes | Tokens Produced | Compression (Bytes/Token) | Encode Latency ($\mu\text{s}$) | Decode Latency ($\mu\text{s}$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **English (Short)** | 73 | 67 | 1.09 | 60.84 | 6.00 |
| **Hindi (Short)** | 146 | 93 | 1.57 | 47.65 | 8.27 |
| **Hinglish (Short)** | 65 | 60 | 1.08 | 27.65 | 5.09 |
| **Source Code** | 83 | 71 | 1.17 | 62.07 | 5.73 |
| **Arithmetic & Symbols** | 76 | 68 | 1.12 | 37.77 | 6.01 |

- **Observations**:
  - Decode latency is consistently sub-10 microseconds on all test inputs.
  - Linear scan merge replacement in pure Python is sufficient for prototype validation.
  - Byte-to-token compression ratio increases as frequent multi-byte Devanagari characters are merged.

---

## 8. Known Prototype Limitations & Transition Path

This prototype validates the mathematical invariants. The following items remain unfrozen for subsequent phases:

1. **Pre-Tokenization RegEx Splitting**:
   The research prototype processes full byte sequences directly. In subsequent phases, grapheme-aware pre-tokenization regex splitting (for Indic conjuncts, punctuation, and code indentation) will be layered before the merge engine.
2. **Merge Vocabulary Scale**:
   The prototype was tested with toy merge tables ($\le 10$ merges). Training a full vocabulary ($V = 2048, 4096, 8192$) will take place in Step 3 on the representative ChakrView training corpus.
3. **Execution Runtime**:
   The prototype runs in pure Python. Production inference will compile the lookup trie and merge tables into optimized native C or memory-mapped arrays for edge and CPU inference.

---

## 9. Conclusion

ChakrView Step 2.2 successfully demonstrates:
- **100% Lossless Round-Trip Reconstruction** ($\text{Decode}(\text{Encode}(S)) \equiv S$).
- **Zero Out-of-Vocabulary Failures** without `<UNK>`.
- **Complete Special Token Safety and Isolation**.
- **Bit-Exact Raw Byte Round-Trip** for all 256 bytes and arbitrary binary data.
- **Deterministic Merge Engine** with mathematical tie-breaking.
