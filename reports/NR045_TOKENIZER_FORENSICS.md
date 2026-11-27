# NR-045: Tokenizer Forensics Report

## 1. Executive Summary

This forensic investigation tested whether tokenizer defects, byte-fallback failures, or token-merging anomalies caused the incoherent generation observed in **ChakrView 1.0 RC1**.

### Primary Verdict:
**THE TOKENIZER IS NOT THE CAUSE OF GENERATION INCOHERENCE.**
The canonical Byte-Level BPE tokenizer (`data/experiments/vocab_4096`, Vocab SHA-256 `db4d6119...`) operates with **100.0% exact byte-for-byte round-trip fidelity** across English, Devanagari Hindi, Hinglish, Python code, mathematical notation, and rare Unicode symbols.

---

## 2. Empirical Round-Trip Fidelity Benchmark

Empirical results from `experiments/nr045_generation_forensics/test_tokenizer_forensics.py`:

| Category | Input String Length (Bytes) | Token Count | Compression (Bytes/Tok) | Strict Round-Trip Match |
| :--- | :---: | :---: | :---: | :---: |
| **English Prose** | 44 bytes | 22 tokens | 2.00 | **MATCH (True)** |
| **English Technical** | 60 bytes | 14 tokens | 4.29 | **MATCH (True)** |
| **Hindi Devanagari** | 111 bytes | 11 tokens | **10.09** | **MATCH (True)** |
| **Hindi Grammar Query** | 90 bytes | 13 tokens | **6.92** | **MATCH (True)** |
| **Hinglish Technical** | 49 bytes | 10 tokens | 4.90 | **MATCH (True)** |
| **Hinglish Code-Switch**| 57 bytes | 19 tokens | 3.00 | **MATCH (True)** |
| **Python Function Code**| 98 bytes | 40 tokens | 2.45 | **MATCH (True)** |
| **Mathematical Formulas**| 47 bytes | 23 tokens | 2.04 | **MATCH (True)** |
| **Devanagari Numerals** | 95 bytes | 29 tokens | 3.28 | **MATCH (True)** |
| **Special Tag Tokens** | 59 bytes | 30 tokens | 1.97 | **MATCH (True)** |
| **Rare Unicode & Emoji** | 89 bytes | 48 tokens | 1.85 | **MATCH (True)** |
| **Empty String** | 0 bytes | 0 tokens | N/A | **MATCH (True)** |
| **Whitespace & Indents** | 28 bytes | 12 tokens | 2.33 | **MATCH (True)** |

---

## 3. Special Token & Byte-Level Robustness Findings
- **Special Token Identifiers:** Verified `BOS = 0`, `EOS = 1`, `PAD = 2`.
- **Byte Fallback:** When encountering rare emojis or mathematical symbols outside pre-merged pairs (e.g. `∀`, `ℝ`, `ॐ`, `𝄞`), the byte-fallback mechanism decomposes the UTF-8 bytes into single-byte base tokens (IDs 3..258) and reconstructs the exact Unicode glyph on decode.
- **Decoding Incomplete Multi-Byte Bytes:** Truncated byte sequences are gracefully handled via `errors="replace"`.
- **Conclusion:** The tokenizer correctly encodes and decodes all text. Incoherence arises purely after token generation by the neural model.
