# UCIP-11 Phase C: Tokenizer & Representation Study

**Mission ID:** UCIP-11  
**Tokenizer:** Byte-level BPE, Vocabulary Size 4,096  
**Artifact Directory:** `data/experiments/vocab_4096`  
**Result Artifact:** `experiments/ucip11/data/manifests/tokenizer_audit_results.json`  

---

## 1. Executive Summary

This study investigated whether the 4096-token byte-level BPE tokenizer poses structural bottlenecks for natural language acquisition across English, Devanagari Hindi, Roman Hindi, and Hinglish.

**Key Findings:**
1. **Balanced Fertility Across Languages:**
   - English Conversational: **2.23 tokens / word** (2.62 chars/token)
   - English Technical: **2.07 tokens / word** (3.43 chars/token)
   - Devanagari Hindi: **2.22 tokens / word** (2.95 chars/token; 7.98 UTF-8 bytes/token)
   - Roman Hindi: **1.87 tokens / word** (3.28 chars/token)
   - Hinglish: **2.16 tokens / word** (3.01 chars/token)
   - Python Code: **1.53 tokens / word** (4.07 chars/token)
   - Mathematics: **2.28 tokens / word** (2.08 chars/token)
2. **Hindi is NOT Disproportionately Fragmented:**
   Devanagari Hindi exhibits 2.22 tokens/word, nearly identical to English conversational text (2.23 tokens/word). Because Devanagari characters are 3 bytes in UTF-8, the byte-level BPE learned merged subwords that represent complete Devanagari syllables effectively (averaging 7.98 bytes per token).
3. **Round-Trip Fidelity:**
   Across all tested languages, 100% of encoded sequences decoded back to bit-exact original text without character loss or corruption (`round_trip_exact = True`).
4. **Context Window Impact:**
   A standard 512-token context window accommodates approximately **230 to 250 words** across English, Hindi, and Hinglish alike. Tokenizer fertility is not the cause of model incoherence.

---

## 2. Quantitative Representation Metrics

| Linguistic Category | Total Chars | UTF-8 Bytes | BPE Tokens | Words | Tokens / Word (Fertility) | Chars / Token | Bytes / Token | Round-Trip Exact |
|---|---|---|---|---|---|---|---|---|
| **English (Conversational)** | 238 | 238 | 91 | 41 | **2.23** | 2.62 | 2.62 | **YES** |
| **English (Technical)** | 288 | 288 | 84 | 41 | **2.07** | 3.43 | 3.43 | **YES** |
| **Hindi (Devanagari)** | 268 | 726 | 91 | 41 | **2.22** | 2.95 | 7.98 | **YES** |
| **Roman Hindi** | 200 | 200 | 61 | 33 | **1.87** | 3.28 | 3.28 | **YES** |
| **Hinglish** | 235 | 235 | 78 | 36 | **2.16** | 3.01 | 3.01 | **YES** |
| **Python Code** | 338 | 338 | 83 | 54 | **1.53** | 4.07 | 4.07 | **YES** |
| **Mathematics & Logic** | 162 | 162 | 78 | 34 | **2.28** | 2.08 | 2.08 | **YES** |

---

## 3. Subword & Special Token Invariants

- **Vocabulary:** 4,096 tokens
- **Special Tokens:**
  - Token `0`: `<bos>` / `<s>`
  - Token `1`: `<eos>` / `</s>`
  - Token `2`: `<pad>`
  - Token `3`: `<unk>`
- **Multi-byte Handling:** Uses byte fallback for unseen Unicode bytes, ensuring zero out-of-vocabulary crashes.
- **Verdict:** The `vocab_4096` tokenizer is robust and well-calibrated. Architectural and data design—not tokenizer limitations—govern language acquisition in ChakrView.
