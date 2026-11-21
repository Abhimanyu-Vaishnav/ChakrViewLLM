# Project Genesis: Tokenizer Decision & Specification

**Date:** 21 November 2026  
**Project:** ChakrView (ULTIMATE-BRAIN-003)  
**Selected Tokenizer Artifact:** `experiments/tokenizer/candidates/v4096`  
**Vocabulary Size:** 4,096 tokens  
**Format:** Byte-level BPE with UTF-8 byte fallback  

---

## 1. Tokenizer Audit & Findings

We audited the existing candidate tokenizers (`v2048`, `v4096`, `v8192`, `v16384`):

1. **`v2048`:** Over-fragments Hindi words (4.2 tokens/word average) and English technical terms, causing excessive sequence length.
2. **`v4096` (Selected):**
   - **Byte Coverage:** Full byte fallback (256 byte tokens), ensuring **zero out-of-vocabulary (`<unk>`) errors**.
   - **Special Tokens:** `<pad>=0`, `<bos>=1`, `<eos>=2`.
   - **Compression Ratio:**
     - English: ~1.28 tokens/word.
     - Hindi (Devanagari): ~1.85 tokens/word (conjuncts and frequent words like "नमस्ते", "कैसे", "हैं" properly merged).
     - Hinglish / Code: Seamlessly preserved via shared Latin subwords.
   - **Pre-existing Tokenized Assets:** The canonical Stage-C corpus (6,651,093 tokens across 27 shards) is already deterministically tokenized with this exact vocabulary checksum (`7498d92adeef7c9e64d4d1bbd5b4b23f8774692d868742e3b98e30c584c4403f`).
3. **`v8192` & `v16384`:**
   - Greatly increases embedding table parameters ($4096 \times 256 = 1.05\text{M}$ params vs $16384 \times 256 = 4.19\text{M}$ params), consuming over 50% of the parameter budget just on embeddings.

---

## 2. Tokenizer Decision

**Decision:** Formally adopt the **`v4096`** Byte-level BPE tokenizer as the foundation tokenizer for Project Genesis.

- Frozen artifacts located at: `experiments/tokenizer/candidates/v4096/`
  - `vocab.json`
  - `merges.json`
  - `config.json`
  - `metadata.json`
- Lossless roundtrip encoding/decoding verified across English, Hindi, numbers, and symbols.
