# ChakrView Step 3 — Empirical Tokenizer Benchmark Summary

**Date**: 2026-09-26T08:29:43Z  
**Random Seed**: 42  
**Corpus Size**: 748 items (61,644 UTF-8 bytes)  

---

## 1. Vocabulary Candidates Comparison (Phase 3.4 & 3.5)

| Metric | V = 2048 | V = 4096 | V = 8192 | V = 16384 |
| :--- | :---: | :---: | :---: | :---: |
| **Actual Vocab Size** | 2048 | 4096 | 8192 | 16351 |
| **Learned Merges** | 1789 | 3837 | 7933 | 16092 |
| **Total Tokens** | 18,159 | 13,397 | 8,907 | 748 |
| **Compression (Bytes/Tok)** | 3.395 | 4.601 | 6.921 | 82.412 |
| **Tokens / Word (Overall)** | 2.678 | 1.976 | 1.313 | 0.110 |
| **Tokens / Char (Overall)** | 0.405 | 0.299 | 0.199 | 0.017 |
| **Hindi (Tokens/Word)** | 2.46 | 1.76 | 1.20 | 0.07 |
| **English (Tokens/Word)** | 2.57 | 1.89 | 1.29 | 0.07 |
| **Hinglish (Tokens/Word)** | 1.87 | 1.30 | 0.97 | 0.08 |
| **Sanskrit (Tokens/Word)** | 3.87 | 3.07 | 2.01 | 0.08 |
| **Code (Tokens/Word)** | 2.56 | 1.98 | 1.19 | 0.15 |
| **Numeric (Tokens/Word)** | 3.03 | 2.32 | 1.56 | 0.15 |
| **Encode Latency (µs/doc)** | 429.8 | 578.2 | 857.6 | 1391.5 |
| **Decode Latency (µs/doc)** | 3.1 | 2.6 | 2.0 | 0.5 |
| **Static Memory (KB)** | 435.7 | 900.3 | 1832.2 | 3875.8 |
| **Lossless Round-Trip** | PASS (100%) | PASS (100%) | PASS (100%) | PASS (100%) |

---

## 2. Numeric Tokenization Strategies (Phase 3.6)

| Strategy | Description | Total Tokens | Avg Tok/Item | Expansion vs Normal BPE | Lossless |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Candidate A** | Single Digits (`\d`) | 154 | 6.16 | 2.11x | True |
| **Candidate B** | 2-Digit Chunks (`\d{1,2}`) | 107 | 4.28 | 1.47x | True |
| **Candidate C** | Normal BPE (Unconstrained) | 73 | 2.92 | 1.00x | True |

---

## 3. Scale Latency & Throughput Benchmark (Phase 3.9)

| Payload Size | Tokens | Warm Encode Median | Warm Decode Median | Encode Throughput | Decode Throughput |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **32_bytes** | 6 | 270.2 µs | 1.2 µs | 0.12 MB/s | 27.74 MB/s |
| **128_bytes** | 17 | 1018.4 µs | 2.4 µs | 0.11 MB/s | 53.07 MB/s |
| **512_bytes** | 120 | 9056.9 µs | 13.5 µs | 0.05 MB/s | 36.85 MB/s |
| **2_KB** | 538 | 35509.5 µs | 59.4 µs | 0.06 MB/s | 32.88 MB/s |
| **8_KB** | 2221 | 140044.7 µs | 228.4 µs | 0.05 MB/s | 35.46 MB/s |
| **32_KB** | 8883 | 548415.6 µs | 781.8 µs | 0.06 MB/s | 39.38 MB/s |
| **128_KB** | 35547 | 2186984.2 µs | 3628.8 µs | 0.06 MB/s | 40.10 MB/s |

---

## 4. Invariant Verification Results

- **Lossless Reconstruction Invariant**: `Decode(Encode(x)) == x` strictly **PASSED** on all candidates and test suites.
- **Raw Byte Invariant**: `decode_bytes(encode_bytes(data)) == data` strictly **PASSED** on all arbitrary, random, and malformed byte sequences.
- **Determinism Invariant**: **PASSED** across all repetitions (bit-exact identical token IDs and statistics).
- **Special Token Contract**: `<BOS>=0, <EOS>=1, <PAD>=2`, raw bytes `3..258` strictly preserved across all candidates.