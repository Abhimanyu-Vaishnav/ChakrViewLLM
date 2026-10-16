# CHAKRVIEW — Batches 1981–2020
## Language Coverage, Multilingual Fertility & Tokenizer Audit

### Executive Summary

ChakrView's indigenous architecture mandates native, byte-level processing of Indian languages alongside English, code, and mathematics. This audit measures the **vocabulary fertility** (tokens per word), Unicode preservation, and representation quality across the staged natural corpus and reference multilingual test strings using the frozen 4,096-vocabulary tokenizer.

---

### 1. Tokenizer Behavior & Empirical Fertility Audit

Testing the official ChakrView 4,096-vocabulary BPE tokenizer (`release/chakrview-v0.5.0/data/experiments/vocab_4096`):

| Language / Domain Sample | Input Words | Encoded Tokens | Token Fertility (Tokens / Word) | Exact Roundtrip Preservation | Character / Token Efficiency |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Hindi Devanagari Prose**<br>(ऋग्वेदकाल से लेकर आज तक संस्कृत भाषा...) | 20 words | 35 tokens | **1.75** | **PASS (True)** | 3.54 chars/token |
| **Hinglish Mixed Conversational**<br>(ChakrView model local hardware par bina API ke run karega.) | 10 words | 12 tokens | **1.20** | **PASS (True)** | 4.83 chars/token |
| **Pure English Expository**<br>(The Transmission Control Protocol is designed to provide reliable end-to-end communication.) | 11 words | 24 tokens | **2.18** | **PASS (True)** | 3.79 chars/token |
| **Mathematics & Formulaic**<br>(For every point P(x, y), x^2 + y^2 = r^2 represents a circle of radius r.) | 16 words | 30 tokens | **1.88** | **PASS (True)** | 2.50 chars/token |
| **Python Algorithm Code**<br>(`def binary_search(a, x, lo=0, hi=None):\n    return bisect_right(...)`) | 10 words | 33 tokens | **3.30** | **PASS (True)** | 2.58 chars/token |

#### Key Insights:
1. **Exceptional Multilingual Fertility on Hinglish and Hindi:**
   - Hinglish achieves **1.20 tokens/word**, reflecting strong shared subword representations for Latin-transliterated Indian conversational phrasing.
   - Hindi Devanagari achieves **1.75 tokens/word**, substantially better than conventional English-centric tokenizers (which often exhibit fertility rates of 4.0–8.0 tokens/word on Indic scripts).
2. **Perfect Roundtrip Invertibility:**
   - Zero byte-corruption or Unicode decomposition errors occurred across all script evaluations (`decoded == original` is True in 100% of cases).
3. **High Fertility on Source Code:**
   - Python code yields **3.30 tokens/word** due to fine-grained subword decomposition of syntax symbols, colons, underscores, and variable identifiers.

---

### 2. Natural Corpus Domain & Language Distribution Audit

Evaluating the staged dataset ([CORPUS_MANIFEST.json](file:///d:/Project/ChakrView/experiments/batch1941_1980/data/CORPUS_MANIFEST.json)):

| Domain Category | Actual Usable Tokens | Document Count | Languages Represented | Primary Source Identity | Quality Assessment |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Mathematics** | 329,285 | 1,934 | English (`en`) | S. L. Loney Coordinate Geometry | Highly detailed, authentic derivations; heavily concentrated in Test split. |
| **Technical Systems** | 61,905 | 367 | English (`en`) | IETF RFC 793 (TCP) | Outstanding natural technical prose; authentic protocol specification. |
| **Literature / Prose** | 34,007 | 195 | English (`en`) | Rabindranath Tagore *Gitanjali* | Rich poetic prose; completely free of synthetic patterns. |
| **Hindi Prose** | 14,866 | 50 | Hindi (`hi`) | Wikipedia Premchand & Ganit | Natural Devanagari expository prose; excellent syntax quality. |
| **Computer Code** | 11,083 | 64 | Python (`code`) | CPython Standard Library (`heapq`, `bisect`) | Gold-standard production Python algorithms with type annotations. |
| **Sanskrit Studies** | 4,807 | 9 | Hindi-Sanskrit (`hi-sa`) | Wikipedia Sanskrit Sahitya | High-quality prose, but primarily Hindi analysis of Sanskrit literature. |
| **Hinglish** | 0 | 0 | Hinglish (`hi-en`) | *None in staged natural corpus* | **Current Gap:** Needs natural conversational text from verified open datasets. |

---

### 3. Gap Analysis & Next Steps for Balanced Coverage

1. **Address Hinglish Absence:**
   The staged natural corpus currently lacks natural Hinglish dialogues. Future ingestion must source verified conversational Hinglish from CC-BY datasets (e.g. L3Cube MahaHinglish) without falling back to synthetic templates.
2. **Attain Authentic Classical Sanskrit:**
   To establish genuine Sanskrit coverage, ingest verified public-domain verse (e.g. Bhagavad Gita verses or Panchatantra classical texts) rather than encyclopedic articles about Sanskrit written in Hindi.
