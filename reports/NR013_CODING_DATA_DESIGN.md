# CHAKRVIEW RESEARCH REPORT: NR-013 CODING DATASET DESIGN
**Mission ID:** CV-GENESIS-NR-013  
**Date:** 2026-11-25  
**Artifact Directory:** `data/coding_sft_013/` & `data/stage_e_corpus_013/`  

---

## 1. Executive Summary & Design Principles
To enable verified Python code generation while strictly observing project constraints (zero external model weights, zero third-party model APIs), we engineered the **ChakrView Python Coding SFT Dataset** from verified programming fundamentals.

---

## 2. Topic Taxonomy & Coverage
1. **Fundamentals & Arithmetic:** Functional arithmetic, temperature conversions, divisibility, modulus arithmetic.
2. **String Manipulation:** Vowel counting, case-insensitive palindrome detection, whitespace removal, string truncation, word counting.
3. **Lists & Sequences:** Accumulation, min/max search, ordered deduplication, even filtering, arithmetic mean computation.
4. **Dictionaries & Sets:** Inverted lookup, frequency counting, dictionary merging, threshold filtering.
5. **Algorithms & Sorting:** Iterative binary search, bubble sort, Miller-Rabin / trial division primality, Euclidean GCD, Fibonacci sequences.
6. **Data Structures:** Object-oriented Stack (LIFO) and Queue (FIFO) classes with boundary checking.
7. **Practical Engineering:** Safe division handling `ZeroDivisionError`, safe parsing with fallbacks, nested list flattening.

---

## 3. Dataset Partitions & Balance
- **Verified Coding Core:** 87 verified samples (69 Train, 9 Val, 9 Test).
- **Composite Stage-E Training Corpus:** 5,238 samples:
  - 4,686 Stage-D Multilingual conversational samples (89.5%)
  - 552 Weighted verified Python coding samples (10.5%)
- **Composite Stage-E Validation Corpus:** 559 samples (551 Stage-D + 8 Coding).
