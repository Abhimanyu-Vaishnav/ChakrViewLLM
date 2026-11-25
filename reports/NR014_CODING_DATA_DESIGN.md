# CHAKRVIEW RESEARCH REPORT: NR-014 CODING DATASET DESIGN
**Mission ID:** CV-GENESIS-NR-014  
**Date:** 2026-11-25  
**Directory:** `data/coding_sft_014/` & `data/stage_f_corpus_014/`  

---

## 1. Executive Summary & Design Principles
To cure the memorization and signature-binding failure of Stage-E, the dataset was expanded from 29 base templates to **234 diverse verified programming tasks**.

---

## 2. Taxonomy & Structural Diversity
1. **Parameterized Arithmetic & Functional Predicates:** Addition, subtraction, multiplication, modulus, divisibility, clamping, bounded ranges.
2. **String Utilities & Formatting:** Prefixes, suffixes, alphanumeric checks, occurrences, hyphens, comma separation, whitespace removal, title casing.
3. **List & Sequence Algorithms:** Cumulative sums, chunking, deduplication, interleaving, in-place reversal, threshold filtering.
4. **Dictionary & Map Structures:** Max/min key extraction, nested lookups, key-value inversion, dictionary merging.
5. **Core Mathematical Algorithms:** Iterative power, iterative factorial, Sieve of Eratosthenes prime counting, Collatz sequence step counts, LCM, digit reversal.
6. **Geometry & Finance:** Rectangle/trapezoid/triangle areas, cube volumes, simple interest, discount calculation, net salary.
7. **Unit Conversions:** Kilometers to miles, kilograms to pounds, meters to feet, time units.

---

## 3. Composite Balance & Anti-Regression Mixing
- Total Stage-F Training Corpus: **5,388 samples**
  - **4,686 samples (87.0%)**: Stage-D multilingual conversational foundation (English, Hindi, Hinglish, factual QA).
  - **702 samples (13.0%)**: Expanded Python coding tasks (weighted 3x).
- Validation Corpus: **570 samples** (551 conversational + 19 coding).
