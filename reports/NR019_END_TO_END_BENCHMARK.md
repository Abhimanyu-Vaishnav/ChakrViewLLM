# NR-019: End-to-End Sealed Benchmark Report

## 1. Benchmark Design
- **Total Sealed Tasks**: 800 (100% unique prompt texts, 0 duplicates)
- **Linguistic Distribution**:
  - English: 200 tasks
  - Hindi (Devanagari numerals, Devanagari words, mixed): 200 tasks
  - Hinglish (Transliterated words, mixed syntax): 200 tasks
  - Compositional, Ambiguous, and Adversarial: 200 tasks
- **Parameter Regimes**:
  - In-Distribution: 300 tasks (small integer ranges $2 \le n \le 50$)
  - Out-of-Distribution: 500 tasks (large integer ranges $100 \le n \le 500$, novel word lists, unseen syntax combinations)

---

## 2. Complete Performance Matrix

| Evaluation Dimension | System A (Pure) | System B (Regex) | System C (Router Free) | System D (Grounded) | System E (Cascade) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Strict Accuracy (800 Tasks)** | 2.62% (21) | 17.62% (141) | 7.38% (59) | **50.50% (404)** | **51.50% (412)** |
| **In-Distribution Accuracy (300)** | 2.33% | 5.67% | 3.33% | 38.33% | **39.67%** |
| **OOD Accuracy (500)** | 2.80% | 24.80% | 9.80% | 57.80% | **58.60%** |
| **English Language Accuracy (200)** | 4.0% | 37.5% | 10.5% | 72.5% | **76.5%** |
| **Hindi Language Accuracy (200)** | 5.0% | 19.0% | 10.0% | **60.5%** | **60.5%** |
| **Hinglish Language Accuracy (200)** | 1.5% | 14.0% | 9.0% | **69.0%** | **69.0%** |
| **Adversarial / Code Injection Rejection**| 4.0% | 70.0% | 37.0% | **100.0%** | **100.0%** |

---

## 3. Detailed Component Breakdown
- **Operation Identification Accuracy**: 82.4% across all parsed neural outputs.
- **Exact Argument Extraction Accuracy**:
  - System C (Free-form): 12.8%
  - System D (Source-Grounded): **96.2%** on recognized operations.
- **Devanagari Numeral Normalization**: **100%** exact mapping on grounded spans without BPE byte-splitting distortions.
