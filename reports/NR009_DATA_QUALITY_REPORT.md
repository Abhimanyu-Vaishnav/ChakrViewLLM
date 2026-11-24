# CV-GENESIS-NR-009 Dataset Quality and Integrity Report

## 1. Executive Summary

Mission **CV-GENESIS-NR-009 Phase 4** executed a forensic quality, duplicate, and leakage audit across all splits of the balanced multilingual SFT corpus (`experiments/genesis_balanced_sft_008/data/`).

---

## 2. Quantitative Integrity Audit

| Metric | Result | Standard Required | Status |
| :--- | :--- | :--- | :--- |
| **Total Examples Audited** | 4,401 | 4,401 | **PASSED** |
| **Internal Train Duplicates** | **0** | 0 | **PASSED (100% Unique)** |
| **Train / Validation Overlap Leakage**| **0** | 0 | **PASSED (Zero Leakage)** |
| **Train / Test Overlap Leakage** | **0** | 0 | **PASSED (Zero Leakage)** |
| **Blind Benchmark Anti-Contamination**| **0** | 0 | **PASSED (300 Prompts Filtered)** |
| **Empty / Short Targets (<3 chars)** | 161 (3.6%) | $< 5.0\%$ | **ACCEPTABLE (Single digit/symbol answers)** |

---

## 3. Linguistic Composition

- **English (`en`):** 1,881 examples (50.3%)
- **Devanagari Hindi (`hi`):** 1,558 examples (41.7%)
- **Hinglish (`hinglish`):** 301 examples (8.0%)

All prompt-response pairs strictly conform to canonical `User: <input>\nAssistant: <target>` formatting.
