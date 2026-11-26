# NR-017 Phase 8 — Coding and Language Regression Report
**Mission ID:** CV-GENESIS-NR-017  
**Date:** 2026-11-26  
**Evaluator Script:** `experiments/genesis_stage_g_017/regression_suite.py`  
**Results Artifact:** `data/stage_g_017/regression_evaluation_results.json`  
**Model Under Test:** Canonical `ChakrGenesis-17M Stage-D Best` (`2c21632e`)

---

## 1. Quantitative Regression Evaluation

| Dimension | Measured Performance | Baseline Expectation | Regression Status |
| :--- | :---: | :---: | :--- |
| **Python Code Execution** | **$0.0\%$ (0 / 5 passed)** | $0.0\%$ | **Zero regression** (consistent with NR-015/016) |
| **UTF-8 Byte Integrity** | **$100.0\%$ clean** | $100.0\%$ | **Flawless multi-byte Devanagari** |
| **Conversational Token Flow** | Emits Hindi phrases | Coherent | **Intact** |

---

## 2. Assessment
1. **Safety and Integrity:** Training on state traces did not alter the canonical checkpoint, nor did it introduce tokenizer corruption or UTF-8 decode anomalies.
2. **Coding Capability Boundary:** Zero-shot novel function synthesis remains at $0.0\%$ executable correctness across all pure neural 17M configurations, confirming that state-trace supervision on synthetic arithmetic does not transfer to Python AST generation.
