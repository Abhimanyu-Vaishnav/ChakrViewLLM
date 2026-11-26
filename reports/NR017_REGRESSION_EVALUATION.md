# NR-017 Phase 8 — Coding and Language Regression Report
**Mission ID:** CV-GENESIS-NR-017  
**Date:** 2026-11-26  
**Evaluator Script:** `experiments/genesis_stage_g_017/run_regression_tests.py`  
**Results Artifact:** `data/stage_g_017/regression_evaluation_results.json`  
**Models Evaluated:** Condition A (Direct), Condition B (Explicit Trace), Condition C (Compact Trace)

---

## 1. Quantitative Regression Evaluation Matrix

| Evaluation Dimension | Direct Answer (Cond A) | Explicit State Trace (Cond B) | Compact Trace (Cond C) | Target / Baseline |
| :--- | :---: | :---: | :---: | :---: |
| **Python AST Syntax Validity (20 tasks)** | $0.0\%$ (0/20) | **$70.0\%$ (14/20)** | $0.0\%$ (0/20) | AST syntax parsing |
| **Python Execution Correctness (20 tasks)** | **$0.0\%$ (0/20)** | **$0.0\%$ (0/20)** | **$0.0\%$ (0/20)** | Subprocess hidden assertions |
| **UTF-8 Byte Integrity** | **$100.0\%$** | **$100.0\%$** | **$100.0\%$** | Zero replacement chars |
| **Conversational Coherence (English/Hindi/Hinglish)** | **$100.0\%$ (6/6)** | **$100.0\%$ (6/6)** | **$33.3\%$ (2/6)** | Sentence flow |

---

## 2. Key Findings & Behavioral Observations
1. **Explicit Trace Syntax Retention:** Condition B preserved a high syntax validity on Python tasks ($70.0\%$, 14/20) because structured XML-like tag generation regularizes sequential block emission. However, execution assertion pass rate remained at **$0.0\%$** across all conditions.
2. **Compact Trace Conversational Collapse:** Condition C suffered severe conversational regression ($33.3\%$ coherence), where general conversational prompts were answered with distorted arrow traces (`S0 -> S1...`), reflecting vocabulary corruption from dense arrow transitions.
3. **Canonical UTF-8 Health:** Zero multi-byte Devanagari character corruption occurred across any run.
