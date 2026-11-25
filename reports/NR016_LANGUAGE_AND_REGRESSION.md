# NR-016 Phase 7 — Language and Regression Testing Report
**Mission ID:** CV-GENESIS-NR-016  
**Date:** 2026-11-25  
**Evaluator Script:** `experiments/eval_language_regression_016.py`  
**Results Artifact:** `data/scientific_eval_016/language_regression_results.json`  
**Model Under Test:** Canonical `ChakrGenesis-17M Stage-D Best` (`2c21632e`)

---

## 1. Quantitative Evaluation Summary

| Evaluation Metric | Pure Neural Performance | Hybrid Cognitive Performance | Verdict |
| :--- | :---: | :---: | :--- |
| **UTF-8 Byte Integrity** | **$100.0\%$ (12/12)** | **$100.0\%$ (12/12)** | Flawless multi-byte Devanagari decoding |
| **Conversational Coherence Rate** | **$100.0\%$ (12/12)** | **$100.0\%$ (12/12)** | Well-formed token flow, 0 runaway loops |
| **Objective Factual Hit Rate** | **$0.0\%$ (0/10)** | **$0.0\%$ (0/10)** | Zero open-domain factual retrieval |
| **Repetition Penalty Stability** | Stable | Stable | No repeated token collapse |

---

## 2. Linguistic Insights & Behavioral Regression Analysis
1. **Flawless Devanagari Tokenization:** All Hindi inputs and responses decode cleanly without byte replacement errors (`\ufffd`), demonstrating that the BPE vocabulary handles Devanagari byte sequences reliably.
2. **The Factual Disconnect:** While grammatical fluency is high (e.g. `'मनानियों की राजधानी में 13 को 11 से गुणा करने पर 74 प्राप्त...'`), the answers completely lack factual semantic grounding (answering "What is the capital of France?" with a candy arithmetic story).
3. **Severe Math Bias:** Because Stage-D fine-tuning heavily prioritized mathematical reasoning and GSM8K-style prompts, the model defaults to math problem tokens (`12 + 60 = 28`, `candies in the bag`, `### 34`) even when prompted with open-domain geography or science questions.
