# NR-015 Phase 8 — Multilingual and Conversational Evaluation
**Mission ID:** CV-GENESIS-NR-015  
**Date:** 2026-11-25  
**Model Under Test:** `ChakrGenesis-17M Stage-D Best` (`2c21632e`)  
**Evaluator:** Pure-Neural Autoregressive vs. Cognitive Controller Hybrid  
**Artifacts:** `data/sealed_eval_015/multilingual_eval_results.json`

---

## 1. Quantitative Performance Overview

| Evaluation Dimension | Pure-Neural Core | Hybrid Cognitive Controller | Verdict |
| :--- | :---: | :---: | :--- |
| **English Conversational Coherence** | $75.0\%$ ($3/4$) | $75.0\%$ ($3/4$) | Fluency without semantic grounding |
| **Hindi Devanagari Generation** | $100.0\%$ ($4/4$) | $25.0\%$ ($1/4$) | Pure-neural outputs Hindi sentences |
| **Hinglish Dialogue Handling** | $100.0\%$ ($4/4$) | $75.0\%$ ($3/4$) | Code-mixed token generation |
| **Overall Coherence Rate** | **$91.7\%$ ($11/12$)** | **$58.3\%$ ($7/12$)** | Pure-neural retains higher text flow |

---

## 2. Linguistic & Behavioral Analysis
1. **Devanagari UTF-8 Integrity:** The BPE tokenizer cleanly generates grammatically valid Hindi phrases without UTF-8 corruption or replacement characters (`\ufffd`), such as:
   - Prompt: `कंप्यूटर क्या होता है? संक्षेप में बताएं।`
   - Output: `'# मेरे पास से 4 घटाने पर 60 प्राप्त होता है। बाकी...'`
   - Output: `': अर्जुन कोलकाता में वास्तुकार के रूप में कार्यरत हैं।...'`
2. **Semantic Disconnect:** While surface grammar and syntax are valid, the underlying factual knowledge is disconnected (answering a computer query with an arithmetic story snippet).
3. **Conversational Drift:** Prompts without explicit arithmetic triggers often hallucinate random mathematical word problems or GSM8k delimiters (`### <num>`), reflecting the heavy weighting of math reasoning in Stage-D fine-tuning.
