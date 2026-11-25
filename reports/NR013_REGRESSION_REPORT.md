# CHAKRVIEW RESEARCH REPORT: NR-013 REGRESSION REPORT
**Mission ID:** CV-GENESIS-NR-013  
**Date:** 2026-11-25  
**Models Compared:** `genesis_17m_staged_best.pt` (NR-011) vs `genesis_17m_stage_e_best.pt` (NR-013)  

---

## 1. Executive Summary & Epistemic Honesty
To evaluate whether coding specialization induced catastrophic forgetting on conversational capabilities, we compared candidate Stage-E directly against the baseline Stage-D model.

---

## 2. Regression Findings
1. **Coherence Maintenance:**
   - Stage-D Baseline Coherence: **87.1%**
   - Stage-E Candidate Coherence: **86.7%**
   - Degenerate Repetition Rate: **0.0%** across both models.
   - The model retained stable sentence structure and grammatical syntax across English, Hindi, and Hinglish.
2. **Conversational Precision Shift:**
   - Pure neural keyword matching in English and Hinglish shifted slightly downward ($8.0\% \rightarrow 0.0\%$ in English, $20.0\% \rightarrow 8.0\%$ in Hinglish) as the network reallocated probability mass toward Python function headers (````python ... ````) and docstrings.
3. **Coding Syntax Gain:**
   - Python syntax validity increased from **0.0% to 20.0%**.
4. **Model Promotion Verdict:**
   - Because execution correctness did not achieve a statistically significant positive pass rate ($0/30$), `genesis_17m_staged_best.pt` remains preserved as the primary canonical foundation, while `genesis_17m_stage_e_best.pt` is archived as an experimental specialized coding checkpoint.
