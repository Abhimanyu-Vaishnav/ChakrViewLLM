# CHAKRVIEW RESEARCH REPORT: NR-014 REGRESSION REPORT
**Mission ID:** CV-GENESIS-NR-014  
**Date:** 2026-11-25  
**Models Evaluated:** Stage-D Baseline vs Stage-E Prior vs Stage-F Balanced Candidate  

---

## 1. Executive Summary & Epistemic Demarcation
Under Phase 8 of Mission `CV-GENESIS-NR-014`, we conducted an explicit cross-model regression audit on the 260-task benchmark.

---

## 2. Regression Analysis Findings
1. **Sentence Coherence & Degeneracy Invariant:**
   - Coherence remained high and stable across all configurations: **83.1% (Stage-D) $\rightarrow$ 85.0% (Stage-E) $\rightarrow$ 85.4% (Stage-F)**.
   - Degenerate looping repetition remained at **0.0%** across all models.
2. **Conversational Preservation in Stage-F:**
   - In Stage-F, balancing 87% conversational data with 13% code prevented the conversational vocabulary degradation observed in Stage-E, recovering English and Hindi conversational hits ($6.7\%$ English, $3.3\%$ Hindi).
3. **Canonical Model Retention Decision:**
   - Because no candidate demonstrated a statistically significant improvement in pure neural execution correctness on unseen programming tasks ($0/40$), **`ChakrGenesis-17M Stage-D Best` (`2c21632e`) is strictly preserved as the primary canonical foundation model**, while Stage-E and Stage-F are preserved and archived as experimental research milestones.
