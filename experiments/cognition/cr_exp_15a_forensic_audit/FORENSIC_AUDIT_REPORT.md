# CHAKRVIEW — CR-EXP-15A INDEPENDENT FORENSIC AUDIT REPORT
## Audit of CR-EXP-15 Results, Checkpoint Integrity, Evaluator Calibration, and Strategic Directives

**Mission Identifier:** CR-EXP-15A  
**Mission Role:** Independent Scientific Auditor & Evaluator Calibration Engineer  
**Audit Standard:** Read-Only First; Strict Empirical Verification; Zero Uncontrolled Modifications  
**Date:** November 2026  
**Final Audit Verdict:** **MODEL CAPABILITY FAILURE CONFIRMED**  

---

## 1. Executive Summary

CR-EXP-15A was executed as an independent forensic audit to determine whether the reported 0/1,200 score across all seven models in CR-EXP-15 represented a genuine capability failure or a shared defect in the benchmark, inference pipeline, tokenizer, checkpoint loading, or evaluator logic.

### Key Audit Findings
1. **Pipeline & Evaluation Validity:** The evaluation scripts, prompt schemas, and greedy inference pipelines were functioning as designed. An independent 12-case calibration suite (`calibrated_evaluator.py`) proved that the evaluator accepts capitalization and punctuation variations, successfully separates answerable from unanswerable queries, and isolates subject from value matching.
2. **Model Loading & Checkpoint Verification:** All 7 checkpoints were strictly verified via PyTorch (`strict=True`, 0 missing, 0 unexpected keys). Weight divergence audits confirmed that Candidate D differed from its baseline parent across 57 out of 57 parameter tensors.
3. **Rescoring Confirmation:** Rescoring all 8,400 raw outputs against the sealed 1,200-probe benchmark confirmed:
   - **Strict Non-Abstention Accuracy ($N=1,050$):** **0.00%** (0/1050) across all 7 models.
   - **Loose Non-Abstention Accuracy ($N=1,050$):** **0.00%** (0/1050) across Candidates B, C, D, Arm B, Arm C; **0.10%** (1/1050) on Parent Baseline and Candidate A (matching `Udaipur` once by random unigram coincidence).
   - **Subject Identification Accuracy ($N=1,050$):** **0.10%** (1/1050) on Candidate B; **0.00%** (0/1050) on all other models.
   - **Calibrated Abstention Accuracy ($N=150$):** **0.00%** (0/150) across all 7 models. Models hallucinated on 100% of unanswerable queries.
4. **Resolution of the Candidate D Validation Paradox:** Candidate D's low validation loss ($1.8418$) was proven to be driven by template and token frequency memorization. In 58.8% of all test generations, Candidate D output one of four high-frequency training names (`Kavita`, `Ankush`, `Mayank`, `Ankita`) paired with high-frequency training cities (`Aurangabad`, `Amritsar`). It acted as a static parametric generator rather than an in-context pointer router.
5. **Experimental Confounders:** Architecture comparisons between Candidate D and Arch Arms B/C were confounded by initialization (Candidate D was fine-tuned from parent; Arms B and C were trained from random initialization for only 200 steps).

---

## 2. Granular Capability Scorecard (Rescored Ground Truth)

| Evaluated Capability | Baseline Parent | Candidate A | Candidate B | Candidate C | Candidate D | Arch Arm B | Arch Arm C |
|---|---|---|---|---|---|---|
| **Literal Entity Copying** | 0/150 (0.00%) | 0/150 (0.00%) | 0/150 (0.00%) | 0/150 (0.00%) | 0/150 (0.00%) | 0/150 (0.00%) | 0/150 (0.00%) |
| ↳ *Single-Token Entities (Dev)* | 0/8 (0.00%) | 0/8 (0.00%) | 0/8 (0.00%) | 0/8 (0.00%) | 0/8 (0.00%) | 0/8 (0.00%) | 0/8 (0.00%) |
| ↳ *Multi-Token Entities* | 0/142 (0.00%) | 0/142 (0.00%) | 0/142 (0.00%) | 0/142 (0.00%) | 0/142 (0.00%) | 0/142 (0.00%) | 0/142 (0.00%) |
| **In-Context Entity Selection** | 0/150 (0.00%) | 0/150 (0.00%) | 0/150 (0.00%) | 0/150 (0.00%) | 0/150 (0.00%) | 0/150 (0.00%) | 0/150 (0.00%) |
| **Static Subject-Value Binding**| 0/150 (0.00%) | 0/150 (0.00%) | 0/150 (0.00%) | 0/150 (0.00%) | 0/150 (0.00%) | 0/150 (0.00%) | 0/150 (0.00%) |
| **Dynamic State Updates (Single)**| 0/150 (0.00%)| 0/150 (0.00%) | 0/150 (0.00%) | 0/150 (0.00%) | 0/150 (0.00%) | 0/150 (0.00%) | 0/150 (0.00%) |
| **Multi-Entity Dynamic Tracking**| 0/150 (0.00%) | 0/150 (0.00%) | 0/150 (0.00%) | 0/150 (0.00%) | 0/150 (0.00%) | 0/150 (0.00%) | 0/150 (0.00%) |
| **Interleaved State Updates** | 0/150 (0.00%) | 0/150 (0.00%) | 0/150 (0.00%) | 0/150 (0.00%) | 0/150 (0.00%) | 0/150 (0.00%) | 0/150 (0.00%) |
| **Delayed Distractor Retrieval**| 0/150 (0.00%) | 0/150 (0.00%) | 0/150 (0.00%) | 0/150 (0.00%) | 0/150 (0.00%) | 0/150 (0.00%) | 0/150 (0.00%) |
| **Calibrated Abstention** | 0/150 (0.00%) | 0/150 (0.00%) | 0/150 (0.00%) | 0/150 (0.00%) | 0/150 (0.00%) | 0/150 (0.00%) | 0/150 (0.00%) |
| **Strict Accuracy (Overall)** | **0/1200 (0.00%)**| **0/1200 (0.00%)**| **0/1200 (0.00%)**| **0/1200 (0.00%)**| **0/1200 (0.00%)**| **0/1200 (0.00%)**| **0/1200 (0.00%)**|

---

## 3. Discovered Defects in CR-15 Methodology

1. **Evaluator Flaw in Historical Predecessor (CR-14):**
   - In CR-14, `has_subj` was hardcoded to `True` for unanswerable probes ($100 / 1200 = 8.33\%$). Real non-abstention subject tracking was 0.00%. CR-15 properly corrected this flaw.
2. **Experimental Confounding in Architecture Ablations:**
   - Architecture Arms B and C were initialized with random weights and trained for only 200 steps (3,200 examples), whereas Candidate D was initialized from the parent checkpoint. The degradation into repetitive loops (`Assistant:` and `AAAA`) was caused by premature evaluation of untrained random weights rather than the inherent failure of $d_{\text{head}}=64$ or width 256.
3. **Optimization Metric Disconnect:**
   - Cross-entropy loss on templated sequences does not measure generalization. Candidate D minimized cross-entropy by learning unigram priors of high-frequency training entities, completely bypassing context retrieval.

---

## 4. Final Verdict and Strategic Directives

- **Official Verdict:** **MODEL CAPABILITY FAILURE CONFIRMED**
- **v0.9.1 Gate Status:** **BLOCKED (NOT READY)**. ChakrMicro cannot be released as v0.9.1 with 0% working memory and entity binding.
- **Recommended Next Experiment (CR-EXP-16):**
  - Synthesize an algorithmic pretraining curriculum focused solely on pointer induction circuits:
    - Task 1: Token sequence echoing ($A B C D \to A B C D$).
    - Task 2: Key-value dictionary lookup ($k_1: v_1, k_2: v_2 \implies k_1: ? \to v_1$).
    - Task 3: Unseen synthetic symbol binding ($\alpha \to 1, \beta \to 2 \implies \alpha = ? \to 1$).
  - Measure induction head formation in Layers 2–4 via attention attribution before re-introducing natural language dialogue.

