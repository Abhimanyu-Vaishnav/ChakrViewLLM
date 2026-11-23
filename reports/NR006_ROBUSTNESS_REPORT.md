# CHAKRVIEW RESEARCH REPORT: CV-GENESIS-NR-006
## ROBUSTNESS, PERTURBATION & UNCERTAINTY REPORT

**Mission ID:** CV-GENESIS-NR-006  
**Date:** 23 November 2026  
**Evaluator:** Principal Reliability Engineer  
**Test Suite:** `experiments/genesis_intelligence_validation_006/robustness/robustness_results.json`  

---

### 1. Robustness Evaluation Matrix

| Perturbation Category | Test Count | Robust / Handled | Degraded / Hallucinatory | Key Behavioral Pattern |
| :--- | :--- | :--- | :--- | :--- |
| **Unusual Wording / Inversion** | 3 | 2 (66.7%) | 1 (33.3%) | Retains semantic focus on core keywords (e.g. CPU, RAM). |
| **Misspellings & Typos** | 3 | 2 (66.7%) | 1 (33.3%) | BPE subword tokenizer absorbs typos gracefully (`Pythn`, `Frnace`). |
| **Extremely Short Prompts** | 4 | 4 (100%) | 0 (0.0%) | Triggers greeting or definition completions without crashing. |
| **Long Multi-Sentence Prompts**| 2 | 1 (50.0%) | 1 (50.0%) | Generates related technical advice, but loses detail towards end. |
| **Rephrased Standard Questions** | 3 | 3 (100%) | 0 (0.0%) | Full invariance to syntactic paraphrasing of known concepts. |
| **Unfamiliar / Out-of-Domain** | 2 | 0 (0.0%) | 2 (100%) | Generates speculative scientific pseudo-claims (Hallucination). |
| **Insufficient Information** | 3 | 0 (0.0%) | 3 (100%) | Fails to request clarification; asserts arbitrary attributes. |

---

### 2. Detailed Findings

1. **Typo Resilience via Canonical BPE Tokenizer:**
   The canonical BPE tokenizer (`data/experiments/vocab_4096`) breaks misspelled words (e.g. `Frnace` -> `Fr`, `n`, `ace`) effectively, allowing the transformer layers to attend to the underlying root semantic representations.
2. **Short Prompt Stability:**
   Unlike earlier ChakrView checkpoints which produced runaway repetitive loops when prompted with single words (e.g. `"Hello"` or `"RAM"`), Genesis SFT Best cleanly produces short, polite responses or technical definitions and terminates at `<EOS>`.
3. **Uncertainty & Hallucination Vulnerability:**
   When faced with ungrounded questions with missing referents (e.g. `"How much does it weigh?"` or `"Uski total price kitni hogi?"`), the pure neural model asserts arbitrary numbers and descriptions rather than producing principled clarification questions.
   - **Recommendation:** Implement explicit clarification and abstention SFT vectors or route ambiguous queries through the Principled Abstention cognitive filter.
