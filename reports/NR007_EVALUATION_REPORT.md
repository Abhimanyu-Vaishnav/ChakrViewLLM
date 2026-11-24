# CHAKRVIEW RESEARCH REPORT: CV-GENESIS-NR-007
## COMPREHENSIVE CONVERSATIONAL EVALUATION SYNTHESIS

**Mission ID:** CV-GENESIS-NR-007  
**Date:** 24 November 2026  
**Evaluator:** Principal Evaluation Researcher & Benchmark Lead  
**Models Evaluated:**
- **Checkpoint A:** Canonical Pretrain Best (`03a4c16d...`)
- **Checkpoint B:** Genesis SFT Best (`ee1b91e...`)
- **Checkpoint C:** Genesis SFT Final (`cbde4b3...`)

---

### 1. Empirical Results Across 300 Unseen Blind Evaluation Prompts

| Metric / Dimension | Baseline Pretrain | Genesis SFT Best | Genesis SFT Final | Evaluation Criteria & Evidence |
| :--- | :--- | :--- | :--- | :--- |
| **Total Test Prompts** | 300 | 300 | 300 | Unseen blind evaluation suite across 8 subcategories. |
| **Strict Accuracy** | 0.0% (0 / 300) | **13.0% (39 / 300)** | 11.3% (34 / 300) | Exact factual / algorithmic correctness without leniency. |
| **Partial Accuracy** | 75.0% (225 / 300) | **69.0% (207 / 300)** | 71.7% (215 / 300)| Relevant topical explanation or appropriate dialogue turn. |
| **Incorrect / Hallucinated** | 17.7% (53 / 300) | **8.3% (25 / 300)** | 8.7% (26 / 300) | Erroneous calculations or false factual assertions. |
| **Unrelated Response** | 2.7% (8 / 300) | **4.3% (13 / 300)** | 4.0% (12 / 300) | Topic drift or non-sequitur completions. |
| **Repetitive Looping Index** | 1.3% (4 / 300) | **0.0% (0 / 300)** | **0.0% (0 / 300)** | Token repetitions or runaway n-gram attractors. |
| **Unsupported Claims** | 3.3% (10 / 300) | **3.0% (9 / 300)** | 3.0% (9 / 300) | Speculation on unknowable future predictions. |

---

### 2. Language-Specific Performance Breakdown

| Language | Total Prompts | Strict Accuracy | General Coherence | Key Observations |
| :--- | :--- | :--- | :--- | :--- |
| **English (`en`)** | 166 | **18.1% (30 / 166)** | **81.4% (135 / 166)** | Robust assistant persona, valid Python syntax, clear explanations. |
| **Hinglish (`hinglish`)** | 68 | **10.3% (7 / 68)** | **75.0% (51 / 68)** | Natural colloquial conversation, technical concept triggers. |
| **Devanagari Hindi (`hi`)** | 66 | **3.0% (2 / 66)** | **90.9% (60 / 66)** | 100% clean Devanagari syntax; factual QA limited by dataset share. |

---

### 3. Multi-Turn Context & Robustness Synthesis

1. **Entity Recall (English):** **60.0% success (3 / 5)** on single-turn prior references (e.g. Liam's game engine in C++).
2. **Fact Correction:** **0.0% success (0 / 4)** under pure neural attention. The model attends to initial tokens and fails to override them with updated facts without working memory.
3. **Perturbation Robustness:**
   - Typos and misspellings: 66.7% resilience (BPE subwords successfully reconstruct meaning).
   - Short prompts (`Hello`, `RAM`): 100% stability with clean `<EOS>` termination.
   - Paraphrased concepts: 100% semantic invariance.
   - Ambiguous queries with missing context: Hallucinates specific answers rather than asking for clarification.
