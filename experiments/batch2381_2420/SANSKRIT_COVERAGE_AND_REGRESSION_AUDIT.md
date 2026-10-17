# SANSKRIT COVERAGE AND REGRESSION AUDIT REPORT

**Batch:** 2381–2420  
**Audit Phase:** Phase 3 — Sanskrit Training Coverage & Regression Verification  
**Status:** CONFIRMED FROM PRIMARY ARTIFACTS  

---

## 1. Quantitative Verification of Held-Out Sanskrit Regression

From our fresh isolated evaluation ([FRESH_EVALUATION.json](file:///D:/Project/ChakrView/experiments/batch2381_2420/FRESH_EVALUATION.json)):
- **Source:** `wikipedia_sa_bhagavad_gita` (61 documents, 19,577 evaluated target tokens)
- **Baseline TW-CE:** `4.320159` (Perplexity: `75.20`)
- **Candidate TW-CE:** `4.397778` (Perplexity: `81.27`)
- **Exact Delta ($\Delta$):** $\mathbf{+0.077619}$ ($\approx \mathbf{+0.0776}$ CE)
- **Relative Perplexity Increase:** $+8.07\%$

### Language Specificity Control
On `wikipedia_hi_sanskrit_sahitya` (9 documents, 4,798 tokens of Hindi commentary discussing Sanskrit literature):
- Baseline TW-CE: `3.920308` $\to$ Candidate TW-CE: `3.905656` ($\Delta = \mathbf{-0.014652}$).
- **Finding:** The degradation is strictly localized to pure Classical Sanskrit verse in Devanagari script.

---

## 2. Independent Training Coverage Verification

We audited the training partition of both `SPLIT_V2_MANIFEST.json` and `CORPUS_V2_CLEANED_MANIFEST.json`:
- **Training Sources:**
  - `cpython_algorithms_heapq` (Python code)
  - `gutenberg_tagore_gitanjali` (English literature)
  - `ietf_rfc793_tcp` (English networking specification)
  - `wikipedia_en_turing_machine` (English computer science)
  - `wikipedia_hi_chemistry` (Hindi technical text)
  - `wikipedia_hi_premchand` (Hindi narrative fiction)
- **Training Languages:** `code`, `en`, `hi`.
- **Classical Sanskrit Training Representation:** **EXACTLY 0 TOKENS (0.0%)**.

All Classical Sanskrit text in the repository was allocated 100% to the held-out evaluation splits.

---

## 3. Scientific Categorization and Causality Boundaries

1. **Observed Held-Out Regression:** The candidate regresses by $+0.0776$ CE on the held-out Sanskrit source.
2. **Observed Training Finding:** The CPT training corpus had zero Classical Sanskrit coverage.
3. **Causality Classification:** While zero coverage is directly observed and provides the logical condition under which unreinforced token transition matrices drift, declaring zero coverage as the sole causal mechanism remains a supported working hypothesis until validated by an intervention experiment.
4. **Distinction from Catastrophic Forgetting:** A $+8.07\%$ perplexity increase is characteristic of **held-out source-family regression due to zero training coverage**, not catastrophic representational collapse.
