# HELDOUT REGRESSION FINAL RECHECK REPORT

**Batch:** 2461–2500  
**Audit Phase:** Phase 3 — Independent Verification of Held-Out Regression  
**Primary Evidence File:** [`experiments/batch2381_2420/PER_DOCUMENT_LOSSES.jsonl`](file:///d:/Project/ChakrView/experiments/batch2381_2420/PER_DOCUMENT_LOSSES.jsonl)  
**Machine-Readable Summary:** [`experiments/batch2461_2500/HELDOUT_REGRESSION_FINAL_RECHECK.json`](file:///d:/Project/ChakrView/experiments/batch2461_2500/HELDOUT_REGRESSION_FINAL_RECHECK.json)  
**Status:** **INDEPENDENTLY REPRODUCED AND VERIFIED (EXACT MATCH)**  

---

## 1. Executive Summary & Verification Methodology

In this phase, we recomputed all baseline and candidate loss metrics directly from raw per-document records without relying on intermediate summaries. We evaluated the 290 individual test document outputs, derived source-level and global negative log-likelihood ($NLL$) totals, and verified document target-token alignments.

### Mathematical Conventions:
- **Target Tokens ($T$):** Sum of next-token prediction targets over chunks of length up to 256.
- **Negative Log-Likelihood ($\text{NLL}$):** $\sum_{i} (\text{loss}_i \times T_i)$.
- **Token-Weighted Cross-Entropy ($\text{TW-CE}$):** $\frac{\sum \text{NLL}}{\sum T}$.
- **Perplexity ($\text{PPL}$):** $\exp(\text{TW-CE})$.
- **Relative PPL Change:** $\frac{\text{PPL}_{\text{cand}} - \text{PPL}_{\text{base}}}{\text{PPL}_{\text{base}}} \times 100\%$.

---

## 2. Recomputed Results vs Batch 2421–2460 Matrix

| Metric Dimension | Batch 2421 Derived | Batch 2461 Recomputed | Discrepancy | Status |
| :--- | :---: | :---: | :---: | :---: |
| **Total Test Documents** | 290 | 290 | 0 | Bit-Exact |
| **Total Target Tokens** | 64,769 | 64,769 | 0 | Bit-Exact |
| **Baseline Global NLL** | 290,094.479876 | 290,094.479876 | $0.000000$ | Bit-Exact |
| **Candidate Global NLL** | 277,607.593446 | 277,607.593446 | $0.000000$ | Bit-Exact |
| **Global Baseline CE** | 4.478909 | 4.478909 | $0.000000$ | Bit-Exact |
| **Global Candidate CE** | 4.286118 | 4.286118 | $0.000000$ | Bit-Exact |
| **Global Delta CE** | **-0.192791** | **-0.192791** | $0.000000$ | Bit-Exact |
| **Candidate Win Rate** | 76.55% (222/290) | 76.55% (222/290) | $0.00\%$ | Bit-Exact |

---

## 3. Source-Level Breakdown & Sanskrit Regression

| Source ID | Docs | Tokens | Base CE | Cand CE | Delta CE | Base PPL | Cand PPL | Rel. PPL Change |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `gutenberg_loney_coordinate_geometry` | 150 | 24,482 | 4.431451 | 4.149476 | **-0.281975** | 84.0536 | 63.3999 | -24.57% |
| `wikipedia_en_information_theory` | 70 | 15,912 | 4.915680 | 4.473698 | **-0.441982** | 136.4114 | 87.6798 | -35.73% |
| `wikipedia_hi_sanskrit_sahitya` (Control) | 9 | 4,798 | 3.920308 | 3.905656 | **-0.014652** | 50.4162 | 49.6826 | -1.45% |
| `wikipedia_sa_bhagavad_gita` (Sanskrit) | 61 | 19,577 | 4.320159 | 4.397778 | **+0.077619** | 75.2008 | 81.2701 | **+8.07%** |

### Key Observations:
1. **Classical Sanskrit Held-Out Regression:** The candidate model increased cross-entropy by **+0.077619** on `wikipedia_sa_bhagavad_gita`, which corresponds to an **+8.07%** increase in perplexity.
2. **Hindi Commentary Control:** The control source `wikipedia_hi_sanskrit_sahitya` showed a slight improvement of **-0.014652 CE** (-1.45% PPL), demonstrating that the regression is specific to Classical Sanskrit and not a degradation in general Devanagari script processing.
3. **Precision Consistency:** The delta between the Batch 2421 derived values and Batch 2461 recomputed values is exactly $0.000000000000$, confirming mathematical reproducibility.

---

## 4. Separation of Empirical Fact from Causal Inference
- **Empirical Observation:** The held-out Classical Sanskrit source degraded by +0.0776 CE (+8.07% PPL) while all other evaluated sources improved.
- **Causal Hypothesis:** The regression was driven by zero training coverage for Classical Sanskrit in the CPT split while other domains received continued pretraining.
- **Conclusion:** While consistent with the zero-coverage finding, causal attribution requires pre-registered experimental validation with an independent Sanskrit-augmented training curriculum.
