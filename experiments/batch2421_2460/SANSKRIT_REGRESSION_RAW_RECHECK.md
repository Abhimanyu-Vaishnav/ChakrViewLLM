# SANSKRIT HELD-OUT REGRESSION INDEPENDENT RAW RECHECK REPORT

**Batch:** 2421–2460  
**Audit Phase:** Phase 2 — Raw Record Evaluation & Regression Verification  
**Primary Evidence File:** [`experiments/batch2381_2420/PER_DOCUMENT_LOSSES.jsonl`](file:///d:/Project/ChakrView/experiments/batch2381_2420/PER_DOCUMENT_LOSSES.jsonl)  
**Machine-Readable Summary:** [`experiments/batch2421_2460/SANSKRIT_REGRESSION_RAW_RECHECK.json`](file:///d:/Project/ChakrView/experiments/batch2421_2460/SANSKRIT_REGRESSION_RAW_RECHECK.json)  
**Status:** **VERIFIED FROM PRIMARY RAW RECORDS**  

---

## 1. Executive Summary & Verification Objective

The objective of Phase 2 is to avoid relying on pre-computed summary metrics (such as `FRESH_EVALUATION.json` or `SANSKRIT_COVERAGE_AND_REGRESSION_AUDIT.json`) and instead recompute every aggregation directly from the 290 individual document records in [`PER_DOCUMENT_LOSSES.jsonl`](file:///d:/Project/ChakrView/experiments/batch2381_2420/PER_DOCUMENT_LOSSES.jsonl).

We independently verified:
1. Every record contains valid model outputs, unambiguous document IDs, source IDs, token counts, and exact accumulated negative log-likelihoods ($NLL$).
2. Both models (Canonical Baseline and CPT Seed 42 Candidate) evaluated identical document sets with identical target-token counts.
3. Source-level and global $NLL$ and token totals equal the exact sums from the raw records.
4. Source-level cross-entropy ($CE$) is derived strictly as $\frac{\sum \text{NLL}}{\sum \text{Target Tokens}}$.
5. The Sanskrit held-out regression delta and the Hindi commentary control delta are computed independently from raw document sums.

---

## 2. Raw Record Structural Verification

- **Total Document Records:** 290
- **Unique Document IDs:** 290 (0 duplicates, 0 missing)
- **Token Count Agreement:** Across all 290 documents, target-token counts are identical between baseline and candidate records.
- **NLL Consistency:** For every record $i$, $\text{NLL}_i = \text{Loss}_i \times \text{Tokens}_i$ within floating point precision ($| \Delta | < 10^{-4}$).

---

## 3. Global and Source-Level Mathematical Re-Derivation

### Global Totals
- **Total Evaluated Target Tokens:** 64,769
- **Baseline Global NLL Sum:** $290,094.479876$
- **Candidate Seed 42 Global NLL Sum:** $277,607.593446$
- **Derived Baseline Token-Weighted CE:** $4.478909$
- **Derived Candidate Token-Weighted CE:** $4.286118$
- **Derived Global Delta CE:** **$-0.192791$** (Overall candidate improvement across the test split)

### Source-Level Breakdown (Derived Directly from Raw Sums)

| Source ID | Docs | Tokens | Baseline NLL | Candidate NLL | Baseline CE | Candidate CE | Delta CE ($\Delta$) | Rel. PPL Change | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `gutenberg_loney_coordinate_geometry` | 150 | 24,482 | 108,490.7818 | 101,587.4697 | 4.431451 | 4.149476 | **-0.281975** | -24.57% | Improved |
| `wikipedia_en_information_theory` | 70 | 15,912 | 78,218.2974 | 71,185.4851 | 4.915680 | 4.473698 | **-0.441982** | -35.73% | Improved |
| `wikipedia_hi_sanskrit_sahitya` (Hindi Control) | 9 | 4,798 | 18,809.6373 | 18,739.3370 | 3.920308 | 3.905656 | **-0.014652** | -1.45% | No Regression |
| `wikipedia_sa_bhagavad_gita` (Classical Sanskrit) | 61 | 19,577 | 84,575.7634 | 86,095.3016 | 4.320159 | 4.397778 | **+0.077619** | **+8.07%** | **Regression** |

---

## 4. Specific Regression & Control Findings

### A. Classical Sanskrit Held-out Regression (`wikipedia_sa_bhagavad_gita`)
- **Document Count:** 61 documents
- **Evaluated Target Tokens:** 19,577 tokens
- **Raw Derived Baseline CE:** 4.320159 (PPL: 75.2008)
- **Raw Derived Candidate CE:** 4.397778 (PPL: 81.2701)
- **Derived Regression Delta CE:** **+0.077619** (+0.0776 to 4 decimal places)
- **Relative Perplexity Increase:** **+8.07%**
- **Conclusion:** The regression in Classical Sanskrit is fully reproduced and verified from raw document records.

### B. Hindi Commentary Control (`wikipedia_hi_sanskrit_sahitya`)
- **Document Count:** 9 documents
- **Evaluated Target Tokens:** 4,798 tokens
- **Raw Derived Baseline CE:** 3.920308
- **Raw Derived Candidate CE:** 3.905656
- **Derived Control Delta CE:** **-0.014652**
- **Conclusion:** The Hindi commentary discussing Sanskrit literature experienced a slight loss reduction (-0.0147 CE), confirming that Devanagari script processing itself did not regress, but the specific Classical Sanskrit source did.

---

## 5. Verification Conclusion

Phase 2 confirms with zero ambiguity that the reported metric values are exact mathematical derivations from primary per-document records.
