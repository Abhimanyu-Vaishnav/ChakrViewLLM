# INDEPENDENT HELDOUT METRICS RECOMPUTATION REPORT

**Batch:** 2501–2540  
**Audit Phase:** Phase 4 — Independent Recomputation of Held-Out Benchmark Metrics  
**Raw Evidence File:** [`experiments/batch2381_2420/PER_DOCUMENT_LOSSES.jsonl`](file:///d:/Project/ChakrView/experiments/batch2381_2420/PER_DOCUMENT_LOSSES.jsonl)  
**Machine-Readable Metrics:** [`experiments/batch2501_2540/HELDOUT_METRICS_INDEPENDENT.json`](file:///d:/Project/ChakrView/experiments/batch2501_2540/HELDOUT_METRICS_INDEPENDENT.json)  
**Reconciliation Record:** [`experiments/batch2501_2540/HELDOUT_METRICS_RECONCILIATION.json`](file:///d:/Project/ChakrView/experiments/batch2501_2540/HELDOUT_METRICS_RECONCILIATION.json)  
**Status:** **INDEPENDENTLY RECOMPUTED AND BIT-EXACT RECONCILED**  

---

## 1. Executive Summary & Audit Mandate

In Phase 4, we independently validated every individual record in [`PER_DOCUMENT_LOSSES.jsonl`](file:///d:/Project/ChakrView/experiments/batch2381_2420/PER_DOCUMENT_LOSSES.jsonl), verified ID-set equality with the frozen split manifest, and re-accumulated all loss, cross-entropy, perplexity, and delta statistics from raw floats.

### Pre-Reconciliation Integrity Checks:
1. **Document ID Uniqueness:** 290 unique test document IDs.
2. **Set Equality:** Exactly matches $\text{Test IDs in } \texttt{SPLIT\_V2\_MANIFEST.json}$ (0 missing, 0 unexpected).
3. **Record Field Integrity:** All target-token counts are positive integers; all NLL and loss values are non-negative, finite real numbers.
4. **Internal Consistency:** For every record $i$, $\text{NLL}_i = \text{Loss}_i \times \text{Tokens}_i$ within floating point tolerance ($< 10^{-4}$).

---

## 2. Recomputed Global Benchmark Metrics

- **Total Test Documents:** 290
- **Total Evaluated Target Tokens:** 64,769
- **Baseline Global NLL Accumulation:** $290,094.479876$
- **Candidate Seed 42 Global NLL Accumulation:** $277,607.593446$
- **Baseline Token-Weighted Cross-Entropy:** $4.478909$ (PPL: 88.1388)
- **Candidate Token-Weighted Cross-Entropy:** $4.286118$ (PPL: 72.6841)
- **Overall Delta Cross-Entropy ($\Delta$):** **$-0.192791$**
- **Candidate Document Win Count:** 222 wins / 290 documents
- **Candidate Win Rate:** **76.55%**

---

## 3. Source-Family Recomputed Metrics & Regression

| Source ID | Docs | Tokens | Base NLL | Cand NLL | Base CE | Cand CE | Delta CE | Base PPL | Cand PPL | Rel. PPL Change |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `gutenberg_loney_coordinate_geometry` | 150 | 24,482 | 108,490.78 | 101,587.47 | 4.431451 | 4.149476 | **-0.281975** | 84.0536 | 63.3999 | -24.57% |
| `wikipedia_en_information_theory` | 70 | 15,912 | 78,218.30 | 71,185.49 | 4.915680 | 4.473698 | **-0.441982** | 136.4114 | 87.6798 | -35.73% |
| `wikipedia_hi_sanskrit_sahitya` (Hindi Control) | 9 | 4,798 | 18,809.64 | 18,739.34 | 3.920308 | 3.905656 | **-0.014652** | 50.4162 | 49.6826 | -1.45% |
| `wikipedia_sa_bhagavad_gita` (Classical Sanskrit) | 61 | 19,577 | 84,575.76 | 86,095.30 | 4.320159 | 4.397778 | **+0.077619** | 75.2008 | 81.2701 | **+8.07%** |

---

## 4. Reconciliation with Prior Batch Comparison Targets

| Metric Target | Target Value | Independently Derived | Absolute Difference | Reconciliation Status |
| :--- | :---: | :---: | :---: | :---: |
| **Global Baseline CE** | 4.478909 | 4.478909 | $0.000000$ | **EXACT MATCH** |
| **Candidate Seed 42 CE** | 4.286118 | 4.286118 | $0.000000$ | **EXACT MATCH** |
| **Sanskrit Delta CE** | +0.077619 | +0.077619 | $0.000000$ | **EXACT MATCH** |
| **Hindi Control Delta CE**| -0.014652 | -0.014652 | $0.000000$ | **EXACT MATCH** |

---

## 5. Summary Finding

All metrics are confirmed without deviation. The Classical Sanskrit held-out regression (+0.0776 CE, +8.07% PPL) and the Hindi commentary stability (-0.0147 CE) are robust mathematical realities of the candidate checkpoint relative to the baseline.
