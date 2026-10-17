# LOSS AGGREGATION AUDIT REPORT

**Batch:** 2381–2420  
**Audit Phase:** Phase 2 — Raw Loss Accumulation and Mathematical Aggregation Audit  
**Status:** FULLY REPRODUCED AND VERIFIED  

---

## 1. Executive Summary

This audit independently recomputed the exact floating-point negative log-likelihood (NLL) and token-weighted cross-entropy (TW-CE) for both the Canonical Baseline (Step 10,000) and the CPT Candidate Seed 42 across all 290 test documents ($64,769$ target tokens) using our reproducible evaluation script [run_isolated_evaluation.py](file:///D:/Project/ChakrView/experiments/batch2381_2420/run_isolated_evaluation.py).

Every document was evaluated using identical sequence windowing (stride $255$, chunk size $256$), canonical tokenizer artifacts, and identical target-token loss masks.

---

## 2. Recomputed Primary Evaluation Results

| Metric | Canonical Baseline | CPT Candidate Seed 42 | Delta ($\Delta$) | Direction |
| :--- | :---: | :---: | :---: | :---: |
| **Exact Accumulated NLL** | **290,094.479876** | **277,607.593446** | -12,486.886430 | Improved |
| **Total Target Tokens** | 64,769 | 64,769 | 0 | Invariant |
| **Exact Token-Weighted CE** | **4.478909** | **4.286118** | **-0.192791** | **Improved** |
| **TW-CE (Rounded 4 d.p.)** | **4.4789** | **4.2861** | **-0.1928** | **Improved** |
| **Perplexity (PPL)** | **88.14** | **72.68** | **-15.46** | **Improved** |
| **Document Win Rate** | — | **76.55%** (222 / 290) | — | Substantial Win |

### Source-Level Breakdown
Across all 4 test source families:

1. `gutenberg_loney_coordinate_geometry` (150 docs, 24,482 tokens):
   - Baseline TW-CE: `4.431451` (PPL: 84.05) $\to$ Candidate: `4.149476` (PPL: 63.40) | $\Delta = \mathbf{-0.281975}$
2. `wikipedia_en_information_theory` (70 docs, 15,912 tokens):
   - Baseline TW-CE: `4.915680` (PPL: 136.41) $\to$ Candidate: `4.473698` (PPL: 87.68) | $\Delta = \mathbf{-0.441982}$
3. `wikipedia_hi_sanskrit_sahitya` (9 docs, 4,798 tokens):
   - Baseline TW-CE: `3.920308` (PPL: 50.42) $\to$ Candidate: `3.905656` (PPL: 49.68) | $\Delta = \mathbf{-0.014652}$
4. `wikipedia_sa_bhagavad_gita` (61 docs, 19,577 tokens):
   - Baseline TW-CE: `4.320159` (PPL: 75.20) $\to$ Candidate: `4.397778` (PPL: 81.27) | $\Delta = \mathbf{+0.077619}$

---

## 3. Mathematical Resolution of the Stored Loss Discrepancy

Historical reports cited two slightly differing loss sums:
- Figure 1: `290,094.04` (Baseline) / `277,605.34` (Seed 42)
- Figure 2: `290,096.76` (Baseline) / `277,608.85` (Seed 42)

Our audit measured all three accumulation paths:
1. **True Evaluator Unrounded Float NLL:** Baseline = **290,094.48** | Seed 42 = **277,607.59**
2. **Sum of Document 4-decimal Losses $\times$ Tokens:** Baseline = **290,094.46** | Seed 42 = **277,607.58**
3. **Sum of Source 4-decimal CEs $\times$ Tokens:** Baseline = **290,096.76** | Seed 42 = **277,608.85**

The variance of $2.28$ units on baseline ($< 0.00079\%$) is mathematically proven to be an artifact of intermediate rounding when multiplying 4-decimal-place rounded averages by thousands of tokens. All methods yield identical 4-decimal cross-entropies: **4.4789** and **4.2861**.
