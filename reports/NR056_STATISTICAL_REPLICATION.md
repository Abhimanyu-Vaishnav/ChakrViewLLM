# NR-056: STATISTICAL REPLICATION & MULTIPLE-COMPARISON AUDIT

**Mission Identifier:** NR-056  
**Date:** November 28, 2026  
**Auditor:** ChakrView Independent Forensic Engineering Team  
**Scope:** Independent recomputation of NR-055 statistical comparisons, bootstrap confidence intervals, and application of Holm-Bonferroni family-wise error rate corrections.

---

## 1. Executive Summary

Mission NR-055 reported that `NR052_38M_SFT` scored statistically significantly higher than `NR051_38M_SFT` ($p = 0.0288$) and `NR051_17M_SFT` ($p = 0.0043$) under uncorrected paired permutation testing.

However, when comparing multiple candidates against each other across $m = 10$ pairs, uncorrected $p$-values suffer from inflated Type I error rates. This audit applied the standard **Holm-Bonferroni multiple-comparison correction** to determine whether these advantages remain statistically defensible.

---

## 2. Recomputed Descriptive Statistics & Bootstrap CIs

| Candidate Identifier | Parameters | Sample Mean (/3.0) | Std Dev | Raw Score Sum | 95% Bootstrap CI (Raw) | 95% Bootstrap CI (Norm %) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **RC3 Baseline** | 17.06M | 0.1368 | 0.5097 | 13 | `[0.0421, 0.2421]` | `[1.40%, 8.07%]` |
| **NR051 17M SFT** | 17.06M | 0.0000 | 0.0000 | 0 | `[0.0000, 0.0000]` | `[0.00%, 0.00%]` |
| **NR051 38M SFT** | 37.50M | 0.0632 | 0.3515 | 6 | `[0.0000, 0.1474]` | `[0.00%, 4.91%]` |
| **NR052 17M SFT** | 17.06M | 0.0842 | 0.4035 | 8 | `[0.0211, 0.1684]` | `[0.70%, 5.61%]` |
| **NR052 38M SFT** | 37.50M | 0.1789 | 0.5739 | 17 | `[0.0737, 0.2947]` | `[2.46%, 9.82%]` |

---

## 3. Paired Permutation Hypothesis Tests & Holm Adjustments

Paired sign-permutation tests ($B = 10,000$ iterations) were re-executed on the 95 paired items:

| Comparison Pair | Observed Mean Difference | Uncorrected Raw $p$-value | Holm-Bonferroni Adjusted $p$-value | Statistically Significant ($\alpha = 0.05$)? |
| :--- | :---: | :---: | :---: | :---: |
| **NR052 38M vs NR051 17M** | $+0.1789$ | **$p = 0.0043$** | **$p_{\text{adj}} = 0.0430$** | **YES (Statistically Significant)** |
| **RC3 Baseline vs NR051 17M** | $+0.1368$ | **$p = 0.0156$** | **$p_{\text{adj}} = 0.1404$** | **NO (Fails Holm Correction)** |
| **NR052 38M vs NR051 38M** | $+0.1158$ | **$p = 0.0288$** | **$p_{\text{adj}} = 0.2304$** | **NO (Fails Holm Correction)** |
| **NR052 17M vs NR051 17M** | $+0.0842$ | **$p = 0.0455$** | **$p_{\text{adj}} = 0.3185$** | **NO (Fails Holm Correction)** |
| **NR052 38M vs NR052 17M** | $+0.0947$ | **$p = 0.1244$** | **$p_{\text{adj}} = 0.7464$** | **NO** |
| **RC3 Baseline vs NR051 38M** | $+0.0737$ | **$p = 0.2440$** | **$p_{\text{adj}} = 1.0000$** | **NO** |
| **RC3 Baseline vs NR052 17M** | $+0.0526$ | **$p = 0.4206$** | **$p_{\text{adj}} = 1.0000$** | **NO** |
| **NR052 38M vs RC3 Baseline**| $+0.0421$ | **$p = 0.5024$** | **$p_{\text{adj}} = 1.0000$** | **NO** |
| **NR051 38M vs NR051 17M** | $+0.0632$ | **$p = 0.0833$** | **$p_{\text{adj}} = 0.5831$** | **NO** |

---

## 4. Scientific Interpretation

1. **Only ONE Comparison Survives Multiple-Comparison Correction:**
   - Under rigorous family-wise error control, only the difference between `NR052_38M_SFT` and the completely non-functional `NR051_17M_SFT` achieves statistical significance ($p_{\text{adj}} = 0.0430$).
2. **The 38M vs 17M Benchmark Superiority Claim is Formally Disproven:**
   - On Benchmark V2 (95 items), the performance gap between `NR052_38M_SFT` (17 pts) and `NR052_17M_SFT` (8 pts) has $p_{\text{raw}} = 0.1244$ and $p_{\text{adj}} = 0.7464$.
   - The gap between `NR052_38M_SFT` (17 pts) and `RC3_Baseline` (13 pts) has $p_{\text{raw}} = 0.5024$ and $p_{\text{adj}} = 1.0000$.
3. **Conclusion:** On a 95-item benchmark, **no model possesses a statistically defensible capability advantage over RC3 Baseline or NR-052 17M**.
