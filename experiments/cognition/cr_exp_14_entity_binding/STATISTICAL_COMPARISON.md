# CR-EXP-14 Phase 7: Statistical Comparison & Significance Analysis

## 1. Comparative Evaluation Matrix

| Metric | CR-11 Cand A | CR-12 Cand A | CR-12 Cand B (Parent) | Cand A (ESVC) | Cand B (ISUC) | Cand C (SED) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Total Benchmark Probes** | 1,200 | 1,200 | 1,200 | 1,200 | 1,200 | 1,200 |
| **Strict Accuracy (%)** | 5.00% | 8.33% | **8.33%** | 1.92% | 2.75% | 7.33% |
| **Strict Hits (Count)** | 60 | 100 | **100** | 23 | 33 | 88 |
| **State-Tracking Probes (Cat 1-6)** | 1,100 | 1,100 | 1,100 | 1,100 | 1,100 | 1,100 |
| **State-Tracking Strict Acc (%)** | **0.00%** | **0.00%** | **0.00%** | **0.00%** | **0.00%** | **0.00%** |
| **Abstention Probes (Cat 7)** | 100 | 100 | 100 | 100 | 100 | 100 |
| **Abstention Accuracy (%)** | 60.0% | **100.0%** | **100.0%** | 23.0% | 33.0% | 88.0% |
| **Training Steps** | — | — | — | 250 | 250 | 250 |
| **Best Val Loss** | — | — | — | **1.7072** | 1.7758 | 3.8244 |

---

## 2. Statistical Significance Analysis

### McNemar's Test on State-Tracking Probes (Cat 1-6, $N=1100$)
- Across all models (CR-11 Cand A, CR-12 Cand A, CR-12 Cand B, Candidate A, Candidate B, Candidate C), performance on Categories 1-6 is identically **0 hits out of 1,100 probes** ($0.00\%$).
- Difference in proportions $\Delta = 0.000$, $\chi^2 = 0.0, p = 1.000$.
- Conclusion: There is no statistically significant improvement in genuine out-of-distribution subject-value binding across any of the experimental interventions.

### Fisher's Exact Test on Abstention Probes ($N=100$)
- Comparing CR-12 Parent ($100/100$) vs. Candidate A ($23/100$):
  - Two-tailed $p < 10^{-15}$ (statistically significant regression in unanswerability calibration).
- Comparing CR-12 Parent ($100/100$) vs. Candidate B ($33/100$):
  - Two-tailed $p < 10^{-15}$ (statistically significant regression in unanswerability calibration).
- Comparing CR-12 Parent ($100/100$) vs. Candidate C ($88/100$):
  - Two-tailed $p = 0.0003$ (minor regression, but preserved significantly better than Candidates A and B).

---

## 3. Statistical Power & Floor Assessment
With $N = 1,200$ total probes and $N = 1,100$ capability probes, the statistical power to detect an effect size of $\ge 1\%$ accuracy at $\alpha = 0.05$ is $> 0.99$. The observed strict accuracy of $0.00\%$ is an exact empirical floor, confirming that zero model configurations acquired true out-of-distribution entity binding.
