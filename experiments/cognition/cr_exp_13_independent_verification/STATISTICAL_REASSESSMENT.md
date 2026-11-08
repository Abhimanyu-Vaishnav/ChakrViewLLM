# CR-EXP-13 STATISTICAL REASSESSMENT REPORT
**Investigation:** Paired vs. Unpaired Statistical Reassessment of CR-EXP-12  
**Auditor:** Autonomous Scientific Auditor & Statistical Validation Engineer  
**Date:** 2026-11-08T10:06:00+05:30  
**Sample Size:** $N = 1,500$ identical paired probes  

---

## 1. REASSESSMENT OF REPORTED P-VALUE ($p = 0.004739$)
In CR-EXP-12, the report computed a **Two-Proportion Z-Test**:
- $z = 2.8243, \quad p = 0.004739$
- **Methodological Error**: Two-proportion Z-tests assume independent samples. Because the two models were evaluated on the **exact same 1,500 probes**, the samples are correlated and strictly paired.
- **Mandatory Correct Test**: The statistically rigorous test for paired binary outcomes is **McNemar's Test with Edwards' Continuity Correction**.

---

## 2. EXACT PAIRED MCNEMAR TESTS ($N=1,500$)

### A. Parent (CR-11 Candidate A) vs. Champion (CR-12 Candidate B)
```
                               CR-12 Candidate B Correct    CR-12 Candidate B Incorrect
CR-11 Cand A Correct                     287                            153 (n10)
CR-11 Cand A Incorrect                   225 (n01)                      835
```
- **Discordant Pairs ($b + c$):** $153 + 225 = 378$
- **McNemar Chi-Square (with continuity correction):**
  $$\chi^2 = \frac{(|153 - 225| - 1)^2}{153 + 225} = \frac{(71)^2}{378} = 13.3360$$
- **Exact Two-Sided P-Value:**
  $$p = 0.00026036 \quad (p = 2.60 \times 10^{-4})$$
- **Odds Ratio:** $1.4706$ ($95\%\text{ CI: } [1.196, 1.808]$)
- **Statistical Verdict**: **HIGHLY STATISTICALLY SIGNIFICANT**.
  Correcting from the unpaired Z-test ($p = 0.0047$) to the paired McNemar test actually **strengthens** the statistical significance by more than an order of magnitude ($p = 0.00026 < 0.001$). Candidate B is unequivocally superior to parent Candidate A across the 1,500-probe suite.

---

### B. Parent (CR-11 Candidate A) vs. Memory Specialist (CR-12 Candidate A)
- Discordant Pairs: $163 + 201 = 364$
- $\chi^2 = 3.7610$
- **P-Value:** $p = 0.05246$
- **Statistical Verdict**: **MARGINALLY NON-SIGNIFICANT AT $\alpha = 0.05$**.
  Candidate A's aggregate gain ($31.87\%$ vs $29.33\%$) is not statistically significant once multiple comparisons (Holm adjustment) are taken into account.

---

### C. CR-12 Candidate A vs. CR-12 Candidate B
- Discordant Pairs: $118 + 152 = 270$
- $\chi^2 = 4.0333$
- **P-Value:** $p = 0.04461$
- **Statistical Verdict**: Candidate B is statistically significantly superior to Candidate A ($p < 0.05$).

---

## 3. HOLM-BONFERRONI MULTIPLE COMPARISONS ADJUSTMENT
1. Parent vs. Candidate B: $p_{\text{raw}} = 0.00026 \times 3 = \mathbf{0.00078}$ (**SIGNIFICANT**, $p < 0.01$).
2. Candidate A vs. Candidate B: $p_{\text{raw}} = 0.0446 \times 2 = \mathbf{0.0892}$ (NOT SIGNIFICANT after FWER control).
3. Parent vs. Candidate A: $p_{\text{raw}} = 0.0525 \times 1 = \mathbf{0.0525}$ (NOT SIGNIFICANT).

**Final Statistical Conclusion**:
Only **CR-12 Candidate B vs. Parent CR-11 Candidate A** survives strict family-wise error rate control ($p < 0.001$). Candidate B is the sole statistically confirmed improvement.
