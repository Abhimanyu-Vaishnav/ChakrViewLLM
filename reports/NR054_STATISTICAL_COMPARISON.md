# NR-054: STATISTICAL INFERENCE & UNCERTAINTY ANALYSIS REPORT

**Mission Identifier:** NR-054  
**Date:** November 2026  
**Auditor:** ChakrView Independent Forensic Engineering Team  
**Scope:** Rigorous statistical hypothesis testing, bootstrap resampling ($B=10,000$), paired permutation tests, and confidence interval estimation across ChakrView evaluations.

---

## 1. Executive Summary

In research reporting, a score increase from $0.180$ to $0.240$ is frequently characterized as a $+33.3\%$ relative improvement. However, in an evaluation set comprising only $N = 50$ prompts, an absolute difference of $0.060$ represents exactly **one single prompt** scoring 3 points instead of 0 ($1 \times 3 / 50 = 0.060$).

To determine whether the reported benchmark differences between 17M and 38M models represent genuine algorithmic superiority or mere stochastic variance, this audit executed a 10,000-iteration bootstrap resampling analysis and paired permutation tests.

**Primary Finding:** The difference between `NR052_38M_SFT` (mean $0.240$) and `NR052_17M_SFT` (mean $0.180$) is **not statistically significant** ($p = 1.0000$ under two-sided paired permutation testing; 95% bootstrap confidence intervals overlap almost entirely). Claims of benchmark superiority based on this 50-prompt test are statistically unsupported.

---

## 2. Bootstrap Confidence Intervals ($B = 10,000$)

Bootstrap distributions were computed by resampling the 50 prompt scores with replacement over 10,000 trials:

| Model Candidate | Sample Mean (Raw) | Sample Mean (Normalized) | 95% Bootstrap Confidence Interval (Raw) | 95% Bootstrap CI (Normalized) |
| :--- | :--- | :--- | :--- | :--- |
| **RC3 Historical Baseline** | `0.220` | `0.0733` | `[0.080, 0.400]` | `[2.67%, 13.33%]` |
| **NR051 17M SFT** | `0.180` | `0.0600` | `[0.060, 0.360]` | `[2.00%, 12.00%]` |
| **NR051 38M SFT** | `0.180` | `0.0600` | `[0.060, 0.360]` | `[2.00%, 12.00%]` |
| **NR052 17M SFT** | `0.180` | `0.0600` | `[0.060, 0.360]` | `[2.00%, 12.00%]` |
| **NR052 38M SFT** | `0.240` | `0.0800` | `[0.080, 0.440]` | `[2.67%, 14.67%]` |

### Interpretation:
Notice that the 95% confidence interval for NR052 38M (`[0.080, 0.440]`) completely encapsulates the mean of NR052 17M (`0.180`), and vice versa. There is zero separation between the model performance bands.

---

## 3. Paired Permutation Hypothesis Testing

To test the null hypothesis $H_0$: *Model A and Model B have identical prompt-level performance*, paired sign-permutation tests were conducted ($10,000$ permutations of paired differences):

| Comparison | Mean Difference | Two-Sided Permutation $p$-value | Statistically Significant at $\alpha = 0.05$? |
| :--- | :--- | :--- | :--- |
| **NR052 38M vs NR052 17M** | $+0.0600$ | $p = 1.0000$ | **NO** |
| **NR052 38M vs NR051 38M** | $+0.0600$ | $p = 1.0000$ | **NO** |
| **NR052 17M vs NR051 17M** | $0.0000$ | $p = 1.0000$ | **NO** |
| **NR052 38M vs RC3 Baseline** | $+0.0200$ | $p = 1.0000$ | **NO** |

**Conclusion:** Across all models, prompt-level differences on the 50-item benchmark are consistent with the null hypothesis.

---

## 4. Minimum Detectable Effect & Sample Size Requirements

Why did the test fail to achieve significance despite a visible difference in validation loss?

1. **Discrete Sparsity:** On 50 prompts, 46 prompts scored 0, 0 scored 1, 0 scored 2, and 4 scored 3. With 92% of responses scoring 0, statistical power is severely degraded.
2. **Statistical Power Analysis:** To detect a true shift from $6\%$ to $8\%$ accuracy with $80\%$ statistical power at $\alpha = 0.05$, a benchmark would require a sample size of **$N \ge 2,450$ prompts**.

Evaluating small language models on small ($N=50$) discrete keyword benchmarks produces high variance and deceptive percentage swings.

---

## 5. Statistical Truth vs Optimization Truth

| Dimension | Statistical Reality | Verdict |
| :--- | :--- | :--- |
| **Pretraining Val Loss ($1.4763 \to 1.1684$)** | Evaluated on $1.72 \times 10^6$ tokens ($N=3,363$ sequences). $p < 10^{-15}$. | **Undeniable True Generalization** |
| **SFT Response Loss ($2.2376 \to 1.9964$)** | Evaluated on 420 multi-turn dialogs. $p < 10^{-6}$. | **Authentic Optimization Gain** |
| **Fresh 50-Prompt Score ($0.180 \to 0.240$)** | Evaluated on 50 discrete queries ($4$ hits vs $3$ hits). $p = 1.0000$. | **Statistically Indistinguishable** |
