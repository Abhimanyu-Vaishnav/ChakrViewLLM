# CR-EXP-15 Statistical Comparison
## Rigorous Significance Testing Across Evaluated Models

**Mission ID:** CR-EXP-15  
**Test Population:** $N = 1,200$ sealed benchmark probes (1,050 non-abstention, 150 abstention).

---

## 1. Metric Comparison Matrix

| Model Key | Parameters | Val Loss | Overall Acc | Strict Acc (Non-Abs) | Subject Acc (Non-Abs) | Value Acc (Non-Abs) | Abstention Acc |
|---|---|---|---|---|---|---|---|
| `baseline_parent` | 3.44M | 5.7251 | 0.000% | 0.000% (0/1050) | 0.000% (0/1050) | 0.095% (1/1050) | 0.000% (0/150) |
| `candidate_a` | 3.44M | 5.0537 | 0.000% | 0.000% (0/1050) | 0.000% (0/1050) | 0.095% (1/1050) | 0.000% (0/150) |
| `candidate_b` | 3.44M | 5.1228 | 0.000% | 0.000% (0/1050) | 0.095% (1/1050) | 0.000% (0/1050) | 0.000% (0/150) |
| `candidate_c` | 3.44M | 4.9690 | 0.000% | 0.000% (0/1050) | 0.000% (0/1050) | 0.000% (0/1050) | 0.000% (0/150) |
| `candidate_d` | 3.44M | 1.8418 | 0.000% | 0.000% (0/1050) | 0.000% (0/1050) | 0.000% (0/1050) | 0.000% (0/150) |
| `arch_arm_b_heads3_d64` | 3.44M | 6.0375 | 0.000% | 0.000% (0/1050) | 0.000% (0/1050) | 0.000% (0/1050) | 0.000% (0/150) |
| `arch_arm_c_width256_d64` | 5.77M | 5.2227 | 0.000% | 0.000% (0/1050) | 0.000% (0/1050) | 0.000% (0/1050) | 0.000% (0/150) |

---

## 2. Statistical Significance Analysis

### McNemar's Test for Paired Proportions
Comparing any candidate model against the parent baseline:
- Both parent and candidates scored $0/1,050$ on strict entity-value binding.
- Number of discordant pairs $(b=0, c=0)$.
- Test statistic $\chi^2 = 0.0, \; p = 1.000$.
- **Conclusion:** There is **no statistically significant difference** in entity binding capability between the baseline and any candidate or architectural variant under test.

### Confidence Intervals (Wilson Score Interval)
For all models on non-abstention strict accuracy:
- Observed successes: $k = 0$, trials: $n = 1,050$.
- 95% Wilson Confidence Interval: $[0.000\%, 0.351\%]$.
- 99% Wilson Confidence Interval: $[0.000\%, 0.630\%]$.

This confirms that the true population generalization accuracy of all evaluated models on unseen entity binding is bounded below $0.35\%$ ($p < 0.05$).

### Correlation Between Validation Loss and Strict Accuracy
- Candidate D achieved a dramatic drop in validation loss from $5.7251 \to 1.8418$ ($\Delta = -3.8833$).
- Despite this 67.8% relative reduction in loss, strict accuracy remained exactly $0.000\%$.
- **Statistical finding:** Pearson correlation $r = 0$ between validation cross-entropy loss on state-update text and generalization accuracy on OOD subject-value tracking. Cross-entropy loss on templated sequences is a deceptive surrogate metric when parametric memorization is rewarded.

