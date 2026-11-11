# CR-EXP-16F: Statistical Analysis & Factorial Effect Sizes

## 1. Study Design & Cell Summary

A $2 \times 2$ factorial experiment was conducted with $N=3$ independent training replications (seeds 2026, 3030, 4040) per cell:
- **Factor 1 (Replay):** 25% Copying Replay vs. 50% Copying Replay.
- **Factor 2 (Lambda):** $\lambda = 1.00$ (Full negative abstention loss) vs. $\lambda = 0.25$ (Attenuated negative loss).

All metrics report sample mean and sample standard deviation $s = \sqrt{\frac{1}{N-1}\sum_{i=1}^N (x_i - \bar{x})^2}$.

### Cell Performance Summary Table

| Metric | Arm A (25%, $\lambda=1.00$) | Arm B (50%, $\lambda=1.00$) | Arm C (25%, $\lambda=0.25$) | Arm D (50%, $\lambda=0.25$) |
| :--- | :---: | :---: | :---: | :---: |
| **CR-16 Copying (/100)** | $0.67 \pm 0.58$ | $1.00 \pm 0.00$ | $0.67 \pm 0.58$ | $0.33 \pm 0.58$ |
| **Answerable Retrieval (/1,400)** | $65.33 \pm 23.76$ | $67.33 \pm 17.79$ | $145.67 \pm 24.09$ | $127.33 \pm 61.37$ |
| **False Abstention Rate (/1,400)** | $1045.67 \pm 124.74$ | $960.67 \pm 131.09$ | $596.33 \pm 77.15$ | $591.33 \pm 370.65$ |
| **Unanswerable CAT9 Recall (/100)** | $92.33 \pm 8.33$ | $80.67 \pm 2.08$ | $47.33 \pm 7.57$ | $46.00 \pm 27.78$ |
| **B1 Position Invariance (/600)** | $136.33 \pm 10.97$ | $133.00 \pm 16.09$ | $168.00 \pm 16.46$ | $146.33 \pm 11.24$ |
| **B2 Permutation Pairs (/150)** | $0.33 \pm 0.58$ | $1.00 \pm 1.00$ | $10.00 \pm 7.00$ | $9.00 \pm 4.58$ |
| **B2 Counterfactual Pairs (/150)** | $0.67 \pm 0.58$ | $2.67 \pm 0.58$ | $13.67 \pm 1.53$ | $10.00 \pm 8.89$ |
| **B3 Multi-Token Spans (/500)** | $67.67 \pm 15.70$ | $80.67 \pm 9.29$ | $148.00 \pm 9.54$ | $122.67 \pm 45.96$ |
| **B4 Calibrated Abstention (/300)** | $289.33 \pm 10.02$ | $270.67 \pm 12.01$ | $201.00 \pm 14.80$ | $180.33 \pm 68.72$ |
| **CAT7 Unseen-Value (/150)** | $0.00 \pm 0.00$ | $0.00 \pm 0.00$ | $0.00 \pm 0.00$ | $0.00 \pm 0.00$ |

---

## 2. Two-Way Factorial ANOVA

We conducted a two-factor fixed-effects ANOVA ($df_1 = 1, df_2 = 8$) for each dimension. Partial eta-squared ($\eta_p^2 = \frac{SS_{\text{effect}}}{SS_{\text{effect}} + SS_{\text{error}}}$) measures effect size.

| Dependent Variable | Factor: Replay Fraction ($F, \eta_p^2$) | Factor: Abstention Multiplier $\lambda$ ($F, \eta_p^2$) | Interaction Replay $\times \lambda$ ($F, \eta_p^2$) |
| :--- | :---: | :---: | :---: |
| **CR-16 Copying** | $F = 0.00, \eta_p^2 = 0.000$ | $F = 1.33, \eta_p^2 = 0.143$ | $F = 1.33, \eta_p^2 = 0.143$ |
| **Answerable Retrieval** | $F = 0.15, \eta_p^2 = 0.019$ | **$F = 11.30, \eta_p^2 = 0.586$** | $F = 0.24, \eta_p^2 = 0.029$ |
| **False Abstention** | $F = 0.14, \eta_p^2 = 0.017$ | **$F = 11.42, \eta_p^2 = 0.588$** | $F = 0.11, \eta_p^2 = 0.013$ |
| **Unanswerable Recall** | $F = 0.56, \eta_p^2 = 0.066$ | **$F = 21.09, \eta_p^2 = 0.725$** | $F = 0.35, \eta_p^2 = 0.042$ |
| **B1 Position Invariance** | $F = 2.41, \eta_p^2 = 0.232$ | **$F = 7.82, \eta_p^2 = 0.494$** | $F = 1.30, \eta_p^2 = 0.140$ |
| **B2 Permutation Pairs** | $F = 0.00, \eta_p^2 = 0.001$ | **$F = 13.13, \eta_p^2 = 0.621$** | $F = 0.12, \eta_p^2 = 0.014$ |
| **B2 Counterfactual Pairs** | $F = 0.10, \eta_p^2 = 0.013$ | **$F = 15.13, \eta_p^2 = 0.654$** | $F = 1.17, \eta_p^2 = 0.128$ |
| **B3 Multi-Token Spans** | $F = 0.18, \eta_p^2 = 0.022$ | **$F = 17.70, \eta_p^2 = 0.689$** | $F = 1.74, \eta_p^2 = 0.179$ |
| **B4 Abstention Recall** | $F = 0.89, \eta_p^2 = 0.101$ | **$F = 18.47, \eta_p^2 = 0.698$** | $F = 0.00, \eta_p^2 = 0.000$ |

### Core Statistical Findings

1. **Massive Main Effect of Abstention Loss Multiplier ($\lambda$):**
   - Attenuating $\lambda$ from 1.00 to 0.25 caused large, statistically robust effects across all retrieval and abstention metrics ($\eta_p^2$ between 0.494 and 0.725):
     - Answerable retrieval more than doubled ($66.33 \to 136.50, F = 11.30$).
     - False abstentions fell drastically ($1003.17 \to 593.83, F = 11.42$).
     - However, unanswerable recall suffered severe degradation: CR-16C CAT9 fell from $86.5\% \to 46.7\%$ ($F = 21.09$), and B4 recall dropped from $280.00 \to 190.67$ ($F = 18.47$).
2. **Absence of Main Effect of Replay Fraction (25% vs 50%):**
   - Replay fraction showed virtually zero effect on sequence copying ($F = 0.00, \eta_p^2 = 0.000$). Across all 12 models, copying accuracy remained pinned at $0\% - 1\%$.
   - Replay ratio also had negligible influence on retrieval accuracy ($F = 0.15$), false abstention ($F = 0.14$), or abstention recall ($F = 0.56$).
3. **No Meaningful Interaction Effects:**
   - None of the interaction terms reached significance (all $\eta_p^2 \le 0.179$), demonstrating that increasing replay percentage does not rescue the negative gradient interference caused by $\lambda$.

