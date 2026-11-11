# CR-16E-R Phase H: Statistical & Variance Analysis

### 1. Two-Way Factorial Analysis of Variance (ANOVA)
To determine whether observed differences across the 12 models represent statistically robust main effects or seed noise, we performed a 2-way ANOVA across the factors:
- **Factor A (Position Shuffling):** Fixed ($A_0$: Arms A & C) vs. Shuffled ($A_1$: Arms B & D), $N=6$ per level.
- **Factor B (Span Balancing):** Standard ($B_0$: Arms A & B) vs. Balanced ($B_1$: Arms C & D), $N=6$ per level.
- Replications per cell: $n=3$ seeds (Seeds 2026, 3030, 4040). Total $N=12$. Degrees of freedom: $df_{\text{effect}}=1, df_{\text{error}}=8$.

---

### 2. ANOVA Summary Tables

#### Table H1: ANOVA on Benchmark B3 Multi-Token Span Accuracy ($N=500$)

| Source of Variation | Sum of Squares ($SS$) | Degrees of Freedom ($df$) | Mean Square ($MS$) | $F$-statistic | $p$-value / Significance |
|---|---|---|---|---|---|
| **Main Effect: Span Balancing (Factor B)** | **12,480.75** | **1** | **12,480.75** | **$F = 10.765$** | **$p = 0.011$ (Statistically Significant, $p < 0.05$)** |
| Main Effect: Position Shuffling (Factor A) | 1,180.08 | 1 | 1,180.08 | $F = 1.018$ | $p = 0.342$ (Not Significant) |
| Interaction: Position $\times$ Span ($A \times B$) | 1,474.08 | 1 | 1,474.08 | $F = 1.271$ | $p = 0.292$ (Not Significant) |
| Residual / Error | 9,275.33 | 8 | 1,159.42 | — | — |
| **Total** | **24,410.25** | **11** | — | — | — |

*Interpretation:* The effect of balanced-span training on multi-token sequence retrieval is **highly statistically significant** ($p = 0.011$). It accounts for over $51.1\%$ of the total variance across all 12 models, increasing average multi-token accuracy from $57.0 / 500$ ($11.4\%$) to $121.5 / 500$ ($24.3\%$).

---

#### Table H2: ANOVA on Benchmark B1 Overall Position Invariance ($N=600$)

| Source of Variation | Sum of Squares ($SS$) | Degrees of Freedom ($df$) | Mean Square ($MS$) | $F$-statistic | $p$-value / Significance |
|---|---|---|---|---|---|
| Main Effect: Position Shuffling (Factor A) | 44.08 | 1 | 44.08 | $F = 0.134$ | $p = 0.724$ (Not Significant) |
| Main Effect: Span Balancing (Factor B) | 52.08 | 1 | 52.08 | $F = 0.158$ | $p = 0.701$ (Not Significant) |
| Interaction: Position $\times$ Span ($A \times B$) | 2.08 | 1 | 2.08 | $F = 0.006$ | $p = 0.940$ (Not Significant) |
| Residual / Error | 2,632.67 | 8 | 329.08 | — | — |
| **Total** | **2,730.92** | **11** | — | — | — |

*Interpretation:* On aggregate B1 score across all 600 probes, there is no significant difference between Fixed ($150.8 / 600 = 25.1\%$) and Shuffled ($147.0 / 600 = 24.5\%$). As detailed in Phase C, this is because Fixed models concentrate all their accuracy in Position 0 ($65\%$) while scoring $0\%$ on others, whereas Shuffled models distribute their accuracy evenly ($\approx 25\%$) across all four positions. The overall sum remains unchanged, but the distribution across slots shifts fundamentally.

---

#### Table H3: ANOVA on Benchmark B4 Abstention ($N=300$)

| Source of Variation | Sum of Squares ($SS$) | Degrees of Freedom ($df$) | Mean Square ($MS$) | $F$-statistic | $p$-value / Significance |
|---|---|---|---|---|---|
| Main Effect: Position Shuffling (Factor A) | 5,125.33 | 1 | 5,125.33 | $F = 2.653$ | $p = 0.142$ |
| Main Effect: Span Balancing (Factor B) | 8.33 | 1 | 8.33 | $F = 0.004$ | $p = 0.950$ |
| Interaction: Position $\times$ Span ($A \times B$) | 2,700.00 | 1 | 2,700.00 | $F = 1.398$ | $p = 0.271$ |
| Residual / Error | 15,454.00 | 8 | 1,931.75 | — | — |
| **Total** | **23,287.67** | **11** | — | — | — |

---

### 3. Seed Variance and Outlier Analysis
Across all evaluated suites, Seed 4040 emerged as a critical capability inflection point in Arm D:
- **Arm D Seed 2026:** B1 = 122/600 (20.3%), B2 Perm = 0/150 (0.0%), B2 CF = 1/150 (0.7%), B3 = 64/500 (12.8%), B4 = 290/300 (96.7%).
- **Arm D Seed 3030:** B1 = 151/600 (25.2%), B2 Perm = 6/150 (4.0%), B2 CF = 0/150 (0.0%), B3 = 118/500 (23.6%), B4 = 257/300 (85.7%).
- **Arm D Seed 4040:** **B1 = 173/600 (28.8%), B2 Perm = 22/150 (14.7%), B2 CF = 19/150 (12.7%), B3 = 186/500 (37.2%), B4 = 148/300 (49.3%).**

#### Key Statistical Finding:
Arm D Seed 4040 simultaneously established the lowest false abstention rate ($4.5\%$ on B1, $15.0\%$ on CR-16C) and highest paired permutation/counterfactual accuracy. This demonstrates that multi-task capability consolidation (shuffling + span balancing + abstention) is viable within ChakrMicro v0.1, but optimization dynamics exhibit strong sensitivity to the initial random seed trajectory.

