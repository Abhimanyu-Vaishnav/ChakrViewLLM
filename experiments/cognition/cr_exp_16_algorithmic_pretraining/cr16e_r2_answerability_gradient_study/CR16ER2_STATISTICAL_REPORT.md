# CR-16E-R2 Phase G: Statistical & Variance Report

### 1. Methodology & Unit of Replication
- **Replication Unit:** Complete independent training runs ($N=3$ seeds per experimental arm: Seeds 2026, 3030, 4040).
- We do **not** inflate degrees of freedom by treating thousands of deterministic evaluation probes as independent observations.
- All metrics are reported with exact sample means, sample standard deviations ($s$ with $N-1$ degrees of freedom), paired seed-level differences ($\Delta = G2 - G1$), paired $t$-statistics ($df=2$), and standardized effect sizes (Cohen's $d$).

---

### 2. Paired Statistical Comparison: Arm G1 vs Arm G2

#### Table G1: Paired Statistics & Effect Sizes ($N=3$ Pairs)

| Evaluation Benchmark | Arm G1 Mean $\pm s$ | Arm G2 Mean $\pm s$ | Paired Difference ($\Delta = G2 - G1$) | Paired $t$-statistic ($df=2$) | Cohen's $d$ (Effect Size) | Scientific Interpretation |
|---|---|---|---|---|---|---|
| **CR-16 Copying Exact (100)** | **0.33 $\pm 0.58$** | **1.00 $\pm 1.00$** | **+0.67 $\pm 0.58$** | $t = 2.000$ | $d = 1.15$ | **No meaningful recovery.** Remains severely collapsed relative to $66.0\%$ parent. |
| **CR-16C Answerable (1,400)** | 101.67 $\pm 59.20$ (7.3%) | **208.33 $\pm 17.79$ (14.9%)** | **+106.67 $\pm 49.80$** | **$t = 3.710$** | **$d = 2.14$** | **Large positive effect.** Masking negative loss doubled answerable retrieval accuracy. |
| **CR-16C False Abstain (1,400)** | 699.00 $\pm 439.42$ (49.9%)| **0.00 $\pm 0.00$ (0.0%)** | **-699.00 $\pm 439.42$** | **$t = -2.755$** | **$d = -1.59$** | **Large effect.** False abstentions completely eliminated under loss masking. |
| **B1 Position Score (600)** | 143.00 $\pm 21.28$ (23.8%)| **163.67 $\pm 9.29$ (27.3%)** | **+20.67 $\pm 14.19$** | $t = 2.522$ | $d = 1.46$ | Moderate positive effect on positional retrieval. |
| **B3 Multi-Token Spans (500)** | 136.00 $\pm 43.86$ (27.2%)| **203.67 $\pm 16.04$ (40.7%)** | **+67.67 $\pm 36.46$** | **$t = 3.215$** | **$d = 1.86$** | **Large positive effect.** Multi-token retrieval accuracy increased substantially. |
| **B4 Abstention Recall (300)** | 227.00 $\pm 71.04$ (75.7%)| **0.00 $\pm 0.00$ (0.0%)** | **-227.00 $\pm 71.04$** | **$t = -5.534$** | **$d = -3.20$** | Complete loss of abstention recall when loss is masked. |

---

### 3. Key Statistical Takeaways
1. **Hypothesis Refutation:** The hypothesis that negative abstention loss caused the sequence copying collapse is statistically refuted. The difference in copying is negligible ($+0.67$ probes out of 100), leaving the model far below the $66.0\%$ benchmark.
2. **The Real Cost of Negative Loss:** The true causal impact of negative abstention loss was a **$2\times$ reduction in answerable retrieval accuracy** ($101.7 \to 208.3$, Cohen's $d = 2.14$) and massive false abstention on answerable contexts ($49.9\%$).
3. **Seed Variance in Arm G1:** Arm G1 exhibited high seed variance ($s = 439.4$ on false abstention), driven by Seed 4040's unique resistance to over-abstention (only $192$ false abstentions vs $970$ in Seed 2026). In contrast, Arm G2 eliminated this variance entirely ($s = 0.0$ false abstentions across all seeds).

