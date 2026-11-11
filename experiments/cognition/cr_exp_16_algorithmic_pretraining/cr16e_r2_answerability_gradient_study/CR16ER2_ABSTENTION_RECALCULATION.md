# CR-16E-R2 Phase C: Recalculated Abstention & Risk-Coverage Metrics

### 1. Corrected Metric Definitions & Formulations

With the verified labels ($N_{\text{ans}} = 2,000$ across CR-16C CAT1–8,10 and B1; $N_{\text{unans}} = 400$ across CR-16C CAT9 and B4):
- **Answerable Coverage:** $\frac{\text{Answerable Probes Answered (not [UNKNOWN])}}{N_{\text{ans}} = 2,000}$.
- **False Abstention Rate:** $\frac{\text{Answerable Probes emitted as [UNKNOWN]}}{N_{\text{ans}} = 2,000}$.
- **Unanswerable Recall (Abstention Recall):** $\frac{\text{Unanswerable Probes correctly emitted as [UNKNOWN]}}{N_{\text{unans}} = 400}$.
- **Selective Accuracy on Answered Probes:** $\frac{\text{Correct Answers emitted}}{\text{Total Answerable Probes Answered}}$.
- **False Answer Rate on Unanswerable:** $\frac{\text{Unanswerable Probes where model hallucinated an answer}}{N_{\text{unans}} = 400}$.

---

### 2. Recalculated Grand Matrix ($N_{\text{ans}} = 2,000$, $N_{\text{unans}} = 400$)

| Model Arm & Seed | Answerable Coverage ($N=2000$) | False Abstention Rate ($N=2000$) | Selective Accuracy on Answered | Unanswerable Recall Total ($N=400$) | B4 Recall ($N=300$) | CR16C CAT9 Recall ($N=100$) | False Answer Rate ($N=400$) |
|---|---|---|---|---|---|---|---|
| **Arm A (Fixed, Std)** | | | | | | | |
| ├── Seed 2026 | 1,025 / 2,000 (51.2%) | 975 / 2,000 (48.8%) | 235 / 1,025 (22.9%) | 269 / 400 (67.2%) | 213 / 300 (71.0%) | 56 / 100 (56.0%) | 131 / 400 (32.8%) |
| ├── Seed 3030 | 1,293 / 2,000 (64.6%) | 707 / 2,000 (35.4%) | 346 / 1,293 (26.8%) | 197 / 400 (49.2%) | 150 / 300 (50.0%) | 47 / 100 (47.0%) | 203 / 400 (50.8%) |
| └── Seed 4040 | 856 / 2,000 (42.8%) | 1,144 / 2,000 (57.2%) | 295 / 856 (34.5%) | 260 / 400 (65.0%) | 203 / 300 (67.7%) | 57 / 100 (57.0%) | 140 / 400 (35.0%) |
| **Arm B (Shuffled, Std)** | | | | | | | |
| ├── Seed 2026 | 809 / 2,000 (40.5%) | 1,191 / 2,000 (59.6%) | 198 / 809 (24.5%) | 363 / 400 (90.8%) | 274 / 300 (91.3%) | 89 / 100 (89.0%) | 37 / 400 (9.2%) |
| ├── Seed 3030 | 804 / 2,000 (40.2%) | 1,196 / 2,000 (59.8%) | 208 / 804 (25.9%) | 376 / 400 (94.0%) | 282 / 300 (94.0%) | 94 / 100 (94.0%) | 24 / 400 (6.0%) |
| └── Seed 4040 | 1,150 / 2,000 (57.5%) | 850 / 2,000 (42.5%) | 280 / 1,150 (24.3%) | 293 / 400 (73.2%) | 224 / 300 (74.7%) | 69 / 100 (69.0%) | 107 / 400 (26.8%) |
| **Arm C (Fixed, Bal)** | | | | | | | |
| ├── Seed 2026 | 959 / 2,000 (47.9%) | 1,041 / 2,000 (52.0%) | 277 / 959 (28.9%) | 292 / 400 (73.0%) | 216 / 300 (72.0%) | 76 / 100 (76.0%) | 108 / 400 (27.0%) |
| ├── Seed 3030 | 786 / 2,000 (39.3%) | 1,214 / 2,000 (60.7%) | 241 / 786 (30.7%) | 307 / 400 (76.8%) | 230 / 300 (76.7%) | 77 / 100 (77.0%) | 93 / 400 (23.2%) |
| └── Seed 4040 | 1,001 / 2,000 (50.0%) | 999 / 2,000 (50.0%) | 287 / 1,001 (28.7%) | 275 / 400 (68.8%) | 215 / 300 (71.7%) | 60 / 100 (60.0%) | 125 / 400 (31.2%) |
| **Arm D (Shuffled, Bal)** | | | | | | | |
| ├── Seed 2026 | 771 / 2,000 (38.6%) | 1,229 / 2,000 (61.5%) | 162 / 771 (21.0%) | **383 / 400 (95.8%)** | 290 / 300 (96.7%) | 93 / 100 (93.0%) | 17 / 400 (4.2%) |
| ├── Seed 3030 | 1,050 / 2,000 (52.5%) | 950 / 2,000 (47.5%) | 226 / 1,050 (21.5%) | **343 / 400 (85.8%)** | 257 / 300 (85.7%) | 86 / 100 (86.0%) | 57 / 400 (14.2%) |
| └── Seed 4040 | **1,777 / 2,000 (88.8%)**| **223 / 2,000 (11.2%)**| **350 / 1,777 (19.7%)**| **177 / 400 (44.2%)** | 148 / 300 (49.3%) | 29 / 100 (29.0%) | 223 / 400 (55.8%) |

---

### 3. Critical Scientific Insights: Risk vs Coverage

1. **The Over-Abstention Regime (Arm B s2026, Arm B s3030, Arm D s2026):**
   - These models achieved extraordinary abstention recall ($90.8\% - 95.8\%$), almost never hallucinating on unanswerable questions (false answer rate $4.2\% - 9.2\%$).
   - However, their answerable coverage was severely suppressed ($38.6\% - 40.5\%$), falsely abstaining on $\approx 60\%$ of answerable questions ($1,191 - 1,229$ false abstentions out of 2,000). They treat `[UNKNOWN]` as an aggressive generic sink whenever prompt syntax deviates from training.
2. **The High-Coverage Regime (Arm D Seed 4040):**
   - Arm D Seed 4040 achieved **88.8% answerable coverage** ($1,777 / 2,000$), maintaining active answering engagement and restricting false abstention to only **11.2%**.
   - On the other hand, its unanswerable recall dropped to **44.2% (177 / 400)**, resulting in a false answer rate of $55.8\%$ on missing-key probes.
3. **Conclusion on Calibration:**
   - No single model yet occupies the optimal Pareto frontier (e.g. $>80\%$ coverage *and* $>80\%$ recall).
   - Rather than claiming "calibrated abstention is solved", the evidence documents a classical **precision-recall / risk-coverage trade-off**: models with high abstention recall pay a steep price in rejected answerable queries.

