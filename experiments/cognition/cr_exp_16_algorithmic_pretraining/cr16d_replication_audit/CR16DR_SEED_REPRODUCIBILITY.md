# CR-16DR Seed Reproducibility & Multi-Seed Statistical Analysis

### 1. 5-Seed Replication Performance Matrix (N=5 Seeds per Arm)
Evaluating across Seeds 42, 101, 202, 303, 404:

#### Sequence Copying Exact Match (100 Frozen Probes)
- **Arm A (0% Replay):** `[39, 43, 38, 54, 39]` $\implies$ **Mean = 42.60%** $\pm 5.95$ (Range: 38% – 54%)
- **Arm B (25% Replay):** `[81, 80, 83, 73, 87]` $\implies$ **Mean = 80.80%** $\pm 4.58$ (Range: 73% – 87%)
- **Arm C (50% Replay):** `[83, 84, 88, 78, 84]` $\implies$ **Mean = 83.40%** $\pm 3.20$ (Range: 78% – 88%)
- **Arm D (75% Replay):** `[91, 89, 87, 87, 87]` $\implies$ **Mean = 88.20%** $\pm 1.60$ (Range: 87% – 91%)

#### Overall Retrieval (1,500 Frozen Probes)
- **Arm A (0% Replay):** `[818, 664, 663, 659, 694]` $\implies$ **Mean = 699.60 (46.6%)** $\pm 60.51$
- **Arm B (25% Replay):** `[797, 807, 821, 799, 819]` $\implies$ **Mean = 808.60 (53.9%)** $\pm 9.91$
- **Arm C (50% Replay):** `[763, 839, 806, 775, 805]` $\implies$ **Mean = 797.60 (53.2%)** $\pm 26.64$
- **Arm D (75% Replay):** `[773, 741, 735, 748, 777]` $\implies$ **Mean = 754.80 (50.3%)** $\pm 17.05$

#### Dynamic Value Reassignment (CAT2, 200 Probes)
- **Arm A (0% Replay):** `[197, 168, 158, 158, 161]` $\implies$ **Mean = 168.40 (84.2%)** $\pm 14.76$
- **Arm B (25% Replay):** `[200, 200, 200, 200, 200]` $\implies$ **Mean = 200.00 (100.0%)** $\pm 0.00$
- **Arm C (50% Replay):** `[200, 200, 200, 200, 200]` $\implies$ **Mean = 200.00 (100.0%)** $\pm 0.00$
- **Arm D (75% Replay):** `[182, 187, 192, 195, 193]` $\implies$ **Mean = 189.80 (94.9%)** $\pm 4.71$

#### Compositional Novel Combinations (G1, 250 Probes)
- **Arm A (0% Replay):** `[190, 146, 193, 162, 193]` $\implies$ **Mean = 176.80 (70.7%)** $\pm 19.32$
- **Arm B (25% Replay):** `[41, 99, 155, 80, 124]` $\implies$ **Mean = 99.80 (39.9%)** $\pm 38.69$
- **Arm C (50% Replay):** `[122, 241, 130, 112, 63]` $\implies$ **Mean = 133.60 (53.4%)** $\pm 58.54$
- **Arm D (75% Replay):** `[196, 110, 158, 142, 181]` $\implies$ **Mean = 157.40 (63.0%)** $\pm 30.10$

---

### 2. Analysis of Reproducibility
1. **Rock-Solid Reproducibility for Reassignment:**
   - In both Arm B and Arm C, dynamic value reassignment scored **200/200 (100.0%) across all 5 independent seeds**. There was zero seed variance ($\sigma = 0.0$).
2. **Rock-Solid Reproducibility for Copying Retention:**
   - In Arms B, C, and D, copying retention replicated consistently with standard deviations under 5 percentage points. Replay $\ge 25\%$ reliably prevents catastrophic forgetting in 100% of runs.
3. **High Variance in Compositional Combinations (G1):**
   - In Arm C, G1 scores varied from 25.2% to 96.4% ($\sigma = 58.5$).
   - In Arm D, G1 scores were substantially more consistent, averaging **63.0%** with a smaller standard deviation ($\sigma = 30.1$).

