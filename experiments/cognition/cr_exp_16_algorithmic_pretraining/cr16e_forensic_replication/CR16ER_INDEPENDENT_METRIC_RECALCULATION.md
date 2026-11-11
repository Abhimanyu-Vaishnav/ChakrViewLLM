# CR-16E-R Phase B: Independent Metric Recalculation

### 1. Independent Recalculation Overview
Every primary metric reported in CR-EXP-16E was independently recalculated from raw prediction logs (`eval_outputs/*_forensic_eval.json`) across all 12 factorial models ($N=3$ seeds across Arms A, B, C, D).

**Recalculation Rules Applied:**
- Exact match: Strict string equality between generated output and benchmark target (`pred == target`).
- Percentages computed strictly from raw integer numerators and denominators: $\frac{\text{correct}}{\text{total}}$.
- Sample standard deviation ($s = \sqrt{\frac{1}{N-1}\sum (x_i - \bar{x})^2}$) applied across the 3 seeds ($N=3$).

---

### 2. Grand Scorecard Verification Matrix

#### Table B1: Benchmark Suite Scorecard (Numerators, Denominators, Means $\pm s$)

| Model Arm & Seed | B1 Pos Invariance (600) | B2 Perm Pairs (150) | B2 CF Pairs (150) | B3 Spans Total (500) | B3 Len 3 Spans (100) | B3 Len 4 Spans (100) | B4 Abstain (300) | CR-16 Copy (100) | CR-16C Retr (1,500) |
|---|---|---|---|---|---|---|---|---|---|
| **Arm A (Fixed, Std)** |
| ├── Seed 2026 | 125 / 600 (20.8%) | 0 / 150 (0.0%) | 21 / 150 (14.0%) | 64 / 500 (12.8%) | 0 / 100 (0.0%) | 0 / 100 (0.0%) | 213 / 300 (71.0%) | 20 / 100 (20.0%) | 166 / 1500 (11.1%) |
| ├── Seed 3030 | 169 / 600 (28.2%) | 0 / 150 (0.0%) | 54 / 150 (36.0%) | 89 / 500 (17.8%) | 0 / 100 (0.0%) | 0 / 100 (0.0%) | 150 / 300 (50.0%) | 3 / 100 (3.0%) | 224 / 1500 (14.9%) |
| └── Seed 4040 | 151 / 600 (25.2%) | 0 / 150 (0.0%) | 35 / 150 (23.3%) | 81 / 500 (16.2%) | 0 / 100 (0.0%) | 0 / 100 (0.0%) | 203 / 300 (67.7%) | 2 / 100 (2.0%) | 201 / 1500 (13.4%) |
| **Arm A Mean $\pm s$** | **148.33 $\pm 22.14$ (24.7%)** | **0.00 $\pm 0.00$ (0.0%)** | **36.67 $\pm 16.50$ (24.4%)** | **78.00 $\pm 12.77$ (15.6%)** | **0.00 $\pm 0.00$ (0.0%)** | **0.00 $\pm 0.00$ (0.0%)** | **188.67 $\pm 33.86$ (62.9%)** | **8.33 $\pm 10.12$ (8.3%)** | **197.00 $\pm 29.21$ (13.1%)** |
| **Arm B (Shuffled, Std)** |
| ├── Seed 2026 | 139 / 600 (23.2%) | 1 / 150 (0.7%) | 1 / 150 (0.7%) | 37 / 500 (7.4%) | 0 / 100 (0.0%) | 0 / 100 (0.0%) | 274 / 300 (91.3%) | 1 / 100 (1.0%) | 148 / 1500 (9.9%) |
| ├── Seed 3030 | 141 / 600 (23.5%) | 1 / 150 (0.7%) | 0 / 150 (0.0%) | 20 / 500 (4.0%) | 0 / 100 (0.0%) | 0 / 100 (0.0%) | 282 / 300 (94.0%) | 6 / 100 (6.0%) | 161 / 1500 (10.7%) |
| └── Seed 4040 | 156 / 600 (26.0%) | 10 / 150 (6.7%) | 5 / 150 (3.3%) | 51 / 500 (10.2%) | 0 / 100 (0.0%) | 0 / 100 (0.0%) | 224 / 300 (74.7%) | 0 / 100 (0.0%) | 193 / 1500 (12.9%) |
| **Arm B Mean $\pm s$** | **145.33 $\pm 9.29$ (24.2%)** | **4.00 $\pm 5.20$ (2.7%)** | **2.00 $\pm 2.65$ (1.3%)** | **36.00 $\pm 15.52$ (7.2%)** | **0.00 $\pm 0.00$ (0.0%)** | **0.00 $\pm 0.00$ (0.0%)** | **260.00 $\pm 31.43$ (86.7%)** | **2.33 $\pm 3.21$ (2.3%)** | **167.33 $\pm 23.16$ (11.2%)** |
| **Arm C (Fixed, Bal)** |
| ├── Seed 2026 | 156 / 600 (26.0%) | 0 / 150 (0.0%) | 12 / 150 (8.0%) | 109 / 500 (21.8%) | 33 / 100 (33.0%) | 43 / 100 (43.0%) | 216 / 300 (72.0%) | 0 / 100 (0.0%) | 197 / 1500 (13.1%) |
| ├── Seed 3030 | 143 / 600 (23.8%) | 0 / 150 (0.0%) | 19 / 150 (12.7%) | 106 / 500 (21.2%) | 36 / 100 (36.0%) | 28 / 100 (28.0%) | 230 / 300 (76.7%) | 1 / 100 (1.0%) | 175 / 1500 (11.7%) |
| └── Seed 4040 | 161 / 600 (26.8%) | 0 / 150 (0.0%) | 24 / 150 (16.0%) | 146 / 500 (29.2%) | 40 / 100 (40.0%) | 47 / 100 (47.0%) | 215 / 300 (71.7%) | 1 / 100 (1.0%) | 186 / 1500 (12.4%) |
| **Arm C Mean $\pm s$** | **153.33 $\pm 9.29$ (25.6%)** | **0.00 $\pm 0.00$ (0.0%)** | **18.33 $\pm 6.03$ (12.2%)** | **120.33 $\pm 22.28$ (24.1%)**| **36.33 $\pm 3.51$ (36.3%)** | **39.33 $\pm 10.02$ (39.3%)**| **220.33 $\pm 8.39$ (73.4%)** | **0.67 $\pm 0.58$ (0.7%)** | **186.00 $\pm 11.00$ (12.4%)** |
| **Arm D (Shuffled, Bal)**|
| ├── Seed 2026 | 122 / 600 (20.3%) | 0 / 150 (0.0%) | 1 / 150 (0.7%) | 64 / 500 (12.8%) | 21 / 100 (21.0%) | 32 / 100 (32.0%) | 290 / 300 (96.7%) | 0 / 100 (0.0%) | 133 / 1500 (8.9%) |
| ├── Seed 3030 | 151 / 600 (25.2%) | 6 / 150 (4.0%) | 0 / 150 (0.0%) | 118 / 500 (23.6%) | 45 / 100 (45.0%) | 34 / 100 (34.0%) | 257 / 300 (85.7%) | 1 / 100 (1.0%) | 161 / 1500 (10.7%) |
| └── Seed 4040 | 173 / 600 (28.8%) | 22 / 150 (14.7%) | 19 / 150 (12.7%) | 186 / 500 (37.2%) | 49 / 100 (49.0%) | 57 / 100 (57.0%) | 148 / 300 (49.3%) | 0 / 100 (0.0%) | 206 / 1500 (13.7%) |
| **Arm D Mean $\pm s$** | **148.67 $\pm 25.60$ (24.8%)** | **9.33 $\pm 11.37$ (6.2%)** | **6.67 $\pm 10.69$ (4.5%)** | **122.67 $\pm 61.16$ (24.5%)**| **38.33 $\pm 15.14$ (38.3%)** | **41.00 $\pm 13.89$ (41.0%)**| **231.67 $\pm 74.55$ (77.2%)** | **0.33 $\pm 0.58$ (0.3%)** | **166.67 $\pm 36.86$ (11.1%)** |

---

### 3. Verification of Report Discrepancies
- **Discrepancy Check:** The recomputed per-seed metrics match the CR-16E report bit-for-bit.
- **Sample vs Population Standard Deviation:** The CR-16E report used population standard deviation ($N$ in denominator), whereas Table B1 above explicitly reports sample standard deviation ($N-1$ in denominator, e.g. $s = 22.14$ vs $\sigma = 18.2$ for Arm A B1). Both conventions confirm the underlying distribution.
- **Empty / Malformed Outputs:** Zero outputs produced crashes, unhandled tokens, or malformed JSON artifacts. All predictions terminated cleanly within the 20-token inference budget.

