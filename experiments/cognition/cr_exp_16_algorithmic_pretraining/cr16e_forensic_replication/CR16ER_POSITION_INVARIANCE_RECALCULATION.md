# CR-16E-R Phase C: Position-Invariance Investigation

### 1. Research Question: Redistribution vs Genuine Invariance
The central inquiry of Phase C is whether position shuffling in CR-16E genuinely established position-invariant retrieval or merely redistributed a weak ~25% accuracy budget across slots without creating an invariant associative lookup.

---

### 2. Forensic Breakdown by Fact Position & Query Structure (Benchmark B1, 600 Probes)

| Factorial Model Arm | Pos 0 Target (227 probes) | Pos 1 Target (192 probes) | Pos 2 Target (131 probes) | Pos 3 Target (50 probes) | Query-Before Facts (135 probes) | Query-After Facts (465 probes) |
|---|---|---|---|---|---|---|
| **Arm A (Fixed Pos, s2026)** | 125 / 227 (55.1%) | 0 / 192 (0.0%) | 0 / 131 (0.0%) | 0 / 50 (0.0%) | 12 / 135 (8.9%) | 113 / 465 (24.3%) |
| **Arm A (Fixed Pos, s3030)** | 168 / 227 (74.0%) | 0 / 192 (0.0%) | 1 / 131 (0.8%) | 0 / 50 (0.0%) | 22 / 135 (16.3%) | 147 / 465 (31.6%) |
| **Arm A (Fixed Pos, s4040)** | 151 / 227 (66.5%) | 0 / 192 (0.0%) | 0 / 131 (0.0%) | 0 / 50 (0.0%) | 4 / 135 (3.0%) | 147 / 465 (31.6%) |
| **Arm A Mean** | **148.0 / 227 (65.2%)** | **0.0 / 192 (0.0%)** | **0.3 / 131 (0.2%)** | **0.0 / 50 (0.0%)** | **12.7 / 135 (9.4%)** | **135.7 / 465 (29.2%)** |
| **Arm B (Shuffled, s2026)** | 61 / 227 (26.9%) | 50 / 192 (26.0%) | 16 / 131 (12.2%) | 12 / 50 (24.0%) | 36 / 135 (26.7%) | 103 / 465 (22.2%) |
| **Arm B (Shuffled, s3030)** | 40 / 227 (17.6%) | 52 / 192 (27.1%) | 30 / 131 (22.9%) | 19 / 50 (38.0%) | 39 / 135 (28.9%) | 102 / 465 (21.9%) |
| **Arm B (Shuffled, s4040)** | 62 / 227 (27.3%) | 45 / 192 (23.4%) | 40 / 131 (30.5%) | 9 / 50 (18.0%) | 37 / 135 (27.4%) | 119 / 465 (25.6%) |
| **Arm B Mean** | **54.3 / 227 (23.9%)** | **49.0 / 192 (25.5%)** | **28.7 / 131 (21.9%)** | **13.3 / 50 (26.7%)** | **37.3 / 135 (27.6%)** | **108.0 / 465 (23.2%)** |
| **Arm C (Fixed Pos, s2026)** | 156 / 227 (68.7%) | 0 / 192 (0.0%) | 0 / 131 (0.0%) | 0 / 50 (0.0%) | 20 / 135 (14.8%) | 136 / 465 (29.2%) |
| **Arm C (Fixed Pos, s3030)** | 143 / 227 (63.0%) | 0 / 192 (0.0%) | 0 / 131 (0.0%) | 0 / 50 (0.0%) | 7 / 135 (5.2%) | 136 / 465 (29.2%) |
| **Arm C (Fixed Pos, s4040)** | 161 / 227 (70.9%) | 0 / 192 (0.0%) | 0 / 131 (0.0%) | 0 / 50 (0.0%) | 15 / 135 (11.1%) | 146 / 465 (31.4%) |
| **Arm C Mean** | **153.3 / 227 (67.5%)** | **0.0 / 192 (0.0%)** | **0.0 / 131 (0.0%)** | **0.0 / 50 (0.0%)** | **14.0 / 135 (10.4%)** | **139.3 / 465 (30.0%)** |
| **Arm D (Shuffled, s2026)** | 48 / 227 (21.1%) | 38 / 192 (19.8%) | 25 / 131 (19.1%) | 11 / 50 (22.0%) | 31 / 135 (23.0%) | 91 / 465 (19.6%) |
| **Arm D (Shuffled, s3030)** | 51 / 227 (22.5%) | 61 / 192 (31.8%) | 31 / 131 (23.7%) | 8 / 50 (16.0%) | 35 / 135 (25.9%) | 116 / 465 (24.9%) |
| **Arm D (Shuffled, s4040)** | 62 / 227 (27.3%) | 64 / 192 (33.3%) | 38 / 131 (29.0%) | 9 / 50 (18.0%) | 46 / 135 (34.1%) | 127 / 465 (27.3%) |
| **Arm D Mean** | **53.7 / 227 (23.6%)** | **54.3 / 192 (28.3%)** | **31.3 / 131 (23.9%)** | **9.3 / 50 (18.7%)** | **37.3 / 135 (27.6%)** | **111.3 / 465 (23.9%)** |

---

### 3. Forensic Interpretation & Quantitative Diagnosis
1. **Mathematical Confirmation of Redistribution:**
   - In fixed-position models (Arms A and C), overall accuracy is $\approx 25\%$ solely because Position 0 represents $37.8\%$ of the test set ($227/600$) and scores $\approx 66\%$, while all other positions score $0.0\%$. Total accuracy: $0.378 \times 66\% \approx 25.0\%$.
   - In shuffled models (Arms B and D), overall accuracy is also $\approx 25\%$ ($145.3$ and $148.7$ / 600). However, the accuracy is **uniformly spread across all four positions**:
     - Position 0: $23.6\% - 23.9\%$
     - Position 1: $25.5\% - 28.3\%$
     - Position 2: $21.9\% - 23.9\%$
     - Position 3: $18.7\% - 26.7\%$
2. **True Invariance vs Uniform Weakness:**
   - Position-shuffling *does* successfully eliminate the absolute blind spot at positions 1, 2, and 3. The attention heads no longer attend strictly to slot 0.
   - However, the model has **not** acquired robust position invariance ($\ge 70\%$, Gate 1). Instead, the model exhibits a diffuse, uncalibrated attention distribution that settles at $\approx 25\%$ accuracy regardless of where the target fact is placed.
3. **Query Bidirectionality:**
   - Query-before-facts accuracy improved from $9.4\%$ in fixed models to $27.6\%$ in shuffled models, demonstrating that shuffling allows the attention heads to attend backwards across context tokens.

