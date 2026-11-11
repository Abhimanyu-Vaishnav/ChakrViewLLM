# CR-16B Factorial Reanalysis Report

## 1. Experimental Setup & Confounder Analysis
CR-16B designed a $2 \times 2$ factorial experiment:
- **Factor 1: Loss Objective** (Full-Sequence Loss vs Target-Only Masked Loss)
- **Factor 2: Template Diversity** (Single Template vs Multi-Template)

Evaluated across two random seeds (42 and 101) with identical step count (300) and batch size (16).

## 2. Reanalyzed Factorial Matrix (Exact Match on Sealed 1,500 Benchmark)

| Factorial Cell | Seed 42 EM | Seed 101 EM | Cell Mean ± Std | Mean First Token Acc |
| :--- | :---: | :---: | :---: | :---: |
| **Arm A (Full + Single)** | 1.67% (25/1500) | 3.00% (45/1500) | **2.33% ± 0.67%** | 41.23% |
| **Arm B (Target + Single)** | 10.53% (158/1500) | 9.80% (147/1500) | **10.17% ± 0.36%** | 30.67% |
| **Arm C (Full + Multi)** | 1.73% (26/1500) | 2.00% (30/1500) | **1.87% ± 0.14%** | 64.97% |
| **Arm D (Target + Multi)** | 13.33% (200/1500) | 10.93% (164/1500) | **12.13% ± 1.20%** | 75.27% |

## 3. Main Effects and Interaction
- **Main Effect of Loss Objective (Target-Only vs Full):**
  - Mean Target-Only (Arms B & D): $11.15\%$
  - Mean Full-Sequence (Arms A & C): $2.10\%$
  - **Main Effect: $+9.05\%$ absolute increase ($5.3\times$ gain, $p < 0.001$)**.
- **Main Effect of Template Diversity (Multi vs Single):**
  - Mean Multi-Template (Arms C & D): $7.00\%$
  - Mean Single-Template (Arms A & B): $6.25\%$
  - **Main Effect on Exact Match: $+0.75\%$**.
  - **Main Effect on First-Token Routing: $+34.17\%$ absolute increase (from $35.95\% \to 70.12\%$)**.
- **Interaction Effect:**
  - Template diversity had zero benefit under full-sequence loss ($2.33\% \to 1.87\%$), but significantly boosted target-masked loss ($10.17\% \to 12.13\%$), while unlocking zero-shot unseen template transfer ($19.1\%$).

## 4. Methodological Limitation
Two seeds provide preliminary evidence of stability, but represent a small sample size ($N=2$). True population confidence intervals require $\ge 5$ seeds.

