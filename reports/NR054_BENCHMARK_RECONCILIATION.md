# NR-054: Benchmark Score Reconciliation & Discrepancy Resolution

**Mission:** CHAKRVIEW NR-054 — Benchmark Reconciliation, Checkpoint Lineage & Scientific Integrity Audit  
**Date:** November 27, 2026  
**Auditor:** ChakrGenesis Scientific Integrity Audit Team  
**Artifact Path:** [statistical_analysis_results.json](file:///D:/Project/ChakrView/experiments/forensics_nr054/statistical_analysis_results.json)

---

## 1. Executive Summary & Resolution of NR-053 Discrepancies

During NR-053, an inconsistency occurred between the reported benchmark means and the markdown distribution tables:

### Reported in NR-053 Reports:
| Candidate | Reported Mean | Erroneously Reported Distribution | Sum of Distribution | Implied Raw Score |
|---|---|---|---|---|
| RC3 Historical | 0.220 | [40, 7, 0, 3] | 50 | $7 \times 1 + 3 \times 3 = 16$ (Implies Mean = 0.320) |
| NR-051 17M | 0.180 | [41, 6, 0, 3] | 50 | $6 \times 1 + 3 \times 3 = 15$ (Implies Mean = 0.300) |
| NR-051 38M | 0.180 | [41, 6, 0, 3] | 50 | $6 \times 1 + 3 \times 3 = 15$ (Implies Mean = 0.300) |
| NR-052 17M | 0.180 | [42, 5, 1, 2] | 50 | $5 \times 1 + 1 \times 2 + 2 \times 3 = 13$ (Implies Mean = 0.260) |
| NR-052 38M | 0.240 | [39, 8, 1, 2] | 50 | $8 \times 1 + 1 \times 2 + 2 \times 3 = 16$ (Implies Mean = 0.320) |

### Root Cause of the Discrepancy
Forensic inspection of `experiments/forensics_nr053/fresh_eval_results_50.json` reveals that:
1. The **reported means (`0.220`, `0.180`, `0.180`, `0.180`, `0.240`) are mathematically correct** and correspond exactly to the raw prompt-level scores in `fresh_eval_results_50.json`.
2. The **markdown distribution table in NR-053 was pasted from an exploratory draft test run** that used a looser partial-matching regex rubric.
3. In the actual, authoritative evaluation file (`fresh_eval_results_50.json`), the true prompt-level distributions are:
   - **RC3:** `[45, 2, 0, 3]` (Raw Score: $2 \times 1 + 3 \times 3 = 11$, Mean: $11 / 50 = \mathbf{0.220}$)
   - **NR-051 17M:** `[47, 0, 0, 3]` (Raw Score: $3 \times 3 = 9$, Mean: $9 / 50 = \mathbf{0.180}$)
   - **NR-051 38M:** `[47, 0, 0, 3]` (Raw Score: $3 \times 3 = 9$, Mean: $9 / 50 = \mathbf{0.180}$)
   - **NR-052 17M:** `[47, 0, 0, 3]` (Raw Score: $3 \times 3 = 9$, Mean: $9 / 50 = \mathbf{0.180}$)
   - **NR-052 38M:** `[46, 0, 0, 4]` (Raw Score: $4 \times 3 = 12$, Mean: $12 / 50 = \mathbf{0.240}$)

Every distribution sums exactly to **50 evaluated prompts**.

---

## 2. Reconciled Benchmark Performance Table (Fresh 50 Prompts)

### Sampling ($T=0.3$, $\text{top\_p}=0.9$, $\text{rep\_pen}=1.15$)
| Candidate System | N | Total Raw Score | Reconciled Distribution [0,1,2,3] | Raw Mean (/3.0) | Normalized Mean (/1.0) | Capability % | 95% Bootstrap CI |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **RC3 Baseline** | 50 | 11 | [45, 2, 0, 3] | **0.220** | 0.0733 | 7.33% | [0.040, 0.440] |
| **NR-051 17M SFT**| 50 | 9 | [47, 0, 0, 3] | **0.180** | 0.0600 | 6.00% | [0.000, 0.420] |
| **NR-051 38M SFT**| 50 | 9 | [47, 0, 0, 3] | **0.180** | 0.0600 | 6.00% | [0.000, 0.420] |
| **NR-052 17M SFT**| 50 | 9 | [47, 0, 0, 3] | **0.180** | 0.0600 | 6.00% | [0.000, 0.420] |
| **NR-052 38M SFT**| 50 | 12 | [46, 0, 0, 4] | **0.240** | 0.0800 | 8.00% | [0.060, 0.480] |

### Greedy ($T=0.0$)
| Candidate System | N | Total Raw Score | Reconciled Distribution [0,1,2,3] | Raw Mean (/3.0) | Normalized Mean (/1.0) | Capability % | 95% Bootstrap CI |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **RC3 Baseline** | 50 | 10 | [46, 1, 0, 3] | **0.200** | 0.0667 | 6.67% | [0.020, 0.420] |
| **NR-051 17M SFT**| 50 | 9 | [47, 0, 0, 3] | **0.180** | 0.0600 | 6.00% | [0.000, 0.420] |
| **NR-051 38M SFT**| 50 | 15 | [45, 0, 0, 5] | **0.300** | 0.1000 | 10.00%| [0.060, 0.540] |
| **NR-052 17M SFT**| 50 | 3 | [49, 0, 0, 1] | **0.060** | 0.0200 | 2.00% | [0.000, 0.180] |
| **NR-052 38M SFT**| 50 | 9 | [47, 0, 0, 3] | **0.180** | 0.0600 | 6.00% | [0.000, 0.420] |
