# NR-016 Phase 4 — Fair 17M vs 38M Capacity Comparison Report
**Mission ID:** CV-GENESIS-NR-016  
**Date:** 2026-11-25  
**Experiment Design:** Matched training from scratch on synthetic tasks across independent seeds (Seed 42 and Seed 101)  
**Evaluator:** `data/scientific_eval_016/scientific_benchmark_320.json` (320 tasks: 160 in-distribution, 160 strict OOD)  
**Results Artifact:** `data/scientific_eval_016/capacity_comparison_results.json`

---

## 1. Executive Summary & Core Scientific Finding
A strictly controlled capacity experiment was executed comparing **ChakrGenesis-17M** ($17,063,808$ parameters) against **ChakrGenesis-38M** ($38,285,824$ parameters, $+124.4\%$ parameters, $+2$ layers, $+33\%$ width, $+50\%$ FFN width).

Both architectures were trained from random initialization on identical token sequences, with identical batch sizes ($16$), identical AdamW optimizer hyperparameters, and identical learning rate schedules across two independent seeds.

### **The Decisive Scientific Verdict:**
**ARCHITECTURE ADVANTAGE NOT DEMONSTRATED FOR OUT-OF-DISTRIBUTION GENERALIZATION**
- While 38M achieved lower final training loss ($0.9766 - 1.0009$ vs $1.0742 - 1.1625$ for 17M), **out-of-distribution (OOD) accuracy remained essentially zero across both architectures ($0.0\% - 0.6\%$)**.
- Averaged across seeds, 17M achieved **$8.6\%$** overall strict accuracy, while 38M achieved **$8.1\%$**.
- Scaling parameters from 17M to 38M without changing the reasoning objective or scratchpad supervision **did not induce symbolic generalization**.

---

## 2. Quantitative Performance Table

| Model Run | Seed | Total Params | Final Train Loss | In-Dist Accuracy (160) | OOD Accuracy (160) | Overall Strict Acc (320) | Peak VRAM | Wall-Clock Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **17M_Seed42** | 42 | 17,063,808 | $1.1625$ | $15.6\%$ (25/160) | **$0.0\%$ (0/160)** | $7.8\%$ (25/320) | 1,143 MB | 28.0s |
| **38M_Seed42** | 42 | 38,285,824 | **$1.0009$** | $16.9\%$ (27/160) | **$0.0\%$ (0/160)** | $8.4\%$ (27/320) | 1,884 MB | 49.1s |
| **17M_Seed101** | 101 | 17,063,808 | $1.0742$ | **$18.1\%$ (29/160)** | **$0.6\%$ (1/160)** | **$9.4\%$ (30/320)** | 1,277 MB | 28.4s |
| **38M_Seed101** | 101 | 38,285,824 | **$0.9766$** | $15.6\%$ (25/160) | **$0.0\%$ (0/160)** | $7.8\%$ (25/320) | 1,879 MB | 51.0s |
| **17M Mean** | — | 17,063,808 | $1.1184$ | $16.9\%$ | **$0.3\%$** | **$8.6\%$** | 1,210 MB | 28.2s |
| **38M Mean** | — | 38,285,824 | **$0.9888$** | $16.3\%$ | **$0.0\%$** | **$8.1\%$** | 1,881 MB | 50.1s |

---

## 3. Detailed Mechanistic Deductions
1. **The In-Distribution Fitting Trap:** Both 17M and 38M learn in-distribution templates with moderate fidelity ($15.6\% - 18.1\%$), memorizing familiar tokens and syntax.
2. **The Out-of-Distribution Generalization Barrier:** When tested on strictly disjoint variables (`zeta, theta, omega`) and numbers ($52-110$), performance crashes to $0.0\%$ in 3 out of 4 runs.
3. **Implication for 38M Pretraining Gate:** Scaling alone from 17M to 38M yields faster loss reduction and higher memory consumption ($1.88\text{ GB}$ vs $1.14\text{ GB}$), but **fails Decision Rule 1** ("38M demonstrates reproducible improvement in strict held-out generalization"). Therefore, **authorizing a full-scale 38M pretraining campaign is scientifically unjustified at this stage**.
