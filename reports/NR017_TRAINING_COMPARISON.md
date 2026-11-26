# NR-017 Phase 4 — Controlled Training Comparison Report
**Mission ID:** CV-GENESIS-NR-017  
**Date:** 2026-11-26  
**Experimental Design:** Controlled Comparison of 3 Conditions across 2 Independent Seeds (Seed 42 and Seed 101)  
**Primary Architecture:** `ChakrGenesis-17M` ($17,063,808$ parameters, 10 layers, $d_{model}=384$)  
**Training Corpus:** `data/stage_g_017/stage_g_train_1800.json` (1,800 identical underlying tasks)  
**Sealed Benchmark:** `data/stage_g_017/stage_g_sealed_benchmark_400.json` (400 tasks: 200 in-distribution, 200 strict OOD)  
**Results Artifact:** `data/stage_g_017/training_comparison_results.json`

---

## 1. Executive Summary & Core Scientific Finding
A strictly controlled empirical comparison was conducted to test whether explicit intermediate state-trace supervision improves algorithmic generalization compared with direct-answer training.

### **The Decisive Scientific Verdict:**
**DIRECT-ANSWER OUTPERFORMS STATE-TRACE SUPERVISION**
- **Condition A (Direct Answer):** Overall Strict Accuracy = **$17.0\%$** (Seed 42: $16.8\%$, Seed 101: $17.2\%$), OOD Accuracy = **$10.25\%$**, Token Budget = **$63,118$ tokens**.
- **Condition B (Explicit State Trace):** Overall Strict Accuracy = **$9.0\%$** (Seed 42: $9.8\%$, Seed 101: $8.2\%$), OOD Accuracy = **$3.25\%$**, Token Budget = **$248,774$ tokens** ($3.94\times$ more tokens).
- **Condition C (Structured Compact Trace):** Overall Strict Accuracy = **$14.9\%$** (Seed 42: $14.0\%$, Seed 101: $15.8\%$), OOD Accuracy = **$8.75\%$**, Token Budget = **$146,376$ tokens** ($2.32\times$ more tokens).

**Conclusion:** Explicit intermediate state traces consumed **$3.94\times$ more target tokens**, but **degraded out-of-distribution (OOD) accuracy** from $10.25\%$ down to $3.25\%$ and overall accuracy from $17.0\%$ down to $9.0\%$.

---

## 2. Quantitative Results Table (3 Conditions $\times$ 2 Seeds)

| Condition & Seed | Format Description | Trained Tokens | Final Loss | In-Dist Acc (200) | OOD Acc (200) | Overall Acc (400) | Peak VRAM |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Condition_A_Seed42** | Direct Answer | $63,118$ | $0.2710$ | $23.0\%$ | $10.5\%$ | $16.8\%$ | 1,680 MB |
| **Condition_A_Seed101** | Direct Answer | $63,118$ | $0.2617$ | $24.5\%$ | $10.0\%$ | $17.2\%$ | 1,682 MB |
| **Condition_A Mean** | **Direct Answer** | **$63,118$** | **$0.2663$** | **$23.75\%$** | **$10.25\%$** | **$17.00\%$** | **1,681 MB** |
| **Condition_B_Seed42** | Explicit State Trace | $248,774$ | $0.4598$ | $14.5\%$ | $5.0\%$ | $9.8\%$ | 1,682 MB |
| **Condition_B_Seed101** | Explicit State Trace | $248,774$ | $0.4383$ | $15.0\%$ | $1.5\%$ | $8.2\%$ | 1,683 MB |
| **Condition_B Mean** | **Explicit State Trace** | **$248,774$** | **$0.4490$** | **$14.75\%$** | **$3.25\%$** | **$9.00\%$** | **1,682 MB** |
| **Condition_C_Seed42** | Compact Trace | $146,376$ | $0.7324$ | $20.5\%$ | $7.5\%$ | $14.0\%$ | 1,685 MB |
| **Condition_C_Seed101** | Compact Trace | $146,376$ | $0.7024$ | $21.5\%$ | $10.0\%$ | $15.8\%$ | 1,688 MB |
| **Condition_C Mean** | **Compact Trace** | **$146,376$** | **$0.7174$** | **$21.00\%$** | **$8.75\%$** | **$14.90\%$** | **1,686 MB** |

---

## 3. Comparison 1 vs Comparison 2 Analysis
- **Comparison 1 (Equal Task Exposure):** Under identical 1,800 task training exposure, explicit state traces required $394\%$ of the token budget, but reduced overall accuracy from $17.00\%$ to $9.00\%$.
- **Comparison 2 (Token Budget Accounting):** Condition A achieved almost double the accuracy of Condition B using only $\approx 25\%$ of the tokens consumed by Condition B.
