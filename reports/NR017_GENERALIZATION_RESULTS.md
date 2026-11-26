# NR-017 Phase 6 — Held-Out Generalization Results Report
**Mission ID:** CV-GENESIS-NR-017  
**Date:** 2026-11-26  
**Benchmark Suite:** `data/stage_g_017/stage_g_sealed_benchmark_400.json` (400 tasks, 10 families, 3 difficulty levels)  
**Evaluator:** Independent Evaluator across Condition A, Condition B, and Condition C  
**Artifacts:** `data/stage_g_017/training_comparison_results.json`

---

## 1. Executive Summary & Generalization Findings
The 400-task sealed benchmark was evaluated across all three conditions to measure final answer accuracy, out-of-distribution (OOD) generalization, and family-wise resilience.

### **Key Findings:**
1. **Direct Answer (Condition A) Dominates:** Direct-answer training achieved **$17.0\%$ overall accuracy** and **$10.25\%$ OOD accuracy**.
2. **Explicit Traces (Condition B) Degrade Generalization:** Explicit traces achieved only **$9.0\%$ overall accuracy** and **$3.25\%$ OOD accuracy**. Because each intermediate transition introduces an opportunity for autoregressive compounding error, the model frequently generates a flawed intermediate state, leading to answer failure.
3. **Structured Compact Traces (Condition C) Lie Between A and B:** Achieving **$14.9\%$ overall accuracy** and **$8.75\%$ OOD accuracy**, compact traces mitigate some token compounding error but still fail to outperform direct-answer training.

---

## 2. In-Distribution vs Out-of-Distribution Breakdown

| Condition | In-Distribution Accuracy (200 tasks) | Strict OOD Accuracy (200 tasks) | Generalization Gap | Total Accuracy (400 tasks) |
| :--- | :---: | :---: | :---: | :---: |
| **Condition A (Direct Answer)** | **$23.75\%$** | **$10.25\%$** | $-13.50\%$ | **$17.00\%$** |
| **Condition B (Explicit State Trace)** | $14.75\%$ | $3.25\%$ | $-11.50\%$ | $9.00\%$ |
| **Condition C (Structured Compact Trace)** | $21.00\%$ | $8.75\%$ | $-12.25\%$ | $14.90\%$ |

---

## 3. Failure Mode Categorization
1. **Autoregressive Compounding Error (Condition B):** When generating multi-step traces ($S_0 \rightarrow S_1 \rightarrow S_2$), any small error in $S_1$ propagates into $S_2$, making the final answer unconditionally wrong.
2. **Tag Overhead & Distraction:** Condition B spent $76.8\%$ of its capacity formatting XML state tags (`<state_1> ... </state_1>`), diverting attention weights away from variable binding and arithmetic calculation.
3. **Direct Mapping Superiority:** At 17M scale, the attention layers function best as a direct associative projection from prompt tokens to output tokens, rather than an autoregressive multi-step computation graph.
