# NR-017 Phase 7 — Compositional and Counterfactual Analysis Report
**Mission ID:** CV-GENESIS-NR-017  
**Date:** 2026-11-26  
**Evaluator Script:** `experiments/genesis_stage_g_017/run_compositional_tests.py`  
**Results Artifact:** `data/stage_g_017/compositional_counterfactual_results.json`  
**Models Evaluated:** Condition A (Direct), Condition B (Explicit Trace), Condition C (Compact Trace)

---

## 1. Experimental Design & Invariance Invariants
To test whether neural models learn reusable functional operations rather than surface lexical n-grams, counterfactual paired tasks were evaluated across all three conditions:
1. **Variable Renaming Invariance:** $f(x=12, y=15)$ vs $f(a=12, b=15)$ (Target: Invariant output).
2. **Value Mutation Sensitivity:** $S_0=20, +10, -5$ vs $S_0=20, +10, -15$ (Target: Distinct output).
3. **Commutativity Invariance:** $14 \times 6$ vs $6 \times 14$ (Target: Invariant output).
4. **Representation Invariance:** Two-step casing/reversal vs Direct reversal command (Target: Invariant output).

---

## 2. Quantitative Paired Evaluation Matrix

| Counterfactual Property | Direct Answer (Condition A) | Explicit Trace (Condition B) | Compact Trace (Condition C) | Theoretical Target |
| :--- | :---: | :---: | :---: | :---: |
| **Variable Renaming Invariance** | **SATISFIED** | **FAILED** ($x=4$ vs $a=64$) | **FAILED** ($x=15$ vs $a=5$) | Invariant |
| **Value Mutation Sensitivity** | **FAILED** (Output locked to `14`) | **FAILED** (Output locked to `S0=64`) | **FAILED** (Output locked to `S0=5`) | Sensitive (Distinct) |
| **Commutativity Invariance** | **SATISFIED** | **SATISFIED** | **FAILED** ($2*3=27$ vs $5*5=50$) | Invariant |
| **Representation Invariance** | **FAILED** | **FAILED** | **FAILED** | Invariant |
| **Total Invariants Satisfied** | **2 / 4** ($50\%$) | **1 / 4** ($25\%$) | **0 / 4** ($0\%$) | **4 / 4** ($100\%$) |

---

## 3. Scientific Discoveries
1. **The State Trace Sensitivity Curse:** In Condition B and Condition C, forcing the model to emit variable names inside intermediate states caused the network to **lose variable renaming invariance**. Changing variable from $x$ to $a$ caused the trace model to hallucinate completely different numbers ($x=4$ vs $a=64$), whereas Direct Answer produced the same answer (`51`).
2. **Value Mutation Insensitivity (Attractor Locking):** Across all three conditions, mutating underlying numbers ($+10, -5$ vs $+10, -15$) failed to produce distinct outputs. The networks remained locked to fixed token attractors (`14`, `S0=64`, `S0=5`).
3. **Core Conclusion:** Explicit state-trace supervision does not induce functional compositional invariance. It introduces additional variable-name tokens that destabilize downstream state generation.
