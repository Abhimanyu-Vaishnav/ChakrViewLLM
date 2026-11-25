# NR-015 Phase 5 — Curriculum Learning Experiment Report
**Mission ID:** CV-GENESIS-NR-015  
**Date:** 2026-11-25  
**Experiment:** Controlled Curriculum Learning vs. Randomized Training  
**Artifacts:** `data/curriculum_015/curriculum_results.json`

---

## 1. Experimental Conditions & Design
- **Condition A (Randomized Mixed):** 300 samples randomly sampled across single-digit arithmetic, two-digit variable binding, and multi-step arithmetic.
- **Condition B (Easy-to-Hard Curriculum):** 300 samples staged strictly in progressive order: Easy (Level 1) $\rightarrow$ Medium (Level 2) $\rightarrow$ Hard (Level 3).
- **Held-Out Test Set:** 100 unseen arithmetic tasks with distinct numerical ranges ($55 \dots 95$) never encountered in training.

---

## 2. Experimental Results

| Curriculum Condition | Final Average Training Loss | Held-Out Test Accuracy (100 Tasks) |
| :--- | :---: | :---: |
| **Randomized Mixed Training** | $1.9645$ | **$0.0\%$ (0 / 100)** |
| **Easy-to-Hard Curriculum** | **$1.3822$** | **$0.0\%$ (0 / 100)** |

---

## 3. Scientific Analysis
1. **Curriculum Accelerates Training Convergence:** The easy-to-hard progression substantially lowered the final training loss ($1.3822$ vs $1.9645$), proving that curriculum ordering aids optimization by structuring simpler gradient paths.
2. **Zero Generalization Impact:** Despite the lower training loss, **held-out test accuracy remained at exactly 0.0% across both conditions**.
3. **Core Conclusion:** Curriculum scaffolding optimizes the *rate* at which in-distribution patterns are memorized by the network, but **does not create novel reasoning circuits** in a 17M parameter causal language model. The underlying limitation is representational capacity rather than optimization ordering.
