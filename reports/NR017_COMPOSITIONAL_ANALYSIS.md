# NR-017 Phase 7 — Compositional and Counterfactual Analysis Report
**Mission ID:** CV-GENESIS-NR-017  
**Date:** 2026-11-26  
**Evaluator Script:** `experiments/genesis_stage_g_017/compositional_tests.py`  
**Results Artifact:** `data/stage_g_017/compositional_counterfactual_results.json`

---

## 1. Executive Summary & Experimental Methodology
To evaluate whether models learned genuine computational rules versus superficial token templates, paired counterfactual prompts were administered across four behavioral regimes:
1. **Variable Renaming Invariance:** Preserves values and operations while renaming variable tokens ($x, y \rightarrow a, b$). Expected: Output preserved.
2. **Value Perturbation Sensitivity:** Preserves variable names while modifying numerical values ($12, 15 \rightarrow 12, 30$). Expected: Output changes.
3. **Commutative Reordering:** Reorders independent commutative operations ($A \text{ then } B$ vs $B \text{ then } A$). Expected: Output preserved.
4. **State Update Perturbation:** Alters sequential delta values. Expected: Output changes.

---

## 2. Empirical Findings: Direct Answer (Condition A) vs Trace (Condition B)

| Counterfactual Regime | Expected Behavior | Condition A (Direct) Result | Condition B (Trace) Result | Mechanistic Analysis |
| :--- | :---: | :---: | :---: | :--- |
| **1. Variable Renaming** | Invariant | **PASS** (Outputs `'<answer> 51 </answer>'`) | FAIL (Hallucinates divergent states) | Direct ignores variable labels; Trace latches onto variable names |
| **2. Value Perturbation** | Sensitive | FAIL (Outputs `'<answer> 51 </answer>'` for both) | **PASS** (States alter: `x=4` vs `x=64`) | Direct suffers from template collapse; Trace reacts to numbers |
| **3. Commutative Order** | Invariant | FAIL (`47` vs `44`) | FAIL (State transitions diverge) | Neither model has internal commutativity |
| **4. State Update Delta** | Sensitive | FAIL (Outputs `'<answer> 51 </answer>'` for both) | **PASS** (States alter: `S0=50` vs `S0=64`) | Direct collapses into attractor; Trace tracks input changes |

---

## 3. Scientific Synthesis
- **The Blind Spot of Condition A:** Direct answer models suffer from severe **attractor collapse** (consistently outputting memorized numbers like `51` regardless of modified input values), passing renaming invariance only because they output the same number regardless of input!
- **The Fragility of Condition B:** While state traces force the model to be sensitive to input numbers (changing intermediate states when numbers change), they break under variable renaming ($x, y \rightarrow a, b$) because the state generation circuit is entangled with specific variable tokens.
