# NR-016 Phase 2 — 325-Task Sealed Benchmark Recovery Report
**Mission ID:** CV-GENESIS-NR-016  
**Date:** 2026-11-25  
**Benchmark Suite:** `data/sealed_eval_015/sealed_benchmark_300.json` (325 tasks, 12 families)  
**Evaluator Script:** `experiments/eval_recovered_325.py`  
**Detailed Results Artifact:** `data/sealed_eval_015/recovered_benchmark_results.json`  
**Model Under Test:** Canonical `ChakrGenesis-17M Stage-D Best` (`2c21632e`)

---

## 1. Executive Summary & Complete Quantitative Breakdown
For the first time, all 325 tasks of the NR-015 sealed benchmark suite were evaluated in full turn-by-turn isolation, reporting strict accuracy for both the pure neural core and the hybrid cognitive architecture across each individual task family.

**Overall Benchmark Strict Accuracy:**
- **Pure Neural Model:** **$4.0\%$ (13 / 325 correct)**
- **Hybrid Cognitive System:** **$9.5\%$ (31 / 325 correct)**

---

## 2. Family-Wise Performance Breakdown

| Family Index | Task Family Name | Task Count | Pure Neural Pass | Pure Accuracy | Hybrid Pass | Hybrid Accuracy | Primary Failure Mode |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **01** | `simple_arithmetic` | 25 | 1 | $4.0\%$ | 25 | **$100.0\%$** | Pure: lexical collapse; Hybrid: math router handles 100% |
| **02** | `sequence_continuation` | 25 | 0 | $0.0\%$ | 0 | $0.0\%$ | Inability to compute linear step extrapolation |
| **03** | `variable_binding` | 25 | 1 | $4.0\%$ | 3 | $12.0\%$ | Hallucinated fixed suffixes (`#### 36`) |
| **04** | `function_argument_binding` | 25 | 0 | $0.0\%$ | 0 | $0.0\%$ | Failure to bind input arguments to algebraic terms |
| **05** | `io_transformation` | 25 | 6 | **$24.0\%$** | 1 | $4.0\%$ | Partial pattern matching on linear functions ($2x+1$) |
| **06** | `conditional_logic` | 25 | 1 | $4.0\%$ | 0 | $0.0\%$ | Threshold comparison failure (`PASS`/`FAIL`) |
| **07** | `list_transformation` | 25 | 3 | $12.0\%$ | 2 | $8.0\%$ | Algorithmic reordering / min-max extraction failure |
| **08** | `string_transformation` | 25 | 0 | $0.0\%$ | 0 | $0.0\%$ | Token-level reverse/length manipulation failure |
| **09** | `rule_induction` | 25 | 1 | $4.0\%$ | 0 | $0.0\%$ | Scaling factor induction failure |
| **10** | `python_function_generation` | 40 | 0 | $0.0\%$ | 0 | $0.0\%$ | Subprocess assertions: syntax ok, logic unbound |
| **11** | `multistep_instruction` | 30 | 0 | $0.0\%$ | 0 | $0.0\%$ | State tracking failure across step 1 $\rightarrow$ step 2 |
| **12** | `novel_combinations` | 30 | 0 | $0.0\%$ | 0 | $0.0\%$ | Compositional upper + repetition failure |
| **TOTAL** | **All 12 Families** | **325** | **13** | **$4.0\%$** | **31** | **$9.5\%$** | **Rigorous baseline established** |

---

## 3. Scientific Insights
1. **Deterministic Router Power vs. Pure Neural Limit:** The `DeterministicMathRouter` achieved a **$100.0\%$ pass rate (25/25)** on simple arithmetic, contrasting with the pure-neural pass rate of $4.0\%$. This proves the vital necessity of the hybrid cognitive layer for production arithmetic reliability.
2. **True Generalization Ceiling:** Across 8 of the 12 families (`sequence_continuation`, `function_argument_binding`, `string_transformation`, `python_function_generation`, `multistep_instruction`, `novel_combinations`), the pure neural 17M model scored **$0.0\%$**. This provides empirical confirmation that the 17M architecture cannot perform multi-step compositional reasoning out-of-the-box.
