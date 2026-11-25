# NR-016 Phase 3 — Scientific Generalization Benchmark Design
**Mission ID:** CV-GENESIS-NR-016  
**Date:** 2026-11-25  
**Generator:** `experiments/generate_scientific_benchmark.py` (Deterministic seed: 2026)  
**Sealed Benchmark File:** `data/scientific_eval_016/scientific_benchmark_320.json` (SHA-256: `48e97598ee3fd64a1fca69bf160a58211e8c9b6d2ceb0f712d900142bbc57209`)  
**Training Corpus File:** `data/scientific_eval_016/scientific_train_1600.json` (SHA-256: `b2fdc0dac114058e71dc30865f4cf5b33abfbf683593b281c5987b86883b74e6`)  
**Task Count:** 320 sealed evaluation tasks across 8 task families (40 tasks per family: 20 in-distribution, 20 strict out-of-distribution)

---

## 1. Design Principles & Anti-Contamination Architecture
To eliminate any possibility of a model succeeding via memorized suffixes or template matching, the task generator enforces:
1. **Disjoint Lexical Partitions:**
   - In-Distribution Variables: `x, y, z, m, n, p, q, u, v, val_a, val_b`
   - Strict Out-of-Distribution Variables: `zeta, theta, omega, psi, sigma, lambda_var, kappa, chi`
2. **Disjoint Numeric Partitions:**
   - In-Distribution Range: $[5, 45]$
   - Out-of-Distribution Range: $[52, 110]$
3. **Disjoint Structural Grammars:** Distinct compositional rules and multi-step state update trajectories.
4. **Independent Deterministic Evaluation:** All benchmark labels and assertions are verified by independent evaluation logic.

---

## 2. Task Family Distribution (320 Tasks)

| Family Index | Family Identifier | Total Tasks | In-Dist Tasks | OOD Tasks | Target Capability |
| :---: | :--- | :---: | :---: | :---: | :--- |
| **01** | `variable_binding_and_renaming` | 40 | 20 | 20 | Dynamic slot binding to unseen variable tokens |
| **02** | `argument_order_changes` | 40 | 20 | 20 | Commutative vs non-commutative order tracking |
| **03** | `arithmetic_transformations` | 40 | 20 | 20 | Unseen scaling multiplier extrapolation |
| **04** | `nested_operations` | 40 | 20 | 20 | Compositional algebraic expression evaluation |
| **05** | `compositional_rule_application` | 40 | 20 | 20 | Rule chaining (case formatting + repetition) |
| **06** | `io_transformations` | 40 | 20 | 20 | Mathematical mapping induction from demonstrations |
| **07** | `multistep_state_updates` | 40 | 20 | 20 | Sequential accumulator tracking across multiple steps |
| **08** | `structured_code_generation` | 40 | 20 | 20 | Function generation tested via isolated subprocess assertions |
| **TOTAL** | **8 Distinct Families** | **320** | **160** | **160** | **Rigorous Generalization Testbed** |
