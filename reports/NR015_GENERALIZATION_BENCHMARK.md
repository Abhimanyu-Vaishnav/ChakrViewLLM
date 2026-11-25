# NR-015 Phase 2 — Controlled Generalization Benchmark Design
**Mission ID:** CV-GENESIS-NR-015  
**Date:** 2026-11-25  
**Benchmark Suite:** `data/sealed_eval_015/sealed_benchmark_300.json`  
**Total Tasks:** 325 strictly unseen tasks  
**Evaluation Harness:** Subprocess execution sandbox + strict assertion checking

---

## 1. Objective & Design Principles
To determine why 17M models achieve low cross-entropy loss but fail on downstream novel execution, the benchmark tests 12 distinct inductive reasoning families with zero prompt overlap against training corpora:
- **Disentangled Variable Binding:** New variable names (`alpha`, `beta`, `foo`, `bar`) and randomized numerical values.
- **Novel Task Templates:** Unseen combinations of string, list, and conditional rules.
- **Python Function Generation:** 40 programming tasks evaluated with isolated Python subprocesses executing strict hidden unit tests (`assert func(...) == ...`).
- **Sealed Ground Truths:** Ground truth labels and assertions are held completely separate from any model training or tuning data.

---

## 2. Task Family Distribution

| Family Index | Task Family | Count | Primary Capability Tested | Evaluation Mode |
| :---: | :--- | :---: | :--- | :--- |
| **01** | `simple_arithmetic` | 25 | Novel operand addition, subtraction, multiplication | Numeric / Exact |
| **02** | `sequence_continuation` | 25 | Linear arithmetic progression extrapolation | Numeric / Exact |
| **03** | `variable_binding` | 25 | Binding named variables to dynamic values | Symbolic Binding |
| **04** | `function_argument_binding` | 25 | Formal parameter substitution in mathematical expressions | Argument Binding |
| **05** | `io_transformation` | 25 | Few-shot pattern induction from input/output pairs | Rule Generalization |
| **06** | `conditional_logic` | 25 | Threshold comparison and branching (`PASS`/`FAIL`) | Logic Branching |
| **07** | `list_transformation` | 25 | Dynamic list operations (`reverse`, `min`, `max`, `sum`) | Algorithmic Lists |
| **08** | `string_transformation` | 25 | String manipulation (`upper`, `reverse`, `length`) | Symbolic Strings |
| **09** | `rule_induction` | 25 | Mapping extrapolation from observed scaling factors | Function Induction |
| **10** | `python_function_generation`| 40 | Function synthesis tested via subprocess execution | Subprocess Execution |
| **11** | `multistep_instruction` | 30 | Multi-step sequential transformation tracking | Sequential State |
| **12** | `novel_combinations` | 30 | Compositional combination of string case and repetition | Compositional Logic |
| **TOTAL** | **All 12 Families** | **325** | **Comprehensive Inductive Reasoning Suite** | **Multi-Modal Verification** |

---

## 3. Epistemic Verification Integrity
No deterministic router, regex extractor, or Python math tool is permitted to substitute for the pure-neural candidate during neural evaluation. Results will strictly distinguish pure-neural autoregressive output from hybrid cognitive system output.
