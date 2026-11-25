# NR-016 Phase 6 — Coding Generalization Report
**Mission ID:** CV-GENESIS-NR-016  
**Date:** 2026-11-25  
**Benchmark Suite:** `data/coding_eval_016/coding_benchmark_50.json` (50 unseen Python programming tasks)  
**Evaluator Script:** `experiments/eval_coding_50.py`  
**Results Artifact:** `data/coding_eval_016/coding_eval_results_50.json`  
**Model Under Test:** Canonical `ChakrGenesis-17M Stage-D Best` (`2c21632e`)

---

## 1. Quantitative Benchmark Results

| Coding Generalization Metric | Measured Value | Percentage |
| :--- | :---: | :---: |
| **Total Tasks Evaluated** | 50 | $100\%$ |
| **Output Generated** | 50 / 50 | $100\%$ |
| **Python AST Syntax Validity** | **0 / 50** | **$0.0\%$** |
| **Execution Assertion Pass Rate** | **0 / 50** | **$0.0\%$** |
| **Dominant Failure Mode** | Runaway backtick repetition (`Fix: Use ````...`) | SyntaxError (100%) |

---

## 2. Mechanistic Failure Isolation
Inspection of the raw token streams across all 50 tasks reveals a severe attention attractor mode:
1. **The Runaway Backtick Attractor:** The prompt asks the model to "Return only the Python function code". The model immediately latches onto markdown fence triggers, outputting consecutive backticks:
   `"Fix: Use `````````````````python\nfrom doctest.testmod()..."`
2. **Defective Function Header Emission:** Rather than emitting the requested function name (`def multiply_elements(lst):`), the model repeatedly emits a distorted generic function header:
   `def master('', '', 'end', 'end...`
3. **Absence of AST Binding:** The model treats Python code tokens not as an abstract syntax tree with hierarchical scope and identifiers, but as a linear token-level texture of docstrings, backticks, and import statements.
