# ChakrView v0.7.0 Environment Setup & Reproducibility Guide

This guide describes how to reproduce the 6-category coding capability evaluation, run the data pipeline tests, and execute the coding assistant CLI.

---

## 1. Prerequisites

- **Python**: Python 3.10 – 3.14 (64-bit).
- **Environment**: Virtual environment active with `torch`, `numpy`, `pytest`.

```powershell
# Activate local environment
.venv\Scripts\Activate.ps1
```

---

## 2. Reproduce the Coding Capability Benchmark

Execute the sandboxed evaluation runner directly:

```powershell
.venv\Scripts\python evaluation/coding_suite/run_coding_evaluation.py
```

Expected Output:
```text
======================================================================
RUNNING CODING EVALUATION: Canonical Pretrained (Step 10,000) (greedy)
======================================================================
  [1/6] CODE-COMP-01   (code_continuation         ) | FAIL | Syntax: True  | Err: AssertionError
  [2/6] CODE-EXPL-01   (code_explanation          ) | FAIL | Syntax: False | Err: None
  [3/6] CODE-BUG-01    (bug_identification_and_repair) | FAIL | Syntax: False | Err: SyntaxError: expected an indented block...
  [4/6] CODE-IMPL-01   (small_function_implementation) | FAIL | Syntax: False | Err: SyntaxError: unterminated triple-quoted...
  [5/6] CODE-TEST-01   (test_case_generation      ) | PASS | Syntax: True  | Err: None
  [6/6] CODE-SUMM-01   (code_summarization        ) | FAIL | Syntax: True  | Err: None
======================================================================
SUMMARY: Success: 1/6 (16.7%) | Syntax: 3/6 (50.0%) | Speed: ~138 tok/s
```

Results are saved deterministically to `evaluation/coding_suite/coding_baseline_results.json`.

---

## 3. Run Focused Test Suites

Run the complete test suite for the coding data pipeline, coding assistant, and sandboxed test execution:

```powershell
.venv\Scripts\pytest tests/test_coding_data_pipeline.py tests/test_coding_assistant.py tests/test_coding_suite.py
```

Expected Output: `8 passed in ~3s`.

Run release regression tests to ensure v0.6.0 integrity:
```powershell
.venv\Scripts\python -m unittest tests/test_v060_release_smoke.py
```

Expected Output: `Ran 7 tests ... OK`.

---

## 4. Run the Coding Assistant CLI

### Draft Code
```powershell
.venv\Scripts\python -m chakrview.runtime.assistant_cli code draft "Write a function to return factorial of n"
```

### Explain Code
```powershell
.venv\Scripts\python -m chakrview.runtime.assistant_cli code explain "def add(a, b): return a + b"
```

### Fix Bug
```powershell
.venv\Scripts\python -m chakrview.runtime.assistant_cli code fix "def f(): return x" --error "NameError: name 'x' is not defined"
```

### Sandboxed Execution [Tool-Assisted]
```powershell
.venv\Scripts\python -m chakrview.runtime.assistant_cli code sandbox "def add(a, b): return a + b" --test-assertions "assert add(2, 3) == 5"
```
