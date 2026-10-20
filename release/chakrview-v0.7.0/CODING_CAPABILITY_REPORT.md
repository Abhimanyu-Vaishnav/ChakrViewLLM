# ChakrView v0.7.0 Coding Capability Report

**Date**: 2026-10-19  
**Core Model**: `ChakrMicro` (Step 10,000 Canonical Core, 3.44M params, SHA-256 `96357705...e9f471ee`)  
**Harness**: `evaluation/coding_suite/run_coding_evaluation.py`  
**Isolation**: Subprocess execution with restricted builtins and AST security filters  

---

## 1. Category-by-Category Results

| Task ID | Task Category | Prompt Target | Syntax Valid | Test Passed | Generation / Execution Failure |
| :--- | :--- | :--- | :---: | :---: | :--- |
| `CODE-COMP-01` | **Code Continuation** | Recursive Factorial | **YES** | NO | Emits valid comment syntax, but fails unit test (`AssertionError`) |
| `CODE-EXPL-01` | **Code Explanation** | Binary Search | NO | NO | Generates repetitive filler tokens without explaining logic |
| `CODE-BUG-01` | **Bug Repair** | Sum First N Off-by-One | NO | NO | Missing indentation block (`SyntaxError`) |
| `CODE-IMPL-01` | **Small Implementation** | Palindrome Function | NO | NO | Unterminated triple-quoted docstring (`SyntaxError`) |
| `CODE-TEST-01` | **Test Generation** | Add Assertion | **YES** | **YES** | Emits `# If the result of...` with correct assert test pass |
| `CODE-SUMM-01` | **Code Summarization**| Compute Stats | **YES** | NO | Valid syntax comments, but misses key summary concepts |

---

## 2. Quantitative Summary

- **Total Tasks**: 6
- **Test Pass Rate**: **1/6 (16.7%)**
- **Syntax Validity Rate**: **3/6 (50.0%)**
- **Exact / Substring Match Rate**: **0/6 (0.0%)**
- **Security Policy Rejections**: **0**
- **Timeouts**: **0**
- **Average CPU Throughput**: **138.84 tokens/second**
- **Average Execution Latency**: **~49 ms**

---

## 3. Real Bottlenecks & Architectural Findings

1. **Token Prediction vs Relational Semantics**:
   The 3.44M base model easily completes short single-line prompt conventions (such as `assert`), but cannot construct multi-hop algorithmic structures from zero-shot prompts.
2. **Need for Relational SFT**:
   Fine-tuning on paired instruction-code examples formatted by `CodingDataPipeline` is necessary to teach the model function scoping and indentation invariants.
3. **AST Tool Grounding**:
   AST pre-checking provides an instant, zero-cost filter to catch 50% of malformed completions before expensive execution.
