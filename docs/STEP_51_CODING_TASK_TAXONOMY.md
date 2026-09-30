# ChakrView Step 51: Coding Task Taxonomy & Capability Levels

- **Version**: 1.0.0
- **Scope**: Step 51 — Coding Language Acquisition & Project Arena Foundation
- **Purpose**: Define measurable, objective progression stages for model coding capabilities
- **Scientific Caveat**: These are target capability levels and evaluation boundaries; they are **NOT claims** that the current model possesses these capabilities.

---

## 1. Architectural Philosophy

Coding capability is not a monolithic binary attribute ("can code" vs "cannot code"). In ChakrView, software engineering competency is decomposed into seven discrete, verifiable capability levels. Each level provides specific, objective evaluation criteria that can be evaluated programmatically without human subjectivity.

```
       LEVEL 7: Iterative Engineering (Diagnose -> Patch -> Rerun -> Improve)
           LEVEL 6: Small Project Construction (Multi-file Spec -> Test -> Pass)
               LEVEL 5: Testing (Test Generation -> Edge Case Identification)
                   LEVEL 4: Transformation (Refactor -> Validate -> Clean)
                       LEVEL 3: Code Understanding (Signature -> I/O -> Bug Detection)
                           LEVEL 2: Local Code Completion (Statement -> Function)
                               LEVEL 1: Token & Syntax Learning (Brackets, Indentation, Keywords)
```

---

## 2. Seven-Level Coding Capability Taxonomy

### LEVEL 1 — Token & Syntax Learning
- **Focus**: Fundamental syntax primitives, lexical tokens, and structure.
- **Specific Tasks**:
  - Balanced delimiter completion: matching `()`, `[]`, `{}`, `""`, `''`.
  - Indentation continuity: matching 4-space indentation block levels.
  - Keyword positioning: valid use of `def`, `class`, `return`, `if`, `else`, `import`, `for`, `while`.
  - Operator syntax: valid assignment (`=`), equality (`==`), arithmetic (`+`, `-`, `*`), and logical (`and`, `or`, `not`).
- **Objective Metric**: AST parseability percentage ($P_{\text{parse}} = N_{\text{valid}} / N_{\text{total}}$), lexical error rate.
- **Expected Current Status**: **Measurable acquisition in Step 51** via pretraining on code shards.

---

### LEVEL 2 — Local Code Completion
- **Focus**: Single-line and single-function block completion.
- **Specific Tasks**:
  - Completing a missing return statement or calculation inside a function.
  - Filling in a loop accumulator or list comprehension.
  - Completing standard type annotations (`-> int:`, `-> List[str]:`).
- **Objective Metric**: Exact match or semantic equivalence on unit-tested functions, docstring-conditioned completion accuracy.
- **Expected Current Status**: **Early micro-scale completion evaluated in Step 51 benchmark**.

---

### LEVEL 3 — Code Understanding
- **Focus**: Structural analysis of existing code without execution.
- **Specific Tasks**:
  - Identifying input parameters, return types, and exceptions from code.
  - Identifying syntax errors, unbound local variables, or undefined imports.
  - Detecting simple anti-patterns (e.g. mutable default arguments in Python).
- **Objective Metric**: Classification accuracy on synthetic bug/property probing datasets.
- **Expected Current Status**: **Future research phase (Step 52+)**.

---

### LEVEL 4 — Code Transformation
- **Focus**: Non-destructive refactoring and semantic preservation.
- **Specific Tasks**:
  - Variable renaming across local scope.
  - Converting iterative loops to vectorized or comprehension expressions.
  - Adding type checks and boundary validation to untyped functions.
- **Objective Metric**: Pre/post test suite preservation: all existing tests must pass after transformation.
- **Expected Current Status**: **Future research phase (Step 53+)**.

---

### LEVEL 5 — Test Generation & Verification
- **Focus**: Verification mindset and boundary discovery.
- **Specific Tasks**:
  - Generating deterministic unit test cases for a target function.
  - Formulating boundary test inputs (e.g., empty string, negative integers, null values).
  - Asserting correct output types and exception handling.
- **Objective Metric**: Line and branch test coverage, mutation testing score (killing synthetic mutants).
- **Expected Current Status**: **Future research phase (Step 54+)**.

---

### LEVEL 6 — Small Project Construction
- **Focus**: Multi-file project synthesis from a specification.
- **Specific Tasks**:
  - Given a `project_spec.md` with interfaces and test requirements:
    1. Create package structure (`__init__.py`, module files).
    2. Implement modules meeting API signatures.
    3. Provide or verify passing test suites.
- **Objective Metric**: Project test suite pass rate ($100\%$ required for project completion).
- **Expected Current Status**: **Architectural Foundation laid in Step 51 Project Arena**.

---

### LEVEL 7 — Iterative Software Engineering
- **Focus**: The full closed-loop developmental cycle.
- **Specific Tasks**:
  - Execute test suite $\to$ capture traceback/assertion failure $\to$ localize defect $\to$ generate patch $\to$ re-run tests $\to$ iterate until all tests pass.
  - Log failures and solutions into persistent cognitive memory (RIL).
- **Objective Metric**: Iteration convergence rate ($N_{\text{iterations}} \le 5$), regression avoidance rate.
- **Expected Current Status**: **The ultimate objective of the ChakrView Project Arena (Steps 55+)**.

---

## 3. Scientific Boundary Summary

| Level | Capability Category | Evaluation Harness | Step 51 Implementation Status |
|:---:|:---|:---|:---:|
| **1** | Token / Syntax | AST Validator | **Implemented & Tested** |
| **2** | Local Completion | Single-Function Probe | **Implemented & Tested** |
| **3** | Code Understanding | Property Probe | Architecture Defined |
| **4** | Transformation | Refactoring Validator | Architecture Defined |
| **5** | Test Generation | Pytest Harness | Architecture Defined |
| **6** | Project Construction| Project Arena Workspace | **Workspace Implemented** |
| **7** | Iterative Engineering| Closed-Loop Arena Loop | **Contract Formulated** |
