# ChakrView Step 51: Progressive Coding Curriculum & Capability Ladder

- **Version**: 1.0.0
- **Scope**: Step 51 — Project-Based Coding Capability Acquisition & Project Arena Foundation
- **Structure**: 11-Level Progressive Curriculum (Level 0 through Level 10)
- **Status**: Formally Specified

---

## 1. Curriculum Philosophy & Pedagogical Axioms

ChakrView rejects the brute-force approach of feeding uncurated internet code into a neural network and hoping for emergent software engineering competence. True software engineering discipline requires an incremental, pedagogical curriculum where basic syntactic regularities are mastered before multi-file projects, and small bounded projects are thoroughly tested before attempting self-modification.

```
 LEVEL 10: Controlled ChakrView Self-Maintenance (Isolated Branches, Invariant Gate)
     LEVEL 9: Autonomous Project Iteration (Closed-Loop Diagnosis & Re-testing)
         LEVEL 8: Complex Multi-Module Projects (Data Pipelines, Layered Abstractions)
             LEVEL 7: Real-World Small Applications (CLI Tools, State Engines)
                 LEVEL 6: Code Refactoring (Semantic-Preserving Transformations)
                     LEVEL 5: Feature Extension (Adding Endpoints Without Regressions)
                         LEVEL 4: Bug Fixing & Defect Repair (Traceback -> Localize -> Patch)
                             LEVEL 3: Projects with Tests (Test-Driven Assertions)
                                 LEVEL 2: Small Multi-File Projects (Local Imports & Packages)
                                     LEVEL 1: Single-File Utilities (Pure Functions & Classes)
                                         LEVEL 0: Basic Programming Generation (Tokens, Syntax, AST)
```

---

## 2. Eleven-Level Coding Curriculum Specification

---

### LEVEL 0 — Basic Programming Generation
- **Task Type**: Elementary token and syntax synthesis.
- **Required Capabilities**:
  - Balanced delimiters: matching `()`, `[]`, `{}`, `""`, `''`.
  - Consistent indentation: 4-space indentation block levels.
  - Keyword positioning: valid use of `def`, `class`, `return`, `if`, `else`, `for`, `while`, `import`.
  - Literals & operators: integers, floats, booleans, string escaping, arithmetic and comparison operators.
- **Evaluation Criteria**: Abstract Syntax Tree parseability ($100\%$ valid `ast.parse()`).
- **Failure Conditions**: Any `SyntaxError`, unclosed strings, or indentation mismatch.
- **Advancement Criteria**: $\ge 98\%$ syntax parseability on 100 random code completion probes.

---

### LEVEL 1 — Single-File Utilities
- **Task Type**: Standalone utility functions and pure algorithms.
- **Required Capabilities**:
  - Pure function implementation given docstrings, signatures, and type annotations.
  - Basic algorithms: math utilities (factorial, gcd, is_prime), string helpers (is_palindrome, count_words), array manipulation (clamp, reverse).
  - Explicit error handling: raising `ValueError` or `TypeError` on invalid inputs.
- **Evaluation Criteria**: Direct execution with input/output assertions.
- **Failure Conditions**: Unhandled runtime exceptions (`TypeError`, `ZeroDivisionError`), incorrect return values.
- **Advancement Criteria**: $\ge 90\%$ test pass rate across 50 canonical utility tasks.

---

### LEVEL 2 — Small Multi-File Projects
- **Task Type**: Package organization and local cross-file module imports.
- **Required Capabilities**:
  - Structuring a package directory (`__init__.py`, `core.py`, `utils.py`).
  - Correct relative and absolute local imports (`from .utils import helper`).
  - Separation of concerns: dividing data structures, business logic, and entrypoints.
- **Evaluation Criteria**: Successful import resolution and execution across files in an isolated workspace.
- **Failure Conditions**: `ModuleNotFoundError`, circular imports, undefined symbol imports.
- **Advancement Criteria**: $\ge 90\%$ clean execution across 30 multi-file project templates.

---

### LEVEL 3 — Projects with Tests
- **Task Type**: Test-driven construction and assertion verification.
- **Required Capabilities**:
  - Writing deterministic unit tests using `pytest` or `unittest`.
  - Formulating boundary test cases (empty inputs, negative values, null bounds).
  - Asserting exact expected values, exception types, and boolean conditions.
- **Evaluation Criteria**: Pytest execution in the Project Arena sandbox with zero test failures.
- **Failure Conditions**: Assertion errors, test suite crashes, tests passing vacuously without executing code.
- **Advancement Criteria**: $100\%$ passing tests across 25 tested project specifications.

---

### LEVEL 4 — Bug Fixing & Defect Repair
- **Task Type**: Localizing and patching intentional bugs from traceback diagnostics.
- **Required Capabilities**:
  - Reading and interpreting Python tracebacks, exception types, and line numbers.
  - Localizing defects: identifying off-by-one errors, inverted conditionals, or missing null checks.
  - Generating surgical patches: modifying only the defective lines without breaking adjacent logic.
- **Evaluation Criteria**: Previously failing test suite transitions from $0\%$ to $100\%$ passing.
- **Failure Conditions**: Patch fails to fix the bug, introduces syntax errors, or causes regressions.
- **Advancement Criteria**: $\ge 80\%$ repair success rate within $\le 3$ iterations across 25 buggy projects.

---

### LEVEL 5 — Feature Implementation
- **Task Type**: Extending an existing project with new functionality.
- **Required Capabilities**:
  - Reading and adhering to existing project architecture, coding conventions, and type contracts.
  - Implementing requested new methods or classes.
  - Adding new tests covering the feature without breaking existing test suites.
- **Evaluation Criteria**: Existing test suite remains $100\%$ green; new feature tests achieve $100\%$ pass rate.
- **Failure Conditions**: Regression in existing tests, broken backward compatibility, missing feature tests.
- **Advancement Criteria**: $\ge 85\%$ feature acceptance rate without regressions across 20 extension tasks.

---

### LEVEL 6 — Code Refactoring
- **Task Type**: Semantic-preserving structural improvement.
- **Required Capabilities**:
  - Variable and function renaming for readability and consistency.
  - Eliminating code duplication (DRY principle).
  - Extracting helper methods and simplifying complex conditional nests.
- **Evaluation Criteria**: Full existing test suite passes identically before and after refactoring; code quality metrics improve (reduced cyclomatic complexity, zero duplication).
- **Failure Conditions**: Any behavioral alteration or broken test assertion.
- **Advancement Criteria**: $100\%$ test suite preservation across 20 refactoring tasks.

---

### LEVEL 7 — Real-World Small Applications
- **Task Type**: Self-contained applications with state and I/O.
- **Required Capabilities**:
  - CLI argument parsing (`argparse` or sys.argv).
  - File-based persistence (JSON serialization, CSV readers).
  - State management (simple key-value stores, task queues, session managers).
- **Evaluation Criteria**: Multi-scenario integration testing in the Arena sandbox.
- **Failure Conditions**: Unhandled I/O exceptions, data corruption, process hangs.
- **Advancement Criteria**: $\ge 80\%$ integration pass rate across 15 application specifications.

---

### LEVEL 8 — Complex Multi-Module Projects
- **Task Type**: Layered architectures with interdependent components.
- **Required Capabilities**:
  - Managing multiple interconnected packages (domain logic, storage, API, interfaces).
  - Configuration injection and dependency management.
  - Comprehensive integration and mock testing.
- **Evaluation Criteria**: Complete test suite pass rate, clean dependency graphs.
- **Failure Conditions**: Dependency cycle, architectural leaking, integration failures.
- **Advancement Criteria**: $\ge 75\%$ project completion rate across 10 complex multi-module projects.

---

### LEVEL 9 — Autonomous Project Iteration
- **Task Type**: Closed-loop autonomous engineering cycle.
- **Required Capabilities**:
  - End-to-end orchestration: Plan $\to$ Write $\to$ Test $\to$ Observe $\to$ Diagnose $\to$ Repair $\to$ Verify.
  - Multi-round convergence without human intervention.
  - Recording lessons learned and failure patterns into persistent cognitive memory (RIL).
- **Evaluation Criteria**: Autonomous convergence rate ($\le 5$ iterations), zero regressions.
- **Failure Conditions**: Infinite repair loops, repeated identical mistakes, degradation of test passes.
- **Advancement Criteria**: $\ge 75\%$ autonomous convergence rate across 10 challenging project specifications.

---

### LEVEL 10 — Controlled ChakrView Self-Maintenance (Future Milestone)
- **Task Type**: Autonomous diagnosis, testing, and patching of ChakrView's own peripheral subsystems.
- **Required Capabilities**:
  - Operating strictly on isolated Git branches in sandboxed environments.
  - Reading and respecting ChakrView architectural invariants ($\Delta W_{\text{baseline}} = 0$, parameter counts, format specs).
  - Running ChakrView's full regression test suite (1,402+ tests).
  - Generating candidate pull requests with automated test evidence and regression proof.
- **Evaluation Criteria**: 100% full repository regression pass rate, zero invariant violations, verified human maintainer sign-off.
- **Failure Conditions**: Any invariant mutation, any broken existing test, any un-sandboxed self-modification.
- **Advancement Criteria**: Zero human interventions required for verified peripheral maintenance tasks.
- **Status in Step 51**: **Strictly Deferred to Future Steps (Step 60+)**. Architectural contract formulated now.
