# ChakrView Step 51: Project Arena Architecture Specification

- **Version**: 1.0.0
- **Scope**: Step 51 — Coding Language Acquisition & Project Arena Foundation
- **Component**: `chakrview/arena/`
- **Execution Model**: CPU-First, Sandboxed Subprocess Isolation

---

## 1. Executive Summary & Purpose

The **Project Arena** is ChakrView's controlled, sandboxed environment for evaluating and cultivating software engineering capabilities through real code execution. Rather than relying solely on static token perplexity, Project Arena grounds evaluation in programmatic reality: generating files, running tests, capturing tracebacks, and measuring test pass rates.

To uphold the strict "fail-closed" security policy, the model and generated code are never granted direct execution privileges within the host ChakrView runtime. All execution occurs in disposable, isolated filesystem sandboxes with strict timeouts, resource ceilings, and sanitized environment variables.

---

## 2. Architectural Boundaries & Isolation Model

```
┌────────────────────────────────────────────────────────────────────────┐
│                        CHAKRVIEW HOST ENVIRONMENT                      │
│                                                                        │
│   ┌─────────────────────┐                 ┌────────────────────────┐   │
│   │   ChakrMicro LM     │                 │   Persistent Memory    │   │
│   │  (Weights / States) │                 │     (Episodic / RIL)   │   │
│   └──────────┬──────────┘                 └───────────▲────────────┘   │
│              │ (token stream)                         │                │
│              ▼                                        │ (sanitized     │
│   ┌─────────────────────┐                             │  experience)   │
│   │    Arena Manager    ├─────────────────────────────┘                │
│   └──────────┬──────────┘                                              │
└──────────────┼─────────────────────────────────────────────────────────┘
               │ (manages lifecycle)
               ▼
┌────────────────────────────────────────────────────────────────────────┐
│                     ISOLATED DISPOSABLE SANDBOX                        │
│                                                                        │
│   arena_scratch/                                                       │
│     └── workspace_<project_id>_<timestamp>/                            │
│           ├── specification/   (project requirements, API contracts)   │
│           ├── source/          (generated / evaluated source files)    │
│           ├── tests/           (isolated test suites & assertions)     │
│           ├── logs/            (stdout, stderr, exit codes, metrics)   │
│           └── artifacts/       (snapshots, diffs, AST parse reports)   │
│                                                                        │
│   [SANDBOX RESTRICTIONS]:                                              │
│   • Subprocess wall-clock timeout: <= 5.0 seconds                      │
│   • Memory ceiling: <= 256 MB                                          │
│   • Environment: Sanitized (zero API keys, zero parent env secrets)    │
│   • Zero write access to chakrview/, tests/, or docs/                  │
└────────────────────────────────────────────────────────────────────────┘
```

### Boundary Definitions:
1. **Model Boundary**: Generates code tokens; has zero filesystem access and zero command execution capabilities.
2. **Arena Manager Boundary**: Validates inputs, creates isolated workspace directories, writes files, initiates execution via sandboxed runner, and collects results.
3. **Execution Runner Boundary**: Runs tests inside a dedicated subprocess with wall-clock timeout and sanitized environment. It cannot import or reference parent ChakrView internals.
4. **Evaluator Boundary**: Parses outputs, computes AST validity, test pass rate, and iteration deltas.
5. **Memory / RIL Bridge**: Consolidates structured execution outcomes into episodic records for future learning.

---

## 3. Directory Layout for Arena Workspaces

Every project managed by the Arena is housed in an isolated workspace:

```
arena_workspace/
    ├── specification/
    │   └── project_spec.json       # Project contract, requirements, interface definition
    ├── source/
    │   ├── __init__.py             # Module root
    │   └── core.py                 # Generated source implementation
    ├── tests/
    │   └── test_core.py            # Automated test suite
    ├── logs/
    │   ├── execution.log           # Full stdout / stderr capture
    │   └── test_results.json       # Structured test counts (passed, failed, errored)
    ├── artifacts/
    │   ├── ast_report.json         # Syntax validity report
    │   └── iteration_summary.json  # Progression history across cycles
    └── evaluation/
        └── score.json              # Aggregated metrics (syntax, tests, latency)
```

---

## 4. Core Arena Subsystem Modules (`chakrview/arena/`)

1. **`models.py`**:
   - `ProjectSpecification`: Formal description of project requirements, language, files, dependencies, and test specifications.
   - `SourceFile`: Represents a single file path, contents, and metadata.
   - `ProjectManifest`: Collection of source and test files belonging to a project.
   - `TestResult`: Structured record of test outcomes (passed, failed, errors, duration, stdout, stderr).
   - `ArenaExecutionResult`: Overall workspace execution summary including timeout, exit code, and failure classification.
   - `EvaluationMetrics`: Combined scoring record (AST syntax validity, test pass rate, latency, memory).

2. **`workspace.py`**:
   - `IsolatedWorkspace`: Manages temporary workspace directory creation, file writing, clean teardown, and artifact preservation.

3. **`executor.py`**:
   - `SandboxedExecutor`: Executes unit tests via `subprocess.run` with:
     - Strict wall-clock timeout ($\le 5.0$ s).
     - Clean `PYTHONPATH` restricted strictly to the workspace `source` directory.
     - Sanitized environment variables (excluding sensitive host credentials).
     - Full stdout/stderr capture and exit code parsing.

4. **`evaluator.py`**:
   - `ArenaEvaluator`: Computes objective metrics:
     - `validate_syntax(code, language)`: Python AST parsing to verify syntax validity without execution.
     - `evaluate_project(workspace)`: Orchestrates execution and aggregates metrics.
     - `classify_failure(result)`: Categorizes failures (SYNTAX_ERROR, IMPORT_ERROR, ASSERTION_FAILURE, TIMEOUT, RUNTIME_EXCEPTION).

5. **`dataset.py`**:
   - `CodingCorpusManager`: Implements project-level split isolation (80/10/10), deduplication (exact SHA-256 and lexical Jaccard), and validation checks.

---

## 5. Security & Safety Invariants

1. **Fail-Closed Execution**: If a test subprocess times out, hangs, or exceeds resource limits, the execution is terminated immediately and marked `TIMEOUT_EXCEEDED`.
2. **Host Immutability**: The Arena executor has zero permissions to alter files outside its designated scratch workspace.
3. **Model Weight Immutability**: During evaluation in the Arena, model weights are strictly frozen ($\Delta W = 0$).
4. **Deterministic Evaluation**: Given the same seed, project spec, and model outputs, the Arena produces identical execution records and scores.
