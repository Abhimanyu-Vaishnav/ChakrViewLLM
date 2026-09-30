# ChakrView Step 51: Project Environment & Sandboxed Execution Specification

- **Version**: 1.0.0
- **Scope**: Step 51 — Project-Based Coding Capability Acquisition & Project Arena Foundation
- **Component**: `chakrview/arena/workspace.py`, `executor.py`
- **Execution Target**: Sandboxed Subprocess Isolation on CPU

---

## 1. Project Workspace Directory Layout

Every project constructed or evaluated within the Project Arena operates inside an isolated, disposable directory tree. The workspace layout enforces clear separation of requirements, source code, test suites, execution logs, and evaluation metrics:

```
project_workspace/
├── project_manifest.json       # Top-level manifest (project ID, version, language, dependencies)
├── specification/
│   └── project_spec.json       # Formal requirements, interface signatures, acceptance criteria
├── source/
│   ├── __init__.py             # Package root
│   ├── core.py                 # Primary module implementation
│   └── utils.py                # Helper utilities and data structures
├── tests/
│   ├── conftest.py             # Shared pytest fixtures (mock data, temp paths)
│   └── test_core.py            # Unit tests and invariant assertions
├── runtime/
│   ├── entrypoint.py           # Optional executable entrypoint
│   └── config.json             # Runtime configuration parameters
├── logs/
│   ├── execution.log           # Raw stdout and stderr stream captures
│   └── subprocess.meta         # Process metadata (PID, exit code, CPU time, wall duration)
├── artifacts/
│   ├── ast_report.json         # Abstract Syntax Tree validation report
│   ├── tracebacks.json         # Parsed execution tracebacks and localized error frames
│   └── diffs/                  # Patch diffs generated across repair iterations
└── evaluation/
    ├── test_results.json       # Quantitative test counts (passed, failed, skipped, errors)
    └── score.json              # Aggregated metrics (pass rate, syntax validity, latency)
```

---

## 2. Sandbox Boundary & Security Controls

To ensure absolute system stability, the model and generated code are never executed directly within the host ChakrView Python process. All execution is governed by six strict sandboxing contracts:

### 1. File Creation & Modification Boundary
- **Allowed Scope**: The Arena workspace manager can create, modify, and delete files **only** within its designated `workspace_dir` in `scratch/arena/`.
- **Prohibited Targets**: Any file creation or modification attempt targeting `chakrview/`, `tests/`, `docs/`, `.git/`, or system directories (`C:\Windows`, `/etc`, `/usr`) is strictly blocked.
- **Path Sanitization**: Relative paths are resolved against `source_dir` or `test_dir`; directory traversal attempts (`../`) are trapped and rejected before disk access.

### 2. Command Execution Boundary
- **Allowed Commands**: Only the sandboxed test runner is permitted to spawn processes, restricted to:
  ```bash
  python -m pytest <test_dir> -v --tb=short --no-header
  ```
- **Prohibited Commands**: Arbitrary shell invocation (`cmd.exe`, `powershell.exe`, `bash`), system manipulation (`rmdir /s`, `rm -rf`, `format`), network utilities (`curl`, `wget`, `nc`), and background daemon spawning are strictly forbidden.

### 3. Package & Import Availability
- **Sanitized PYTHONPATH**: The test execution subprocess is passed a `PYTHONPATH` restricted strictly to the workspace `source/` directory:
  ```python
  env["PYTHONPATH"] = str(workspace.source_dir.resolve())
  ```
- **Import Confinement**: Generated code can only import standard Python library modules (e.g. `math`, `typing`, `json`, `collections`, `itertools`, `re`, `dataclasses`) and modules located inside its local `source/` directory.
- **Host Code Quarantine**: Generated code cannot import `chakrview.*`, accessing model weights, memory stores, or tokenizers is architecturally impossible.

### 4. Resource Limits & Ceilings
- **Subprocess Memory Ceiling**: Bounded at $\le 256\text{ MB}$ RSS. Fork bombs or runaway memory allocations are terminated.
- **Thread Count**: Maximum 1 worker process with 0 worker sub-threads.
- **Disk Allocation Ceiling**: Maximum $10\text{ MB}$ per project workspace. File generation is halted if the workspace exceeds this limit.

### 5. Wall-Clock Timeout Handling (Fail-Closed)
- **Timeout Threshold**: Maximum $5.0$ seconds wall-clock per test run (configurable down to $1.0$s for micro-benchmarks).
- **Enforcement Mechanism**:
  ```python
  proc = subprocess.run(cmd, timeout=timeout_seconds, ...)
  ```
- **Timeout Action**: If execution exceeds the allotted duration, a `TimeoutExpired` exception is triggered, the subprocess tree is forcefully terminated (`SIGKILL` / `TerminateProcess`), exit code is recorded as `-1`, and failure category is marked `TIMEOUT`.

### 6. Failure Recovery & Error Trapping
- **Non-Destructive Exception Handling**: All subprocess crashes, uncaught Python exceptions, and memory faults are captured in `logs/execution.log` without terminating the host evaluation session.
- **Structured Error Parsing**:
  - `SyntaxError`: Extracted with line number, column offset, and message.
  - `ImportError` / `ModuleNotFoundError`: Missing module symbol identified.
  - `AssertionError`: Target expression, expected vs actual values extracted.

---

## 3. Separation of Responsibilities

```
┌────────────────────────────────────────────────────────────────────────┐
│                      STRICT SEPARATION OF CONCERNS                     │
├───────────────┬────────────────────────────────────────────────────────┤
│ SUBSYSTEM     │ SOVEREIGN RESPONSIBILITIES & ACCESS BOUNDARY           │
├───────────────┼────────────────────────────────────────────────────────┤
│ MODEL         │ • Receives prompt tokens.                              │
│               │ • Emits generated code tokens.                         │
│               │ • Has ZERO filesystem, network, or command access.     │
├───────────────┼────────────────────────────────────────────────────────┤
│ TOOLS         │ • Abstract capability gates (CapabilityGate).          │
│               │ • Strictly blocked from arbitrary OS execution.        │
├───────────────┼────────────────────────────────────────────────────────┤
│ ARENA         │ • Creates and destroys isolated workspace directories. │
│               │ • Writes source and test files to disk.                │
│               │ • Executes tests via SandboxedExecutor subprocess.     │
│               │ • Captures execution stdout/stderr and exit codes.     │
├───────────────┼────────────────────────────────────────────────────────┤
│ EVALUATOR     │ • Parses AST syntax validity.                          │
│               │ • Aggregates test pass rates and failure modes.        │
│               │ • Emits structured evaluation reports.                 │
├───────────────┼────────────────────────────────────────────────────────┤
│ MEMORY        │ • Ingests execution logs and test results.             │
│               │ • Stores structured episodic experiences for RIL.      │
│               │ • Cannot modify code or execute tests.                 │
├───────────────┼────────────────────────────────────────────────────────┤
│ CHAKRVIEW     │ • The core framework codebase (chakrview/, tests/).    │
│ SOURCE        │ • STRICTLY READ-ONLY across all evaluation phases.     │
└───────────────┴────────────────────────────────────────────────────────┘
```

---

## 4. Teardown & Workspace Retention

- **Ephemeral Mode (Default)**: Upon evaluation completion, the workspace directory is automatically cleaned up and removed via Python's `tempfile.TemporaryDirectory`.
- **Preserved Mode (Debugging / Benchmark)**: When `preserve=True`, the workspace is retained under `artifacts/step51/workspaces/workspace_<project_id>_<timestamp>/` for human inspection, diagnostic auditing, and regression fixture capture.
