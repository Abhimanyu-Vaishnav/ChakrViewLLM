# ChakrKshetra: Execution, Experimentation, Observation & Evaluation Environment

- **Document Version**: 2.0.0
- **Status**: Ratified Architectural Specification
- **Former Name**: Project Arena (migrated in Step 54)
- **Subsystem Package**: `chakrview/arena/` (preserved for backward compatibility, conceptually ChakrKshetra)

---

## 1. Foundational Axiom

> **ChakrKshetra is ChakrView's controlled execution, experimentation, observation, and evaluation environment. ChakrKshetra is NOT the neural brain.**

ChakrKshetra provides a safe laboratory for the brain to interact with external tools, execute code, observe real runtime consequences, collect error diagnostics, and record episodic trajectories. It is an external environment, strictly decoupled from the neural parameters:
$$\text{CHAKRVIEW NEURAL BRAIN} \neq \text{CHAKRKSHETRA} \neq \text{MEMORY} \neq \text{RIL} \neq \text{EVALUATOR} \neq \text{HOST APPLICATION}$$

---

## 2. Core Responsibilities of ChakrKshetra

1. **Workspace Confinement**:
   - Creates isolated, temporary directory trees for each experimental task.
   - Enforces strict path traversal guards (`PathTraversalError` on any attempt to escape via `../` or symlinks).
   - Enforces disk and file count quotas (max 50 files, max 10MB; `WorkspaceQuotaExceededError`).
2. **Controlled Subprocess Execution**:
   - Executes untrusted code inside child Python processes with sanitized environments.
   - Enforces fail-closed timeout traps (process termination on hanging loops).
   - Scrubs all host secrets, credentials, tokens, and model checkpoint paths from child environment variables.
3. **Observation & Diagnostics Extraction**:
   - Captures stdout, stderr, and exit codes.
   - Automatically parses stack traces to extract failing test names, line numbers, exception types, and traceback frames.
4. **Patch & Diff History Tracking**:
   - Records all file modifications as unified diffs (`PatchDiff`) in `artifacts/diffs/`.
   - Maintains multi-turn iteration records (`IterationRecord`, `ExecutionHistory`).
5. **Deterministic Reset & Teardown**:
   - Supports clean rollback (`reset_to_initial()`) to initial project manifests without cross-test leakage.
   - Recursive cleanup on session completion.
6. **Episodic Memory Bridge (RIL Preparation)**:
   - Serializes complete task trajectories (`SPEC -> ACTION -> OBSERVATION -> DIAGNOSIS -> PATCH -> RESULT`) into structured experience records.

---

## 3. Directory Layout in ChakrKshetra

```
chakrkshetra_workspace/
├── project_manifest.json          # Root project manifest with metadata & file checksums
├── specification/
│   └── project_spec.json          # Task requirements, constraints, and test framework
├── source/
│   ├── __init__.py                # Package initialization
│   └── [source files]             # Generated or target code
├── tests/
│   └── test_[module].py           # Verification assertions & unit tests
├── runtime/
│   └── [configs/entrypoints]      # Runtime configuration
├── logs/
│   ├── exec_[run_id].log          # Stdout and stderr logs
│   └── pytest.log                 # Detailed pytest execution trace
└── artifacts/
    ├── diffs/
    │   └── patch_[iter]_[file].diff # Unified diff patch files
    └── evaluation_report.json     # Final scoring and metrics
```

---

## 4. Security Boundaries & Limitations (Honest Accounting)

1. **Process-Level Isolation**:
   - Execution isolation is currently enforced via `subprocess.run(timeout=...)` with environment scrubbing.
   - **Limitation**: This provides application-level containment. It does **not** implement kernel-level Linux cgroups, seccomp, Windows Job Objects memory caps, or hypervisor micro-VM isolation. Stronger platform-specific isolation remains a planned future layer.
2. **Zero Host Execution**:
   - Generated code is never executed via `eval()`, `exec()`, or dynamic imports inside the ChakrView process.
3. **No ChakrView Source Access**:
   - Child processes have no access to ChakrView's repository, baseline checkpoints, or development environment.
