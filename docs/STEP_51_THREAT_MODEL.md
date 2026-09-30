# ChakrView Step 51: Project Arena & Code Execution Threat Model

- **Version**: 1.0.0
- **Scope**: Step 51 — Project-Based Coding Capability Acquisition & Project Arena Foundation
- **Target**: Security, Sandboxing, Isolation, and Invariant Preservation
- **Status**: Formally Specified & Audited

---

## 1. Threat Modeling Overview & Core Posture

Executing model-generated source code introduces severe security vectors that do not exist during natural language generation. If unconstrained, generated code can crash the host Python interpreter, exhaust CPU or memory resources, escape into the host filesystem, exfiltrate environment secrets, or accidentally overwrite the ChakrView repository and model weights.

ChakrView adopts an uncompromising **Fail-Closed Default-Deny Security Posture**:
- Generated code is treated as **untrusted, hostile bytecode**.
- No generated code is ever evaluated or imported within the host ChakrView Python process.
- All execution is isolated inside disposable filesystem sandboxes with enforced wall-clock timeouts, memory ceilings, and sanitized environments.
- Model weights and baseline checksums are protected by hard programmatic immutability guards ($\Delta W = 0$).

---

## 2. Threat Vector Enumeration & Defenses

| Threat ID | Threat Vector | Risk Level | Attack Mechanism | Implemented Defense-in-Depth |
|:---|:---|:---:|:---|:---|
| **T-01** | **Host Process Execution / Code Injection** | **CRITICAL** | Model outputs `os.system("rmdir /s")` or `__import__('os').remove(...)` executed via `eval()` or `exec()`. | **Zero In-Process Execution**: `eval()` and `exec()` are strictly forbidden. All code executes inside isolated OS subprocesses managed by `SandboxedExecutor`. |
| **T-02** | **Filesystem Traversal & Escape** | **HIGH** | Code attempts path traversal (`../../chakrview/model.py`) to alter host repository or system files. | **Path Confinement**: All file I/O is resolved strictly inside `workspace.source_dir` or `test_dir`. Path resolution checks reject any path outside the workspace boundary. |
| **T-03** | **CPU Pegging & Infinite Loops** | **HIGH** | Generated code contains `while True: pass` or non-terminating recursive loops that hang the CPU. | **Fail-Closed Subprocess Timeout**: `subprocess.run(timeout=5.0)` forcefully kills the child process tree on timeout expiry. |
| **T-04** | **Memory Exhaustion / Fork Bomb** | **HIGH** | Code allocates giant arrays (`[0] * 10**9`) or spawns recursive processes (`os.fork()`). | **Resource Ceilings**: Subprocess RSS capped at $\le 256\text{ MB}$; process spawning restricted to single-threaded test execution. |
| **T-05** | **Disk Flooding & Storage DoS** | **MEDIUM** | Code writes multi-gigabyte log or binary files inside the sandbox. | **Disk Ceilings & Ephemeral Teardown**: Workspace disk usage capped at $10\text{ MB}$; workspaces are automatically destroyed on evaluation completion. |
| **T-06** | **Credential & Secret Exfiltration** | **HIGH** | Code scans `os.environ` to inspect API keys, tokens, or system passwords. | **Sanitized Environment**: Child subprocess receives a stripped environment dictionary containing only essential OS keys (`PATH`, `SYSTEMROOT`). All tokens and secrets are purged. |
| **T-07** | **Host ChakrView Module Pollution** | **HIGH** | Code imports `chakrview` to mutate runtime settings, model weights, or memory stores. | **Restricted PYTHONPATH**: Subprocess `PYTHONPATH` points strictly to the workspace `source/` directory. `chakrview.*` is completely unimportable from within the sandbox. |
| **T-08** | **Premature Self-Modification Escape** | **CRITICAL** | Model attempts to edit ChakrView's own architecture or training code before reaching verified maturity. | **Read-Only Codebase Invariant**: The ChakrView repository (`chakrview/`, `tests/`, `docs/`) is strictly read-only during all Arena runs. Model self-editing is architecturally forbidden in Step 51. |
| **T-09** | **Neural Core Weight Tampering** | **CRITICAL** | Training loop or interactive session accidentally overwrites the frozen baseline model. | **Frozen Baseline Hash Guard**: Baseline model SHA-256 (`c5571c...`) is verified before and after every operation. Any weight mutation raises an immediate assertion error ($\Delta W_{\text{baseline}} = 0$). |
| **T-10** | **Training Data Cross-Contamination** | **MEDIUM** | Files from Project $X$ appear in both train and validation splits, causing false generalization claims. | **Project-Level Hash Partitioning**: `CodingCorpusManager` hashes the `project_id` to enforce strict project-level split isolation. Zero file-level leakage. |

---

## 3. Sandboxing Architecture Diagram

```
┌────────────────────────────────────────────────────────────────────────┐
│                        CHAKRVIEW HOST ENVIRONMENT                      │
│                                                                        │
│   • Frozen Baseline Weights (SHA-256: c5571c9c... [LOCKED])           │
│   • ChakrView Source Code (chakrview/, tests/ [READ-ONLY])            │
│   • Active Host Process (PID: Parent)                                 │
│                                                                        │
│   [SECURITY GATE]:                                                     │
│   - Zero eval() / exec()                                               │
│   - Stripped os.environ (zero secrets/tokens passed)                   │
│   - Locked PYTHONPATH (workspace/source only)                          │
└────────────────────────────────────┬───────────────────────────────────┘
                                     │ (spawns isolated worker)
                                     ▼
┌────────────────────────────────────────────────────────────────────────┐
│                     ISOLATED SUBPROCESS SANDBOX                        │
│                                                                        │
│   Process: Python Subprocess (PID: Child)                              │
│   Working Dir: scratch/arena/workspace_<project_id>_<timestamp>/       │
│                                                                        │
│   [HARD CEILINGS]:                                                     │
│   ├── Timeout Ceiling: <= 5.0 seconds (SIGKILL on expiry)             │
│   ├── Memory Ceiling: <= 256 MB RSS                                   │
│   ├── Storage Ceiling: <= 10 MB                                       │
│   ├── Network: Local loopback only (zero outbound sockets)            │
│   └── Import Boundary: Standard library + local source/*.py only       │
│                                                                        │
│   [TRAPPED STREAMS]:                                                   │
│   ├── STDOUT -> Captured into logs/execution.log                       │
│   └── STDERR -> Parsed for SyntaxError / AssertionError / Tracebacks   │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Invariant Verification Matrix

| Invariant Name | Mathematical Definition | Verification Method | Enforcement Status |
|:---|:---|:---|:---:|
| **Baseline Immutability** | $\Delta W_{\text{baseline}} \equiv 0$ | SHA-256 pre/post forward check | **ENFORCED** (`c5571c...`) |
| **Parameter Invariant** | $|\Theta| = 3,443,136$ | Tensor element counting | **ENFORCED** |
| **Vocabulary Invariant** | $|V| = 4,096$ | Tokenizer embedding bounds | **ENFORCED** |
| **Context Ceiling** | $T \le 512$ | Assertion guard in attention mask | **ENFORCED** |
| **Subprocess Timeout** | $t_{\text{exec}} \le 5.0\text{ s}$ | `subprocess.TimeoutExpired` trap | **ENFORCED** |
| **Workspace Containment**| $\text{path} \subseteq \text{workspace\_dir}$ | Relative path canonicalization | **ENFORCED** |
| **Zero GPU Dependency** | $\text{device} = \text{"cpu"}$ | Native CPU device verification | **ENFORCED** |
