# ChakrView Step 51: Evaluation Protocol & Real-World Feedback Loop

- **Version**: 1.0.0
- **Scope**: Step 51 — Project-Based Coding Capability Acquisition & Project Arena Foundation
- **Target**: Objective, Verifiable Multi-Metric Evaluation Framework
- **Status**: Formally Specified

---

## 1. Core Evaluation Philosophy: Real-World Grounding

A generated software project must **never** be considered successful simply because the code looks plausible, has no syntax errors, or achieves low perplexity. In software engineering, code is successful if and only if:
1. It parses cleanly into a valid syntax tree.
2. It executes without crashing.
3. It passes all functional test assertions.
4. It satisfies the behavioral specifications without introducing regressions.

ChakrView evaluates code through an active **Real-World Feedback Loop**:

```
Requirement -> Planning -> Code Generation -> Execution -> Tests ->
Failure Observation -> Diagnosis -> Repair -> Re-test -> Objective Evaluation
```

---

## 2. Tripartite Quality Separation

To maintain scientific rigor, ChakrView enforces a strict three-tier separation of concerns:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        TRIPARTITE EVALUATION FRAMEWORK                 │
├─────────────────────────┬──────────────────────┬───────────────────────┤
│    1. MODEL QUALITY     │  2. ENGINE QUALITY   │   3. ARENA QUALITY    │
├─────────────────────────┼──────────────────────┼───────────────────────┤
│ • Cross-Entropy Loss    │ • KV Cache Validity  │ • Workspace Isolation │
│ • Perplexity (PPL)      │ • Sampling Control   │ • Timeout Enforcement │
│ • Token Distribution    │ • Forward Latency    │ • Subprocess Sandbox  │
│ • Syntax AST Validity   │ • TTFT (Time to 1st) │ • Stdout/Stderr Trap  │
│ • Functional Pass Rate  │ • Memory Peak (RSS)  │ • Failure Diagnosis   │
└─────────────────────────┴──────────────────────┴───────────────────────┘
```

---

## 3. The 12 Real-World Software Engineering Metrics

ChakrView defines 12 concrete, programmatically computable metrics for evaluating project-based coding capability:

### 1. Task Completion Rate ($R_{\text{complete}}$)
$$\text{Task Completion Rate} = \frac{N_{\text{fully\_passing\_projects}}}{N_{\text{total\_attempted\_projects}}}$$
Measures the proportion of projects where all specification requirements and test assertions are completely satisfied.

### 2. First-Pass Success Rate ($R_{\text{first\_pass}}$)
$$\text{First-Pass Success Rate} = \frac{N_{\text{passed\_iteration\_1}}}{N_{\text{total\_attempted\_projects}}}$$
Measures whether the model synthesized a fully working project on its initial attempt without requiring iterative debugging.

### 3. Test Pass Percentage ($P_{\text{tests}}$)
$$P_{\text{tests}} = \frac{1}{M} \sum_{j=1}^M \left( \frac{N_{\text{passed\_tests}}^{(j)}}{N_{\text{total\_tests}}^{(j)}} \right)$$
The aggregate percentage of unit test assertions passed across all executed test suites.

### 4. Repair Success Rate ($R_{\text{repair}}$)
$$R_{\text{repair}} = \frac{N_{\text{repaired\_after\_failure}}}{N_{\text{initially\_failed\_projects}}}$$
Measures the system's ability to take error tracebacks from failing tests, formulate a patch, and achieve passing tests.

### 5. Regression Rate ($R_{\text{regress}}$)
$$R_{\text{regress}} = \frac{N_{\text{previously\_passing\_tests\_broken\_by\_patch}}}{N_{\text{total\_previously\_passing\_tests}}}$$
Monitors whether repair attempts inadvertently break previously functioning features ($0.0\%$ target).

### 6. Iterations Required ($\bar{K}_{\text{iter}}$)
$$\bar{K}_{\text{iter}} = \frac{1}{N_{\text{successful\_projects}}} \sum_{i=1}^{N_{\text{successful}}} K_i$$
The average number of repair iterations ($1 \le K \le 5$) required to reach all-passing tests. Lower values indicate superior diagnostic precision.

### 7. Token Efficiency ($E_{\text{token}}$)
$$E_{\text{token}} = \frac{\text{Lines of Passing Code Synthesized}}{\text{Total Tokens Generated (Prompt + Output)}}$$
Measures verbosity versus density. Penalizes bloated or repetitive boilerplate generation.

### 8. Execution Efficiency ($E_{\text{exec}}$)
$$E_{\text{exec}} = \frac{T_{\text{baseline\_duration}}}{T_{\text{model\_code\_duration}}}$$
Compares the CPU execution runtime of model-generated code against reference implementations to ensure generated algorithms do not introduce exponential slowdowns.

### 9. Project Complexity Handled ($C_{\text{proj}}$)
A compound structural score factoring:
$$C_{\text{proj}} = w_1 \cdot N_{\text{files}} + w_2 \cdot N_{\text{functions}} + w_3 \cdot N_{\text{classes}} + w_4 \cdot N_{\text{tests}}$$
Measures the structural scale of projects successfully completed.

### 10. Context Usage Ratio ($U_{\text{ctx}}$)
$$U_{\text{ctx}} = \frac{\text{Tokens Used in Prompt + Generation}}{T_{\text{max}} \quad (512\text{ tokens})}$$
Tracks how efficiently the model utilizes ChakrMicro's strict 512-token context ceiling.

### 11. Memory Recall Precision ($P_{\text{recall}}$)
$$P_{\text{recall}} = \frac{\text{Relevant Episodic Experiences Retrieved}}{\text{Total Experiences Ingested into Context}}$$
Measures whether the RIL memory retriever pulls useful past bug fixes or API signatures into the active prompt.

### 12. Error Diagnosis Accuracy ($A_{\text{diag}}$)
$$A_{\text{diag}} = \frac{N_{\text{correctly\_identified\_defective\_lines}}}{N_{\text{total\_diagnosed\_failures}}}$$
Measures whether the model's repair reasoning accurately localizes the true cause of a test failure rather than making random edits.

---

## 4. Failure Categorization Contract

When a project run fails in the Project Arena, it is deterministically classified into one of six mutually exclusive categories:

| Failure Category | Trigger Condition | Diagnostic Action |
|:---|:---|:---|
| **`SYNTAX_ERROR`** | `ast.parse()` raises `SyntaxError` | Extract line and column; prompt for balanced brackets/syntax. |
| **`IMPORT_ERROR`** | Subprocess raises `ModuleNotFoundError` | Verify local package structure and symbol exports in `__init__.py`. |
| **`ASSERTION_FAILURE`** | Pytest reports assertion violation | Extract expected vs actual values; pass failure frame to repair prompt. |
| **`RUNTIME_ERROR`** | Subprocess raises uncaught runtime exception | Extract exception type (`KeyError`, `IndexError`, `TypeError`) and traceback. |
| **`TIMEOUT`** | Subprocess exceeds wall-clock ceiling ($5.0$s) | Flag potential infinite loop or blocking I/O; terminate process. |
| **`WORKSPACE_ERROR`** | Missing test files or file write failure | Sandbox infrastructure failure; log and abort without penalizing model. |

---

## 5. Objective Scoring Pipeline

```
┌────────────────────────────────────────────────────────────────────────┐
│                        ARENA SCORING PIPELINE                          │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│   ProjectManifest                                                      │
│        │                                                               │
│        ▼                                                               │
│   1. AST Syntax Check   ──[Invalid]──> Score: SyntaxValid=False        │
│        │                               (Halt execution)                │
│        ▼ [Valid]                                                       │
│   2. Sandboxed Subprocess                                              │
│        │                                                               │
│        ├──[Timeout]──────────────────> Score: Timeout=True, PassRate=0 │
│        │                                                               │
│        └──[Exited]                                                     │
│             │                                                          │
│             ▼                                                          │
│        3. Pytest Parser ─────────────> TestResult:                     │
│                                        - Passed, Failed, Errors        │
│                                        - Pass Rate (0.0 to 1.0)        │
│                                        - Duration (seconds)            │
│                                        - Failure Category              │
│                                                                        │
│        4. Artifact Serialization ────> score.json                      │
│                                        ast_report.json                 │
│                                        execution.log                   │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```
