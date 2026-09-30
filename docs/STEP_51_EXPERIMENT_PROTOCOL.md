# ChakrView Step 51: Closed-Loop Experiment Protocol

- **Protocol Version**: 1.0.0
- **Status**: Verified Operational Protocol
- **Component**: `chakrview/arena/loop.py`, `chakrview/arena/memory.py`

---

## 1. Closed-Loop Execution Protocol

The closed-loop controller executes the real-world feedback cycle:
$$\text{Project Spec} \to \text{Generation} \to \text{Execution} \to \text{Observation} \to \text{Diagnosis} \to \text{Repair} \to \text{Retest} \to \text{Memory}$$

```
                +----------------------------+
                |   ProjectSpecification     |
                +-------------+--------------+
                              |
                              v
                +----------------------------+
                |   Format Initial Prompt    |
                +-------------+--------------+
                              |
                              v
                +----------------------------+
                | Model Generates Source     |
                +-------------+--------------+
                              |
                              v
                +----------------------------+
                | IsolatedWorkspace Material |
                +-------------+--------------+
                              |
+---------------------------->+
|                             |
|                             v
|               +----------------------------+
|               |  SandboxedExecutor Tests   |
|               +-------------+--------------+
|                             |
|                      Pass?  |
|                     /      \
|                   Yes       No
|                   /          \
|                  v            v
|          +------------+  +-------------------------------+
|          | CONVERGED  |  | Extract Traceback Diagnostic  |
|          +-----+------+  +---------------+---------------+
|                |                         |
|                |        Iteration < Max? |
|                |       /                 \
|                |     Yes                  No
|                |     /                     \
|                |    v                       v
|                |  +--------------------+  +------------+
|                |  | Format Repair Promp|  | EXHAUSTED  |
|                |  +---------+----------+  +-----+------+
|                |            |                   |
|                |            v                   |
|                |  +--------------------+        |
|                |  | Apply Patch & Diff |        |
|                |  +---------+----------+        |
|                |            |                   |
|                +------------+-------------------+
|                             |
|                             v
|               +----------------------------+
|               |  ArenaMemoryBridge (RIL)   |
+---------------+----------------------------+
```

---

## 2. Multi-Turn Repair Algorithm

1. **Iteration Initialization ($k = 1$)**:
   - Materialize workspace from manifest.
   - Run tests via `SandboxedExecutor.run_tests(ws)`.
   - Record test counts (`passed`, `failed`, `errors`).
2. **Convergence Evaluation**:
   - If `failed == 0` and `errors == 0`: set `history.converged = True`, record `IterationRecord`, break loop.
3. **Diagnostic Extraction**:
   - If tests fail, run `extract_traceback_diagnostic(test_res)`.
   - Parse failing test name, target source file, line number, exception type, and traceback snippet.
4. **Repair Generation ($k < K_{\text{max}}$)**:
   - Identify source file under test (excluding test files).
   - Format repair prompt with current code and error traceback.
   - Invoke repair generator (model or agent).
   - If generator returns identical or empty code: terminate loop (stalled).
5. **Patch Application & Tracking**:
   - Call `ws.apply_patch(target_file, new_code)`.
   - Record unified diff artifact in `artifacts/diffs/`.
   - Append `PatchDiff` and `IterationRecord` to `ExecutionHistory`.
   - Advance iteration $k \leftarrow k + 1$ and repeat from step 2.

---

## 3. Diagnostic Extraction Format

The diagnostic parser produces structured dictionaries:
```python
{
    "failing_test": "test_multiply",
    "target_file": "core.py",
    "target_line": 2,
    "exception_type": "AssertionError",
    "error_message": "assert 13 == 12",
    "raw_snippet": "tests/test_core.py:4: AssertionError\nassert 13 == 12"
}
```

---

## 4. Episodic Memory Record Schema (RIL Bridge)

Upon session completion, `ArenaMemoryBridge.record_experience` serializes the trajectory into an episodic record:
```json
{
  "episode_id": "ep_arena_proj_buggy_math_01_1727712000000",
  "project_id": "proj_buggy_math_01",
  "timestamp": 1727712000.0,
  "specification": { ... },
  "converged": true,
  "total_iterations": 2,
  "final_pass_rate": 1.0,
  "initial_failure_category": "ASSERTION_FAILURE",
  "total_patches_applied": 1,
  "iterations": [
    {
      "iteration": 1,
      "passed": 0,
      "failed": 1,
      "errors": 0,
      "failure_category": "ASSERTION_FAILURE",
      "diagnosis": "assert 13 == 12",
      "patches": []
    },
    {
      "iteration": 2,
      "passed": 1,
      "failed": 0,
      "errors": 0,
      "failure_category": "SUCCESS",
      "diagnosis": null,
      "patches": [
        {
          "path": "core.py",
          "diff": "--- a/core.py\n+++ b/core.py\n@@ -1,2 +1,2 @@\n def multiply(a: int, b: int) -> int:\n-    return a * b + 1\n+    return a * b\n"
        }
      ]
    }
  ]
}
```

---

## 5. Curriculum Progression & Gating Protocol

No level may be marked complete without meeting its empirical gating criteria:

| Level | Name | Gating Benchmark | Pass Threshold | Current Status |
|:---|:---|:---|:---|:---|
| **Level 0** | Basic Syntax & Tokens | AST parse validity across 50 prompt completions | $\ge 90\%$ valid AST | Infrastructure ready; model unverified |
| **Level 1** | Single-File Utilities | Pass rate on basic arithmetic/string tasks | $\ge 60\%$ first-pass | Infrastructure ready; model unverified |
| **Level 2** | Multi-File Packages | Correct inter-module import & resolution | $\ge 50\%$ pass | Infrastructure ready; model unverified |
| **Level 3** | Projects with Tests | Passing assertions on unit test suites | $\ge 50\%$ pass | Infrastructure ready; model unverified |
| **Level 4** | Bug Fixing & Patching | Multi-turn closed-loop repair convergence | $\ge 40\%$ convergence | Controller verified with mock; model unverified |
| **Level 5+**| Complex Systems | Multi-module application synthesis | Empirical evidence required | Future milestone |

---

## 6. Resource & Immutability Invariants

1. **Baseline Immutability**:
   - Baseline checkpoint hash must equal `c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da`.
   - $\Delta W_{\text{baseline}} = 0$ must be verified at every test run.
2. **CPU-First Execution**:
   - Zero CUDA calls (`torch.cuda.is_available()` returns False or is not utilized).
   - Peak RSS $\le 256$ MB during test runs.
