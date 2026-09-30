# ChakrView Step 53: Trajectory Data Contract for Future RIL Integration

- **Document Version**: 1.0.0
- **Status**: Ratified Data Specification (Foundation for RIL)
- **Scope**: Interaction Trajectory Encoding

---

## 1. Architectural Purpose

ChakrView's long-term intelligence loop is recursive:
$$\text{ACTION} \longrightarrow \text{CONSEQUENCE} \longrightarrow \text{OBSERVATION} \longrightarrow \text{CORRECTION}$$

This feedback loop is central to RIL (Recursive Intelligence Loop). While RIL is **not** implemented in Step 53, establishing a standardized, serialized trajectory format ensures that current training datasets and Project Arena execution records can be consumed by future RIL iterations without architectural friction.

---

## 2. Trajectory Schema Specification

A complete interaction trajectory consists of an ordered sequence of cognitive steps:

```json
{
  "trajectory_id": "traj_20260930_math_clamp_001",
  "version": "1.0.0",
  "domain": "software_engineering",
  "specification": {
    "goal": "Implement clamp(val, min_val, max_val)",
    "requirements": ["Returns min_val if val < min_val", "Returns max_val if val > max_val", "Returns val otherwise"],
    "language": "python"
  },
  "steps": [
    {
      "step_idx": 1,
      "step_type": "INITIAL_SYNTHESIS",
      "action": {
        "type": "WRITE_FILE",
        "path": "core.py",
        "content": "def clamp(val, min_val, max_val):\n    if val < min_val: return min_val\n    return max_val  # Defect: ignores intermediate values\n"
      },
      "observation": {
        "type": "TEST_EXECUTION",
        "command": "pytest test_core.py",
        "exit_code": 1,
        "stdout": "FAILED test_core.py::test_clamp_mid - assert clamp(5, 0, 10) == 5\nwhere clamp(5, 0, 10) returned 10",
        "passed": 2,
        "failed": 1
      },
      "diagnosis": {
        "failing_test": "test_clamp_mid",
        "defect_type": "LOGICAL_DEFECT",
        "description": "Function unconditionally returns max_val when val >= min_val without checking if val <= max_val."
      }
    },
    {
      "step_idx": 2,
      "step_type": "REPAIR_PATCH",
      "action": {
        "type": "APPLY_PATCH",
        "path": "core.py",
        "diff": "--- a/core.py\n+++ b/core.py\n@@ -2,2 +2,3 @@\n     if val < min_val: return min_val\n+    if val > max_val: return max_val\n-    return max_val\n+    return val\n",
        "new_content": "def clamp(val, min_val, max_val):\n    if val < min_val: return min_val\n    if val > max_val: return max_val\n    return val\n"
      },
      "observation": {
        "type": "TEST_EXECUTION",
        "command": "pytest test_core.py",
        "exit_code": 0,
        "stdout": "3 passed in 0.02s",
        "passed": 3,
        "failed": 0
      },
      "diagnosis": null
    }
  ],
  "result": {
    "converged": true,
    "total_iterations": 2,
    "final_state": "SUCCESS"
  }
}
```

---

## 3. Training Representation (Prompt / Target Serialization)

When tokenized for autoregressive training, trajectories are formatted as canonical, self-delimited text blocks:

```text
<TRAJECTORY>
<SPEC>
Goal: Implement clamp(val, min_val, max_val)
Requirements: Clamp val between min_val and max_val
</SPEC>
<STEP 1>
<ACTION>
def clamp(val, min_val, max_val):
    if val < min_val: return min_val
    return max_val
</ACTION>
<OBSERVATION>
FAILED test_clamp_mid: clamp(5, 0, 10) returned 10, expected 5
</OBSERVATION>
<DIAGNOSIS>
Need to check if val > max_val before returning val.
</DIAGNOSIS>
<PATCH>
def clamp(val, min_val, max_val):
    if val < min_val: return min_val
    if val > max_val: return max_val
    return val
</PATCH>
<OBSERVATION>
PASSED all tests
</OBSERVATION>
<RESULT>SUCCESS</RESULT>
</TRAJECTORY>
```

This format exposes the model to the causal sequence of defect diagnosis and repair without requiring unconstrained runtime code execution during pretraining.
