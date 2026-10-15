# Neural Decomposition Contract: Batches 701–740
## Neural Subtask Decomposition, Generalization & Learnability Investigation

**Version:** 1.0.0  
**Status:** FROZEN  
**Contract Date:** 2026-10-15  
**Governing Module:** `chakrview/cognition/neural_subtask_planner.py`  

---

### 1. Specification & Target Representation

Rather than emitting brittle, variable-length raw JSON with high token serialization overhead, the neural decomposition head predicts a compact structured plan encoding:

1. **Goal Intent Category:**
   - Classification over: `compound_arithmetic`, `text_transform_chain`, `evidence_reasoning`, `clarification_needed`, `unsupported`.
2. **Subtask Action Sequence:**
   - Fixed-horizon subtask prediction (up to 3 subtasks per prompt).
   - Each subtask position predicts:
     - Action type (`tool_call`, `text_transform`, `clarification`, `none`).
     - Target tool (`calculator`, `text_utility`, `none`).
3. **Dependency Edges:**
   - Binary adjacency matrix / dependency link for each subtask $i$: whether subtask $i$ depends on output of subtask $j$ ($j < i$).
4. **Clarification Need:**
   - Binary confidence head indicating whether the instruction is missing required inputs.

---

### 2. Canonical Serialization Format

```
Goal: <goal_category>
Subtask 1: <action_type> | <tool_name> | deps: none
Subtask 2: <action_type> | <tool_name> | deps: [st_01]
Clarification: <yes|no>
```

---

### 3. Deterministic Graph Validation Rules

A predicted plan is passed to the deterministic validator (`TaskGraphValidator`) which verifies:
1. **Acyclicity:** All dependencies must satisfy $dep\_id < current\_id$. Backward or self-dependencies trigger immediate validation rejection.
2. **Legal Tool Usage:** Tool calls must map to registered runtime tools (`calculator`, `text_utility`).
3. **Input Binding Validity:** Any subtask with a dependency must have valid upstream output types matching the required tool input.
4. **No Silent Gold Fallback:** If validation fails, the plan is marked `FAILED` or `AWAITING_CLARIFICATION`. The validator **must never** substitute gold fixtures.
