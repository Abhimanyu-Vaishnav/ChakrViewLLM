# ChakrView Step 54: Action / Observation Trajectory Specification

- **Document Version**: 2.0.0
- **Status**: Ratified Action/Observation Trajectory Format (Step 54)
- **Scope**: Canonical Trajectory Serialization for ChakrMicro Context Window ($T \le 512$)

---

## 1. Canonical Trajectory Structure

Every causal action/observation trajectory adheres to the following canonical XML-delimited schema:

```text
<TRAJECTORY>
<SPEC>
[Task statement, inputs, goals, and constraints]
</SPEC>

<STATE>
[Initial environment or system state]
</STATE>

<ACTION>
[Action taken: code generation, arithmetic calculation, state modification]
</ACTION>

<OBSERVATION>
[Outcome observed: test result, compiler output, runtime exception, or success confirmation]
</OBSERVATION>

<DIAGNOSIS>
[Analysis of defect or discrepancy, or 'None' if action succeeded]
</DIAGNOSIS>

<NEXT_ACTION>
[Corrective patch, revised step, or terminal verification]
</NEXT_ACTION>

<RESULT>
[SUCCESS or FAILURE]
</RESULT>
</TRAJECTORY>
```

---

## 2. Standardized Trajectory Archetypes

### Archetype 1: Arithmetic Multi-Step Trajectory
```text
<TRAJECTORY>
<SPEC>
Compute total price: 4 items at $6 each with a $5 discount coupon.
</SPEC>
<STATE>
Items = 4, UnitPrice = 6, Discount = 5
</STATE>
<ACTION>
Calculate subtotal: 4 * 6 = 24. Apply discount: 24 - 5 = 19.
</ACTION>
<OBSERVATION>
Verification check: 24 - 5 equals 19.
</OBSERVATION>
<DIAGNOSIS>
None. Subtotal and discount applied in correct order.
</DIAGNOSIS>
<NEXT_ACTION>
Return final answer: 19
</NEXT_ACTION>
<RESULT>
SUCCESS
</RESULT>
</TRAJECTORY>
```

### Archetype 2: Code Bug Diagnosis & Repair Trajectory
```text
<TRAJECTORY>
<SPEC>
Implement absolute_value(x). Return x if x >= 0, else -x.
</SPEC>
<STATE>
Empty function stub in core.py.
</STATE>
<ACTION>
def absolute_value(x: float) -> float:
    return x  # Bug: misses negative numbers
</ACTION>
<OBSERVATION>
FAILED test_abs_neg: absolute_value(-5.0) returned -5.0, expected 5.0
</OBSERVATION>
<DIAGNOSIS>
Function fails to negate negative inputs when x < 0.
</DIAGNOSIS>
<NEXT_ACTION>
def absolute_value(x: float) -> float:
    if x < 0:
        return -x
    return x
</NEXT_ACTION>
<RESULT>
SUCCESS
</RESULT>
</TRAJECTORY>
```

### Archetype 3: Deterministic Planning Trajectory
```text
<TRAJECTORY>
<SPEC>
Plan data ingestion pipeline: fetch, validate, transform, store.
</SPEC>
<STATE>
Raw unvalidated payload arriving on queue.
</STATE>
<ACTION>
Step 1: Parse payload JSON.
Step 2: Validate required fields against schema.
Step 3: Normalize text to UTF-8.
Step 4: Write to database.
</ACTION>
<OBSERVATION>
All 4 steps form an acyclic sequence with clear dependencies.
</OBSERVATION>
<DIAGNOSIS>
None. Order preserves validation prior to database write.
</DIAGNOSIS>
<NEXT_ACTION>
Emit ingestion execution plan.
</NEXT_ACTION>
<RESULT>
SUCCESS
</RESULT>
</TRAJECTORY>
```
