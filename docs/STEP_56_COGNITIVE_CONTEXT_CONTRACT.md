# ChakrView Step 56: Cognitive Context Contract Specification

- **Document Version**: 1.0.0
- **Status**: Ratified Contract Specification
- **Subject**: Bounded, Structured Cognitive Context Schema (<CORTEX_CONTEXT>)

---

## 1. Schema Definition

To allow external memory, task mode, goals, and diagnostic states to condition `ChakrMicro` without polluting raw instructions, the canonical XML context envelope is formally defined:

```xml
<CORTEX_CONTEXT>
<TASK_MODE>{task_mode}</TASK_MODE>
<GOAL>{goal}</GOAL>
<STATE>{state}</STATE>
<MEMORY>{memory_context}</MEMORY>
<REASONING_STATE>{reasoning_state}</REASONING_STATE>
<CONSTRAINTS>{constraints}</CONSTRAINTS>
</CORTEX_CONTEXT>
```

---

## 2. Element Semantics

| Tag | Purpose | Constraints | Example Value |
| :--- | :--- | :--- | :--- |
| `<TASK_MODE>` | Informs the model of the active cognitive role | 1 to 2 words | `REASONING`, `SYNTAX_REPAIR`, `EXECUTION` |
| `<GOAL>` | Precise target statement | Max 20 words | `Fix off-by-one error in add function` |
| `<STATE>` | Current environment / execution state | Max 20 words | `Code failed unit assertion test_add` |
| `<MEMORY>` | Relevant factual or episodic recall | Max 40 words | `Previous observation: add returned a - b instead of a + b` |
| `<REASONING_STATE>`| Diagnostic finding or hypothesis | Max 30 words | `Diagnosis: operator mismatch; use '+' instead of '-'` |
| `<CONSTRAINTS>` | Output format constraints | Max 20 words | `Return single line valid Python AST` |

---

## 3. Strict Token Budget Allocations

Under ChakrMicro's 512-token context window:
- **Maximum Cortex Context Size**: $\le 192$ tokens.
- **Maximum Task Prompt Size**: $\le 192$ tokens.
- **Reserved Output Horizon**: $\ge 64$ tokens.
- **Safety Cushion**: $64$ tokens.

Total: $192 + 192 + 64 + 64 = 512$ tokens.

---

## 4. Invariants

1. **Deterministic Serialization**: Elements are always serialized in the exact canonical tag order.
2. **Secret-Scanned**: Context strings undergo prohibited keyword checks (`private_key`, `secret`, `auth_token`).
3. **Passive Context**: The context envelope is treated strictly as passive prompt conditioning data; it never executes commands directly.
