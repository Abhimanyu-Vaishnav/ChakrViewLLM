# ChakrView Step 53: Foundation Curriculum Specification

- **Document Version**: 1.0.0
- **Status**: Ratified Curriculum Specification
- **Focus**: Multi-Domain Balanced Learning for ChakrMicro v0.1

---

## 1. Curriculum Architecture

To prevent ChakrView from degenerating into a fragile, single-domain code generator while establishing robust deterministic capabilities, the curriculum is organized into balanced, progressive tiers:

```
+-------------------------------------------------------+
|  LEVEL 1: Controlled Reasoning & Trajectories        |
|  - Multi-step logic, fault diagnosis, patch sequences |
+-------------------------------------------------------+
                           ^
+-------------------------------------------------------+
|  LEVEL 0D: Basic Programming                          |
|  - Functions, return values, if/else, loops, utilities|
+-------------------------------------------------------+
                           ^
+-------------------------------------------------------+
|  LEVEL 0C: Basic Computation                          |
|  - Arithmetic, boolean comparisons, string transforms |
+-------------------------------------------------------+
                           ^
+-------------------------------------------------------+
|  LEVEL 0B: Basic Language & Structured Data           |
|  - Factual statements, JSON, YAML, key-value mappings |
+-------------------------------------------------------+
                           ^
+-------------------------------------------------------+
|  LEVEL 0A: Token & Sequence Stability                 |
|  - Brackets, punctuation, identifiers, whitespace      |
+-------------------------------------------------------+
```

---

## 2. Detailed Level Definitions

### Level 0A — Token & Sequence Stability
- **Objective**: Anchor token transition probabilities for structural delimiters, preventing repetition loops on punctuation.
- **Components**:
  - Matched pairs: `()`, `[]`, `{}`, `""`, `''`.
  - Python indentation: 4-space indent patterns.
  - Operators: `=`, `==`, `!=`, `+`, `-`, `*`, `/`, `%`.
  - Whitespace preservation and clean newline terminations.

### Level 0B — Basic Language Structure & Formats
- **Objective**: Ensure the model retains fluent natural language and structured serialization formats.
- **Components**:
  - Factual statements: Geography, basic science, cardinal directions, temporal sequences (days of week, months).
  - JSON objects: Key-value formatting, string values, numeric values, nested dicts.
  - YAML / Markdown: Headings, bullet points, key-value configurations.

### Level 0C — Basic Computation
- **Objective**: Learn token associations for deterministic arithmetic and boolean logic.
- **Components**:
  - Arithmetic tables: Addition, subtraction, multiplication up to 100 (`1 + 1 = 2`, `7 * 8 = 56`).
  - Comparisons: Greater than, less than, equality (`5 > 2 = True`).
  - String transformations: Uppercase, lowercase, reversal, capitalization.

### Level 0D — Basic Programming
- **Objective**: Learn standard Python function signatures, AST validity, and algorithmic patterns.
- **Components**:
  - Function declarations: `def func_name(args) -> type:`.
  - Deterministic return statements: `return a + b`, `return val >= 0`.
  - Control flow: `if condition: ... else: ...`, `for i in range(n): ...`.
  - Canonical utilities: `clamp`, `is_even`, `is_palindrome`, `factorial`, `linear_search`.

### Level 1 — Controlled Reasoning & Debugging Trajectories
- **Objective**: Model causal reasoning and error correction patterns.
- **Components**:
  - Multi-step arithmetic: Order of operations, step-by-step evaluations.
  - Traceback understanding: Reading an `AssertionError` or `TypeError` and identifying the offending line.
  - Patch generation: Modifying buggy code to satisfy assertions.

---

## 3. Dataset Balance & Quota Policy

To prevent domain bias, every curriculum training shard enforces strict composition ratios:
- **Level 0A (Stability)**: 15%
- **Level 0B (Language & Formats)**: 25%
- **Level 0C (Computation)**: 25%
- **Level 0D (Programming)**: 25%
- **Level 1 (Reasoning / Trajectories)**: 10%
