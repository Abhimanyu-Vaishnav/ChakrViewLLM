# ChakrView Step 54: Structured Reasoning Curriculum Specification

- **Document Version**: 1.0.0
- **Status**: Ratified Curriculum Specification
- **Focus**: Multi-Tier Structured Reasoning & Action-Observation Trajectories

---

## 1. Curriculum Architecture

To prepare `ChakrMicro` for the eventual ChakrView intelligence loop without premature complexity, the reasoning curriculum is organized into four progressive tiers:

```
+-------------------------------------------------------+
|  LEVEL R3: Task Decomposition & Trajectory Synthesis   |
|  - UNDERSTAND -> DECOMPOSE -> ACTION -> OBSERVATION   |
|  - Error localization, diagnosis, next action, result |
+-------------------------------------------------------+
                           ^
+-------------------------------------------------------+
|  LEVEL R2: Multi-Step Reasoning                       |
|  - 2 to 4-step arithmetic, chained transformations    |
|  - Logic state transitions, deterministic planning    |
+-------------------------------------------------------+
                           ^
+-------------------------------------------------------+
|  LEVEL R1: Single-Step Reasoning                      |
|  - Comparisons, classification, conditional decisions |
|  - Selecting correct result from candidate options    |
+-------------------------------------------------------+
                           ^
+-------------------------------------------------------+
|  LEVEL R0: Foundational Stability                     |
|  - Delimiters, brackets, operators, structured tokens  |
+-------------------------------------------------------+
```

---

## 2. Tier Details

### Level R0: Foundational Stability
- Re-anchors punctuation, bracket closures, indentation, and delimiters.
- Prevents structural regression and ensures clean sequence termination.

### Level R1: Single-Step Reasoning
- **Comparisons**: `Is 15 > 7? Decision: True`.
- **Classification**: Categorizing tokens (e.g. `Classify 'apple': fruit`, `Classify 'pytest': testing_tool`).
- **Conditional Branching**: Single-condition outcomes (`If x > 0 return positive else negative`).
- **Selection**: Choosing the correct outcome from explicit alternatives (`Which is greater, 12 or 19? Answer: 19`).

### Level R2: Multi-Step Reasoning
- **Chained Arithmetic**: `Calculate (5 + 3) * 2: Step 1: 5 + 3 = 8. Step 2: 8 * 2 = 16. Answer: 16`.
- **Chained Transformations**: `Reverse then capitalize 'cat': Step 1: reverse -> 'tac'. Step 2: uppercase -> 'TAC'. Answer: TAC`.
- **State Transitions**: `State = IDLE. Event = START. New State = RUNNING`.
- **Deterministic Planning**: Small ordered steps (e.g. `Goal: Bake bread. Step 1: Mix flour and water. Step 2: Knead dough. Step 3: Bake in oven.`).

### Level R3: Task Decomposition & Causal Trajectories
- Explicit structured decomposition formatted for 512-token context windows:
```text
<TRAJECTORY>
<SPEC>
Task: Implement is_even(n)
Requirement: Return True if n is divisible by 2, False otherwise
</SPEC>
<UNDERSTAND>
Input is an integer n. Even numbers have remainder 0 when divided by 2.
</UNDERSTAND>
<DECOMPOSE>
Step 1: Compute n % 2
Step 2: Check if remainder equals 0
Step 3: Return boolean result
</DECOMPOSE>
<ACTION>
def is_even(n: int) -> bool:
    return n % 2 == 0
</ACTION>
<OBSERVATION>
test_is_even(4) == True (PASS)
test_is_even(7) == False (PASS)
</OBSERVATION>
<RESULT>SUCCESS</RESULT>
</TRAJECTORY>
```

---

## 3. Balance and Quotas

To prevent domain collapse:
- **Level R0 (Delimiters & Stability)**: 15%
- **Level R1 (Single-Step Reasoning & Classification)**: 25%
- **Level R2 (Multi-Step Chains & State Transitions)**: 30%
- **Level R3 (Decomposition & Action Trajectories)**: 30%
