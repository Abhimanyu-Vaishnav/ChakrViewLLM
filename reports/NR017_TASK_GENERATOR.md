# NR-017 Phase 1 & 2 — Verifiable Algorithmic Task Generator & Condition Specifications
**Mission ID:** CV-GENESIS-NR-017  
**Date:** 2026-11-26  
**Generator:** `experiments/genesis_stage_g_017/task_generator.py` (Deterministic seed: 777)  
**Training Corpus (1,800 tasks):** `data/stage_g_017/stage_g_train_1800.json` (SHA-256: `1da85a77f2200beea3f2d5b37c7878e40c61bc102acb41c25e3af05cb14ab81c`)  
**Sealed Benchmark (400 tasks):** `data/stage_g_017/stage_g_sealed_benchmark_400.json` (SHA-256: `8bf8ff917118307f3c72687e67d3d75b81388b632830bb3c7c55bd28712a1bcc`)

---

## 1. Algorithmic Families & Verifiable State Specifications
The generator produces problems with mathematically deterministic state transitions across 10 families:
1. `variable_binding`: Dynamic mapping of slots $v_1, v_2$ to numeric values.
2. `variable_renaming`: Inter-variable dependency tracking ($v_2 = v_1 + 10$).
3. `arithmetic_operations`: Multiplication, division, addition, subtraction.
4. `sequential_state_updates`: Accumulator state transitions ($S_0 \rightarrow S_1 \rightarrow S_2$).
5. `conditional_branching`: Threshold comparison and deterministic branching.
6. `nested_operations`: Parenthesized expression evaluation (inner $\rightarrow$ outer).
7. `list_transformations`: Algorithmic list mutations (reversal, max, slicing).
8. `string_transformations`: Multi-step string operations (casing, reversal).
9. `stack_operations`: LIFO push/pop state tracking.
10. `algorithmic_compositions`: Interacting variable, arithmetic, and logic checks.

---

## 2. Specification of the Three Research Conditions

### Condition A: Direct Answer
- **Format:** `<task> ... </task>` $\rightarrow$ `<answer> ... </answer>`
- **Objective:** The network must map directly from problem description to final answer in a single autoregressive jump without producing intermediate computation tokens.

### Condition B: Explicit State Trace
- **Format:** `<task> ... </task>` $\rightarrow$ `<state_1> ... </state_1> <state_2> ... </state_2> <answer> ... </answer>`
- **Objective:** The network must emit every discrete intermediate computational state in an explicit XML-like container before emitting the final answer.

### Condition C: Structured Compact Trace
- **Format:** `<task> ... </task>` $\rightarrow$ `<trace> S0=... -> S1=... -> answer=... </trace>`
- **Objective:** A compact token-efficient arrow-delimited sequence that preserves 100% of the mathematical state transitions while reducing token overhead.

---

## 3. Disjoint OOD Split Protocol
- **In-Distribution Variables:** `x, y, z, a, b, c, p, q, m, n`
- **Strict Out-of-Distribution Variables:** `alpha, beta, gamma, delta, omega, sigma, theta, phi`
- **In-Distribution Numeric Ranges:** $[2, 20]$ (L1), $[10, 50]$ (L2), $[20, 80]$ (L3)
- **Strict Out-of-Distribution Numeric Range:** $[85, 160]$
- **Benchmark Partition:** 200 in-distribution tasks, 200 strict out-of-distribution tasks (Total: 400 sealed tasks).
