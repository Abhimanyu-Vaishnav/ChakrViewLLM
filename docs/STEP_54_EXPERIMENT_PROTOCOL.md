# ChakrView Step 54: Experiment Protocol & Reasoning Evaluation

- **Document Version**: 1.0.0
- **Status**: Ratified Experiment Protocol
- **Target Model**: `ChakrMicro` v0.1 (3,443,136 parameters, FP32, CPU-only)

---

## 1. Experiment Matrix

| Experiment | Curriculum Scope | Training Steps | Learning Rate | Batch Size | Device |
|:---|:---|:---|:---|:---|:---|
| **EXP-54A** | Reasoning Foundation (R0 + R1) | 250 | $1 \times 10^{-3}$ | 1 (seq 512) | CPU |
| **EXP-54B** | Chained Reasoning (R0 + R1 + R2) | 500 | $1 \times 10^{-3}$ | 1 (seq 512) | CPU |
| **EXP-54C (Full)** | Comprehensive (R0 + R1 + R2 + R3 Trajectories) | 500 | $1 \times 10^{-3}$ | 1 (seq 512) | CPU |

---

## 2. Benchmark Evaluation Protocol

Each checkpoint is evaluated against three non-overlapping suites under identical greedy sampling (`seed = 42`):
1. **Step-52 Anchor Benchmark** (20 tasks): Stability and foundation tracking.
2. **Step-53 Held-Out Benchmark** (10 tasks): Generalization tracking across unseen prompts.
3. **Step-54 Reasoning Benchmark** (10 tasks): Evaluating multi-step arithmetic, chained logic, state transitions, fault localization, and task decomposition.

---

## 3. Step-54 Reasoning Benchmark Tasks

1. `REASON_MATH_01`: `"Compute (3 + 5) * 2 = "` $\to$ `16`
2. `REASON_MATH_02`: `"Calculate 20 - 4 * 3 = "` $\to$ `8`
3. `REASON_COMP_01`: `"Which is greater, 42 or 37? Answer: "` $\to$ `42`
4. `REASON_TRANS_01`: `"Reverse then capitalize 'dog': "` $\to$ `GOD`
5. `REASON_STATE_01`: `"Current state is IDLE. Event is START. New state is "` $\to$ `RUNNING`
6. `REASON_PLAN_01`: `"Goal: Cook pasta. Step 1: Boil water. Step 2: Add pasta. Step 3: "` $\to$ `Drain` / `Cook`
7. `REASON_DIAG_01`: `"Code 'return a - b' failed in test_add. The error is "` $\to$ `subtraction` / `wrong operator`
8. `REASON_DECOMP_01`: `"To check if a number is negative: Step 1: Check if n < 0. Step 2: Return "` $\to$ `boolean` / `True`
9. `REASON_LOGIC_01`: `"If x > 10 return 'large' else 'small'. For x = 15, result is "` $\to$ `large`
10. `REASON_TRAJ_01`: `"<SPEC>\nTask: Add 10 to x\n</SPEC>\n<ACTION>\ndef add_ten(x):\n    return x + "` $\to$ `10`
