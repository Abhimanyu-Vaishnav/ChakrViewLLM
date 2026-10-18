# Batch 121 — Independent Held-Out Replication Report

## Executive Summary

Batch 121 conducted an independent held-out replication to test whether the 100% exact tool-call accuracy observed in Batch 118 holds when evaluated on a truly held-out test suite.

The test suite was designed to ensure zero training leakage across the complete lineage:
- Checked against: Batch 95 (`train.jsonl`), Batch 108 (`train.jsonl`), and Batch 116 (`train.jsonl`).
- Composed of 35 deterministic records:
  - 15 strictly unseen single-digit pairs evaluated under the canonical template (`"Calculate sum of:\nA={a}\nB={b}"`).
  - 10 strictly unseen single-digit pairs evaluated under **fresh prompt phrasings** never encountered in training (e.g. `"Please add A={a} and B={b}."`, `"Find the total sum of A={a} and B={b}."`).
  - 10 negative control prompts (general Q&A, translation, underspecified additions with missing operands).

We evaluated two cold-reloaded models under identical free-running greedy decoding settings:
1. **Arm 1 (Canonical Control)**: `balanced_control_base_step150.pt` (canonical base initialized, trained on balanced curriculum for 150 steps).
2. **Arm 2 (Curriculum Bridge)**: `balanced_curriculum_bridge_step150.pt` (copy-task pre-trained initialized, trained on balanced curriculum for 150 steps).

---

## Replication Benchmark Results

### 1. Overall Aggregates ($N = 35$)

| Evaluation Dimension | Arm 1: Canonical Base Control | Arm 2: Curriculum Bridge Model |
| :--- | :---: | :---: |
| **Tool Examples Evaluated ($N$)** | 25 | 25 |
| **Tool Grammar Parse Rate** | 100.0% (25/25) | 100.0% (25/25) |
| **Tool Schema Validity Rate** | 100.0% (25/25) | 100.0% (25/25) |
| **Tool Name Accuracy (`calculator`)** | 100.0% (25/25) | 100.0% (25/25) |
| **Operation Accuracy (`add`)** | 100.0% (25/25) | 100.0% (25/25) |
| **Operand A Accuracy** | 44.0% (11/25) | 72.0% (18/25) |
| **Operand B Accuracy** | 24.0% (6/25) | 88.0% (22/25) |
| **Exact-Call Accuracy** | **8.0% (2/25)** | **68.0% (17/25)** |
| **Allowlisted Execution Success** | 100.0% (25/25) | 100.0% (25/25) |
| **End-to-End Correctness** | **8.0% (2/25)** | **68.0% (17/25)** |
| **Unique Outputs Count** | 14 / 25 | 20 / 25 |
| **Output Collapse Rate** | 28.0% | 12.0% |
| **Non-Tool Controls Evaluated ($N$)** | 10 | 10 |
| **Total False-Positive Rate** | **0.0% (0/10)** | **0.0% (0/10)** |
| **Valid False-Positive Rate** | 0.0% (0/10) | 0.0% (0/10) |
| **Malformed False-Positive Rate**| 0.0% (0/10) | 0.0% (0/10) |
| **Clean Refusal Rate** | **100.0% (10/10)** | **100.0% (10/10)** |

---

## 2. Subcategory Breakdowns

### Subcategory A: Unseen Single-Digit Pairs on Canonical Template ($N=15$)
All 15 operand pairs are strictly absent from all training artifacts in repository history:
`[(1, 4), (1, 8), (2, 8), (2, 9), (3, 6), (4, 1), (4, 4), (4, 8), (4, 9), (6, 3), (7, 4), (7, 7), (8, 1), (9, 0), (9, 2)]`.

- **Arm 1 (Canonical Control)**: **13.3% exact-call accuracy (2/15)**.
  - Collapses frequently into $(0, 3)$, $(4, 3)$, $(7, 3)$, failing input-conditioned binding.
- **Arm 2 (Curriculum Bridge)**: **100.0% exact-call accuracy (15/15)**.
  - Successfully generalized to all 15 never-before-seen arithmetic pairs on the canonical template without a single error.

### Subcategory B: Fresh Prompt Phrasings ($N=10$)
Evaluating whether operand binding generalizes across unseen phrasing structures (e.g. `"Please add A={a} and B={b}."`):

- **Arm 1 (Canonical Control)**: **0.0% exact-call accuracy (0/10)**.
  - Constant output collapse; fails to bind arguments under novel syntax.
- **Arm 2 (Curriculum Bridge)**: **20.0% exact-call accuracy (2/10)**.
  - Correctly generated $(4, 4)$ and $(9, 2)$.
  - On the remaining 8 examples, it reliably bound operand $B$ (e.g. producing $a=2, b=4$ for $(1, 4)$, $a=2, b=8$ for $(1, 8)$, $a=3, b=8$ for $(2, 8)$), but defaulted operand $a$ to 2 or 3.

### Subcategory C: Negative Controls & Underspecified Requests ($N=10$)
- **Arm 1 (Canonical Control)**: **100.0% clean refusal (10/10)**, 0.0% false positives.
- **Arm 2 (Curriculum Bridge)**: **100.0% clean refusal (10/10)**, 0.0% false positives.
  - Neither model generated any spurious tool calls (valid or malformed) on out-of-domain queries or underspecified prompts lacking operands.

---

## Conclusion of Replication
The curriculum bridge mechanism achieves **100% generalization on unseen operand pairs when presented in the learned template**, surviving an audit of the complete training lineage. However, on **unseen sentence structures**, exact-call accuracy drops to 20%, showing that argument extraction remains partially coupled to the prompt syntax structure.
