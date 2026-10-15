# WP 947–953: Topology Failure Taxonomy & Empirical Diagnostics

## 1. Failure Taxonomy Classification across Frozen Benchmark ($N=70$)

A granular diagnostic audit was conducted for all 70 items comparing expected task graphs against predicted topologies:

| Failure Category | Item Count | Percentage | Primary Underlying Mechanism |
| :--- | :---: | :---: | :--- |
| **Wrong Step Count (3-step as 2-step)** | 21 / 70 | 30.0% | Model lacks neural step-count prediction; heuristic defaults to 2-step template |
| **Wrong Step Count (4-step as 2-step)** | 12 / 70 | 17.1% | Multi-clause arithmetic prompt structure not parsed into 4 distinct steps |
| **Slot Recency Bias (Tail Number Duplication)**| 20 / 70 | 28.6% | Attention pooling prioritizes last mentioned operand (e.g. `14`), duplicating it into slot 0 |
| **String Sub-phrase Fragmentation** | 6 / 70 | 8.6% | Multi-word phrase slicing dropped initial title word |
| **Unsupported Operation Mis-routing** | 6 / 70 | 8.6% | Routed to generic arithmetic rather than tool rejection |
| **Correct Plan Execution** | 5 / 70 | 7.1% | 2-step arithmetic and clean clarification abstentions |

---

## 2. Breakdown by Task Topology Step-Count

- **1-step / Clarification Tasks ($N=12$)**:
  - Predicted Structure Match: 6/12 (50.0%)
  - Failure: Unsupported tools generated default 2-step plans instead of rejecting.
- **2-step Arithmetic Tasks ($N=15$)**:
  - Predicted Structure Match: 15/15 (100.0%)
  - Exact Plan Match: 2/15 (13.3%)
  - Failure: Slot recency bias (`expr=14*14` instead of `expr=600*5`).
- **3-step Arithmetic Tasks ($N=21$)**:
  - Predicted Structure Match: 0/21 (0.0%)
  - Exact Plan Match: 0/21 (0.0%)
  - Dominant Failure: Truncation to 2-step topology.
- **4-step Arithmetic Tasks ($N=12$)**:
  - Predicted Structure Match: 0/12 (0.0%)
  - Exact Plan Match: 0/12 (0.0%)
  - Dominant Failure: Dependency chain mis-wiring and step count collapse.
- **Text Chain Tasks ($N=10$)**:
  - Predicted Structure Match: 10/10 (100.0%)
  - Exact Plan Match: 0/10 (0.0%)
  - Failure: Partial string extraction.

---

## 3. Dominant Bottleneck Identified
The single dominant failure mode is **two-fold**:
1. **Topology Collapse**: Heuristic topology prediction cannot generalize across variable-length dependency graphs (1-step vs 2-step vs 3-step).
2. **Slot Recency Bias**: Span grounding without explicit token-slot affinity heads repeatedly assigns the final prompt operand to all slots.
