# WP 938–940: Failure Analysis & Dissection of Fusion Metrics

## 1. Quantitative Dissection: Why Did End-to-End Match Yield 2/70 (2.9%)?

The comparative 6-configuration benchmark evaluated on the frozen 70-item dataset revealed the exact behavioral divergence between individual components and the unified pipeline:

| Configuration | Syntax Validity | Exact Plan Match | Slot Correctness |
| :--- | :---: | :---: | :---: |
| **Config A: Base AR Planner** | 37/70 (52.9%) | 6/70 (8.6%) | 6/70 (8.6%) |
| **Config B: Raw Span Model** | 49/70 (70.0%) | 0/70 (0.0%) | 23/70 (32.9%) |
| **Config C: Unified Fusion** | **70/70 (100.0%)** | 2/70 (2.9%) | 3/70 (4.3%) |
| **Config D: Oracle Struct + Pred Spans** | 70/70 (100.0%) | 0/70 (0.0%) | 23/70 (32.9%) |
| **Config E: Pred Struct + Oracle Spans** | 70/70 (100.0%) | 2/70 (2.9%) | 70/70 (100.0%) |
| **Config F: Oracle Struct + Oracle Spans**| 70/70 (100.0%) | 70/70 (100.0%) | 70/70 (100.0%) |

---

## 2. Root Failure Analysis of Unified Fusion

1. **Topological Discrepancy (Config E = 2/70 vs Config F = 70/70)**:
   - Config E (Predicted Structure + Oracle Spans) scored only **2/70 (2.9%)**, demonstrating that the high-level plan topology (number of steps, intermediate variable references like `$0`, `$1`, and arithmetic operators) predicted from the prompt did not match the ground-truth benchmark topologies for 68 of the 70 items.
   - Specifically, the frozen benchmark contains diverse multi-step mathematical chains (items 15–45 are 3-step and 4-step chains), while the prompt topology heuristic defaults to 2-step structures.
2. **Span Slot Mis-assignment (Config C = 3/70 vs Config D = 23/70)**:
   - In Config D, slot 0 was evaluated in isolation against the ground-truth operand, scoring 23/70 (32.9%).
   - In Config C, slot 0 and slot 1 had to be bound simultaneously. When slot 0 captured the wrong span (e.g. `14` instead of `600`), the entire step expression failed exact match.
3. **Repeated Numbers and Recency Bias**:
   - In causal test 1 (`Calculate 750 * 5 and then add 14`), the model bound `14` into slot 0, producing `expr=14*14`. The attention mechanism has an overwhelming bias towards the tail of the prompt sentence.

---

## 3. Key Takeaway
The unified pipeline achieves **100.0% valid Plan DSL syntax** and completely eliminates parse errors, but genuine end-to-end exact plan match requires joint autoregressive generation of slot-tagged grammar tokens rather than two-stage post-hoc template substitution.
