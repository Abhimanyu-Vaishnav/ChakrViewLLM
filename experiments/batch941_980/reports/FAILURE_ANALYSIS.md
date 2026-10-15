# WP 979–980: Failure Analysis & Root Cause Audit — Batches 941–980

## 1. Which Failure Category Dominates?

Across the comprehensive diagnostic audit and experimental evaluations, the dominant failure category is **Topology-Slot Decoupling with Tail Recency Bias**:
1. **Topology Collapse (47.1% of frozen benchmark failures)**: Heuristic topology generation collapsed all multi-step plans into a fixed 2-step structure.
2. **Tail Recency Bias (28.6% of argument failures)**: Standard span attention heads preferentially assign the final operand mentioned in the prompt (e.g. `14` from "...and then add 14") to *all* argument slots, emitting expressions like `14*14` or `14+14`.

---

## 2. Impact of Left-to-Right Positional Slot Binding

By introducing positional sequence tracking in `NeuralTopologyPlannerPipeline`:
- Exact plan match on the focused arithmetic task family jumped from **0/33 (0.0%)** to **12/33 (36.4%)**.
- Operands like `600`, `5`, and `14` were correctly allocated to their semantic step roles (`expr=600*5` followed by `expr=$0+14`), completely avoiding the `expr=14*14` self-duplication bug.

---

## 3. What Remains Unproven?

While exact plan match reached 36.4% on the focused family, generalizing this capability to arbitrary 3-step and 4-step dependency DAGs without heuristic scaffolding remains unproven. A truly general planner requires end-to-end autoregressive token generation of abstract slot markers.
