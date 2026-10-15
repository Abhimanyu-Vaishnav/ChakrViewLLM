# Next Milestone Recommendation: Batches 741–780
## Autoregressive Plan Generation, Variable-Horizon DAG Emission & Runtime Tool Synthesis

**Target Batches:** 741–780  
**Current Milestone State:** Batches 701–740 Complete  
**Date:** 2026-10-15  

---

### 1. Analysis of Bottlenecks Uncovered in Batches 701–740

Batches 701–740 demonstrated that while fixed multi-task heads can learn subtask typing and dependency edges (reaching 80% in a hybrid pipeline), fixed heads inherently suffer from **horizon truncation**:
- Instructions requiring 3 or more hops (e.g. *multiply -> divide -> add*) cannot be represented cleanly without scaling head complexity.
- Novel tools with unfamiliar names cannot be predicted by a fixed softmax head over a static tool enum.

---

### 2. Recommended Next Engineering Priorities

1. **Autoregressive Plan Token Emission (Batches 741–755):**
   - Train `ChakrMicro`'s language modeling head to emit structured plan DSL tokens autoregressively (e.g. `<PLAN> <STEP tool="calc"> <DEP 0> ... </PLAN>`) using constrained decoding.
2. **Open-Vocabulary Argument Slot Copying (Batches 756–765):**
   - Combine the span pointer head from Batches 581–620 with autoregressive plan generation to directly extract and bind variable operands into subtask argument slots.
3. **Dynamic Graph Execution Engine (Batches 766–780):**
   - Integrate autoregressively emitted DAG plans into the `HierarchicalDialoguePlanner` runtime, evaluating complete end-to-end execution across arbitrary chain lengths.
