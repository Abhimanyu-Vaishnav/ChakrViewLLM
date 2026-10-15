# Next Milestone Recommendation: Batches 781–820
## Grammar-Constrained Decoding, Schema-Guided Beam Search & Robust Plan Execution

**Target Batches:** 781–820  
**Current Milestone State:** Batches 741–780 Complete  
**Date:** 2026-10-15  

---

### 1. Key Finding from Batches 741–780

The empirical results of Batches 741–780 clearly show that while Autoregressive Plan DSL emission successfully unlocks variable-length decomposition, unconstrained token generation results in syntax drift on longer sequences (>2 steps).

---

### 2. Recommended Next Engineering Priorities

1. **Grammar-Constrained Decoding / CFG Logit Masking (Batches 781–795):**
   - Implement a trie- or regex-based vocabulary logit mask that constrains next-token sampling to legal Plan DSL tokens according to `PLAN_DSL_SPEC.md`.
   - Guarantee 100% syntax validity without retraining by preventing punctuation hallucinations at decoding time.
2. **Schema-Guided Beam Search (Batches 796–805):**
   - Implement beam search (beam width 3–4) scoring candidate plans by graph validity and likelihood, ensuring robust `<END_PLAN>` termination.
3. **End-to-End Multi-Step Execution Benchmarking (Batches 806–820):**
   - Connect constrained plan generation directly into `HierarchicalDialoguePlanner` and evaluate end-to-end multi-step tool execution across 3-hop and 4-hop arithmetic and text pipelines.
