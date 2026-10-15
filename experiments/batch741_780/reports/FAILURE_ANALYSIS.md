# Failure Analysis: Batches 741–780
## Autoregressive Plan DSL, Variable-Length Neural Planning & Safe Execution

**Analysis Date:** 2026-10-15  
**Candidate:** `ChakrMicro` Autoregressive Plan DSL Generator (`candidate_step250_autoregressive_planner.pt`)  

---

### 1. Honest Assessment of Autoregressive Plan Generation

In accordance with non-negotiable principles #7 and #14, we record the concrete failure modes uncovered during autoregressive generation on the frozen held-out benchmark:

1. **Syntax Drift & Token Drifting on Long Sequences:**
   - On 1-step and 2-step tasks, the model achieved **50.0–100.0%** syntax validity and clean `<END_PLAN>` termination.
   - However, on 3-step and 4-step pipelines, free-form greedy decoding drifted into punctuation artifacts (e.g. `args={text='Ne}`), resulting in 0% valid parsing on 4-step tasks.
   - *Root Cause:* Without a constrained grammar mask during token sampling, the small 6-layer model accumulates sequence drift over 80+ generated tokens.
2. **Comparison with Fixed-Head Model (Batch 701–740):**
   - Fixed-Head Model: 43.3% accuracy on <= 2 steps, but mathematically incapable of predicting >= 3 steps.
   - Autoregressive Model: 26.7% raw accuracy across all lengths. It *can* emit variable lengths (1, 2, and 3 steps), but suffers from lower token-level exact precision than classification heads.
3. **Safety Through Gating:**
   - Because `PlanDSLValidator` enforces terminal `<END_PLAN>` and strict syntax parsing, drifted or truncated plans were **safely rejected** with zero unauthorized executions.

---

### 2. Strategic Technical Takeaway

Autoregressive plan generation solves the structural limitation of fixed output heads (allowing variable step counts and arbitrary argument names), but requires **Constrained Context-Free Grammar (CFG) Decoding** to eliminate syntax drift on long sequences.
