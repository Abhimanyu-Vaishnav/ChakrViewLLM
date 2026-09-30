# ChakrView Step 54: Controlled Learning Loop Specification

- **Document Version**: 1.0.0
- **Status**: Ratified Learning Loop Architecture (Pre-RIL)
- **Scope**: Controlled Experience Ingestion, Selection, and Gated Checkpoint Admission

---

## 1. Architectural Purpose

To prevent unconstrained or catastrophic self-modification, ChakrView separates learning into a disciplined, multi-stage gated pipeline:

```
[ChakrKshetra Execution]
           |
           v
  (Capture Trajectory)
           |
           v
  [Experience Validation] ---> Invalid? ---> (Discard / Log Defect)
           |
           v (Valid)
  [Episodic Memory Store]
           |
           v
  [Sample Curation & Sharding]
           |
           v
  [Train Candidate Checkpoint]  <-- Isolated from Frozen Baseline
           |
           v
  [Dual-Benchmark Evaluation Gate]
           |
      Meets Criteria?
      /             \
    Yes              No
    /                 \
[Validated Checkpoint] [Reject Candidate]
```

---

## 2. Checkpoint Hierarchy & Nomenclature

1. **Frozen Baseline Checkpoint**:
   - SHA-256: `c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da`.
   - Invariant: $\Delta W_{\text{baseline}} = 0$. Never overwritten, never modified in-place.
2. **Experimental Candidate Checkpoint**:
   - Checkpoints produced by curriculum or training runs (e.g., `checkpoint_step00500.pt`).
   - Isolated in `artifacts/stepNN/checkpoints/`.
3. **Validated Checkpoint**:
   - A candidate checkpoint that passes the **Dual-Benchmark Evaluation Gate**:
     - $\ge 50\%$ pass rate on anchor benchmark.
     - $\ge 40\%$ pass rate on held-out generalization benchmark.
     - $0\%$ repetition collapse.
     - Zero regression on existing test suite.

---

## 3. Experience Validation Schema

Before an experience record can be converted into training shards, it must satisfy:
1. `converged == True` or clear diagnosis recorded.
2. Token length $\le 512$ tokens.
3. No secret or private key leakage.
4. Deterministic outcome (re-execution produces identical test result).
