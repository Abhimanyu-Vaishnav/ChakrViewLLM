# Starting Point Audit: Batches 741–780
## Autoregressive Plan DSL, Variable-Length Neural Planning & Safe Execution

**Audit Date:** 2026-10-15  
**Repository:** `D:\Project\ChakrView`  
**Current Branch:** `dev/v0.6-v0.7-validation`  
**Base Commit:** `065061d` (`feat(batch701-740): train neural subtask decomposition model, validate generalization, and evaluate causal sensitivity`)  

---

### 1. Lineage, Working Tree & Checkpoint Verification

- **Branch Status:** On authorized development branch `dev/v0.6-v0.7-validation`.
- **Working Tree Preservation:** User's pre-existing modifications to `chakrview/runtime/session.py` and untracked files remain preserved.
- **Canonical Step 10,000 Checkpoint Hash:**
  - File: `experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt`
  - Measured SHA-256: `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee`
  - Status: **VERIFIED IMMUTABLE AND INTACT**.
- **Batch 701–740 Neural Planner Candidate:**
  - File: `experiments/batch701_740/checkpoints/candidate_step150_neural_planner.pt`
  - Measured SHA-256: `c4e172c8c182f09cdca60ed925254aca59c558b503ff96bfb279e4e07398fec3`
  - Status: **VERIFIED PRESENT AND INTACT**.

---

### 2. Independent Reproduction of Prior Milestone Results

1. **Unit & Integration Suite (`tests/test_batch701_740_neural_decomposition.py`):**
   - Re-executed: **6/6 passed in 19.83s**.
   - Verified: Checkpoint presence and weights, DAG acyclicity validation, compound math prediction, text chain prediction, Hindi generalization, and causal clarification responsiveness.
2. **Held-Out Generalization Benchmark (`experiments/batch701_740/run_neural_planning_evaluation.py`):**
   - Re-executed on 25 held-out tasks:
     - Single-Turn Baseline: **20.0% (5/25)**
     - Deterministic Planner: **20.0% (5/25)**
     - Raw Neural Planner: **48.0% (12/25)**
     - Hybrid Neural + Validator: **80.0% (20/25)**
3. **Causal Sensitivity Interventions:**
   - 3/3 passed (missing input clarification flip, action domain shift, unsupported action gating).

---

### 3. Detailed Component Attribution & Failure Diagnostics

1. **Fixed-Horizon Bottleneck:**
   - `ChakrSubtaskDecompositionModel` predicts fixed heads for step 1 and step 2.
   - For 3-hop tasks (e.g. `test_decomp_0000` through `0007`), the model predicted a 2-step plan, achieving 0% raw neural accuracy on 3-hop tasks. The hybrid pipeline was only able to pass them because the 2-step prefix was syntactically valid.
2. **Unsupported Tool Rejection:**
   - On novel proprietary tools (`test_decomp_0020` to `0024`), the model achieved 0% because its fixed softmax head over `['calculator', 'text_utility', 'none']` had no representation for open-world tool names.
3. **Mission Objective for Batches 741–780:**
   - Replace fixed classification heads with an **Autoregressive Plan DSL Generator**, capable of emitting variable-length plans with arbitrary dependency depth, explicit operand arguments, and clear EOS termination.
