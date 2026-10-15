# Starting Point Audit: Batches 701–740
## Neural Subtask Decomposition, Generalization & Learnability Investigation

**Audit Date:** 2026-10-15  
**Repository:** `D:\Project\ChakrView`  
**Current Branch:** `dev/v0.6-v0.7-validation`  
**Base Commit:** `a9eb0be` (`feat(batch661-700): implement hierarchical dialogue planning, episodic memory, and evidence-grounded multi-step reasoning`)  

---

### 1. Working Tree, Lineage & Checkpoint Verification

- **Branch Status:** On authorized development branch `dev/v0.6-v0.7-validation`.
- **Working Tree Preservation:** User's pre-existing modifications to `chakrview/runtime/session.py` and all untracked docs/experiment files are intact and preserved.
- **Canonical Step 10,000 Checkpoint Hash:**
  - File: `experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt`
  - Measured SHA-256: `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee`
  - Status: **VERIFIED IMMUTABLE AND INTACT**.

---

### 2. Independent Reproduction of Prior Milestone Results

1. **Planning Unit & Integration Suite (`tests/test_batch661_700_dialogue_planning.py`):**
   - Re-executed: **7/7 passed in 1.91s**.
   - Verified: episodic persistence, DAG compound execution, text transform pipeline, missing info abstention, dynamic replanning invalidation cascades, process restart recovery, and cross-session privacy isolation.
2. **Prior 8-Scenario Planning Benchmark (`experiments/batch661_700/build_planning_dataset.py`):**
   - Re-executed:
     - Single-Turn Baseline: **25.0% (2/8)**
     - Hierarchical Task Planner: **100.0% (8/8)**

---

### 3. Detailed Decision Provenance Audit

A rigorous trace of every decision in the Batch 661–700 planner confirms:
1. **Subtask Generation:** Handled by deterministic pattern matching (regex heuristics matching compound math, text transforms, or discount queries) inside `HierarchicalDialoguePlanner.plan_user_goal()`.
2. **Dependency Edge Binding:** Deterministically declared (`dependencies=["st_01"]`, `input_bindings={"base": "st_01.result"}`).
3. **Execution:** AST tools (`CalculatorTool`, `TextUtilityTool`).
4. **Episodic Persistence:** `EpisodicTaskMemory` file storage.
5. **Conclusion:** While the planning system provides an auditable execution framework, **the neural core itself did NOT predict the subtask DAG**. Batches 701–740 directly tackle this learnability bottleneck by training the neural model to predict structured subtasks and dependency edges directly from natural language instructions.
