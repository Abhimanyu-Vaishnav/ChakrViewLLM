# Reproducibility and Security Review: Batches 661–700
## Hierarchical Dialogue Planning, Episodic Memory & Evidence-Grounded Multi-Step Reasoning

**Audit Date:** 2026-10-15  
**Repository:** `D:\Project\ChakrView`  
**Current Branch:** `dev/v0.6-v0.7-validation`  
**Base Commit:** `b56db74`  

---

### 1. Cryptographic Checkpoint & Artifact Integrity

| Checkpoint Identifier | File Path | Expected SHA-256 | Measured SHA-256 | Integrity Status |
| :--- | :--- | :--- | :--- | :--- |
| **Canonical Step 10,000** | `experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt` | `96357705a683...` | `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` | **VERIFIED IMMUTABLE** |
| **Batch 541–580 Intent Candidate** | `experiments/batch541_580/checkpoints/candidate_step250_intent_understanding.pt` | `9c3204906e8e...` | `9c3204906e8e94f5e80c5db9f6ffb0fa6903c6fe3159206fb164e44d52b8357e` | **VERIFIED MATCH** |
| **Batch 581–620 Span Pointer** | `experiments/batch581_620/checkpoints/candidate_step250_span_pointer.pt` | `196384e9c29c...` | `196384e9c29ca1042ecf65e379972257bdcbce1667f4befb0e4249e3f7c5b569` | **VERIFIED MATCH** |

---

### 2. Dataset Leakage & Scenario Auditing

All planning scenarios were generated via `experiments/batch661_700/build_planning_dataset.py` with explicit disjoint task structures:
- `multi_step_arithmetic` (chained DAG execution)
- `text_draft_transform` (intermediate artifact generation and transformation)
- `clarification_abstention` (missing parameter detection and refusal to execute side effects)
- `dynamic_replanning` (selective invalidation of downstream subtasks while preserving upstream completed artifacts)
- `restart_and_resume` (persistent episodic recovery across process restarts)
- `session_isolation` (privacy isolation across independent tenant/session identifiers)

**Leakage Audit Result:** 0.0% overlap with prior training sets or held-out test splits.

---

### 3. Execution Commands to Reproduce Findings

To reproduce all benchmarks and tests from a clean shell:

```bash
# 1. Reproduce Hierarchical Planning Benchmark
python -m experiments.batch661_700.build_planning_dataset

# 2. Run Batch 661-700 Unit & Integration Suite (7 passed)
python -m pytest tests/test_batch661_700_dialogue_planning.py -v

# 3. Run Full Combined Regression Suite across Batches 541 to 700 (48 passed)
python -m pytest tests/test_batch541_580_neural_intent.py tests/test_batch581_620_span_pointer.py tests/test_batch621_660_multiturn_context.py tests/test_batch661_700_dialogue_planning.py tests/test_persistent_memory.py -v
```

---

### 4. Security & Privacy Hardening

1. **Strict Cross-Session Privacy Isolation:**
   - Evaluated in `test_cross_session_privacy_isolation` and `scen_plan_008`.
   - Verified that an unauthorized session querying `get_active_plan()` receives `None` and cannot access intermediate artifacts or executed subtask results of other sessions.
2. **Side-Effect Prevention on Ambiguous Plans:**
   - If a required parameter (such as a discount rate or missing operand) is absent, the plan transitions to `AWAITING_CLARIFICATION`, subtasks remain `BLOCKED`, and tools are prevented from executing on ungrounded defaults.
3. **User Working Tree Preservation:**
   - Pre-existing user changes in `chakrview/runtime/session.py` were strictly preserved.
