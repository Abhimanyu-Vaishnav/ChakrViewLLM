# Batch 221 — Independent Audit of Previous Milestone (Batches 211–220)

**Date**: 2026-10-13  
**Auditor**: Autonomous Principal AI / Systems Engineer  
**Workspace**: `d:\Project\ChakrView`  
**Target Branch**: `dev/v0.6-v0.7-validation`  
**Delivered Commit Audited**: `2a10c5a` (`feat(batch211-220): train and evaluate relational pointer learning`)

---

## 1. Executive Summary & Verification of Invariants

| Invariant / Claim | Expected State | Actual Verified State | Status |
|---|---|---|---|
| **Git Branch** | `dev/v0.6-v0.7-validation` | `dev/v0.6-v0.7-validation` | **VERIFIED** |
| **Commit Presence** | `2a10c5a` at HEAD | `2a10c5a` at HEAD | **VERIFIED** |
| **Canonical Checkpoint Hash** | `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` | `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` | **VERIFIED INTACT** |
| **Canonical Unique Parameters** | 3,443,136 | 3,443,136 across 6 layers, $d_{model}=192$, vocab 4096, context 512 | **VERIFIED** |
| **Device Configuration** | CPU (Intel/AMD x86_64, Windows) | `torch.cuda.is_available() == False` | **CONFIRMED CPU ONLY** |
| **Protected `session.py`** | Protected user edit preserved | Working tree modified: contains ChatML formatting modification | **PRESERVED** |

---

## 2. Investigation of Critical Discrepancies

### A. Discrepancy 1: Working Tree Status of `session.py`
- **Previous Claim**: The prior summary claimed `session.py` was "untracked/unstaged".
- **Actual Investigation**: `git status --short` shows ` M chakrview/runtime/session.py`.
- **Finding**: It is **tracked and modified in the working tree** (lines 314–333 modified to format turn dialogue using `ConversationTemplate` and `ChatMessage`). It is neither committed nor untracked.
- **Resolution**: This user modification must remain strictly uncommitted, unstaged, and untouched.

### B. Discrepancy 2: Synthetic Corpus Cluster Count vs Record Count
- **Investigation**: The 10,000 generated records in `experiments/batch211_220_relational_learning/data/` were partitioned across:
  - Train: 6,360 records (25 entities, 50 relations, 100 underlying facts).
  - Val: 1,097 records (25 entities, 50 relations, 100 underlying facts).
  - Test: 2,543 records (9 entities, 24 relations, 36 underlying facts).
- **Leakage Audit**:
  - Unique entity overlap between Train and Test: **0 entities** (0% leakage).
  - Unique fact overlap between Train and Test: **0 facts** (0% leakage).
- **Cluster Finding**: The 14 entity clusters mentioned in prior prompts corresponded to the original seed clusters, but the generator expanded them deterministically into 25 training entities and 9 held-out test entities.

### C. Discrepancy 3: Sample Equivalence Between Baseline (32/100) and Trained Pointer (47/100)
- **Investigation**:
  - In `run_baselines_eval.py`, line 52: `random.seed(42); test_sample = random.sample(all_test, 100)`.
  - In `run_relational_pointer_training.py`, lines 37 and 60: `random.seed(42); train_subset = random.sample(all_train, 200)` called *before* sampling `test_subset = random.sample(test_data, 100)`.
- **Critical Finding**: Calling `random.sample(all_train, 200)` advanced the Python random state. Consequently, the test sample evaluated in `run_baselines_eval.py` shared **only 5 out of 100 records** with the test sample evaluated in `run_relational_pointer_training.py`!
- **Impact**: The reported 32/100 $\to$ 47/100 improvement was **not measured on identical test sets**. While the direction of improvement is plausible, it violates the scientific requirement for identical evaluation cases.
- **Resolution for Batches 221–230**: We must freeze an immutable, serialized test manifest (`frozen_test_manifest_100.json`) with fixed example IDs, candidate sets, and normalization rules. All arms must evaluate against the exact same frozen file.

### D. Discrepancy 4: Generalization Slice Overlap
- **Investigation**: In `run_generalization_eval.py`, the `in_distribution_seen` slice drew 50 examples from the *training pool*, while `held_out_entity`, `numeric_distractors`, and `multilingual_devanagari` drew from the test pool.
- **Finding**: Slices had non-mutually exclusive criteria (e.g., an example with numeric distractors could also be held-out entity).
- **Resolution for Batches 221–230**: Establish a single primary test manifest with fixed denominator, explicitly recording diagnostic slice tags per record rather than presenting overlapping subsets as independent partitions.

### E. Discrepancy 5: Runtime Checkpoint Loading
- **Investigation**: In `experiments/batch211_220_relational_learning/run_runtime_regression_eval.py`, line 39: `p_head = ChakrPointerHead(d_model=192); p_head.load_state_dict(torch.load(pointer_ckpt))`.
- **Finding**: The test script did load the real checkpoint weights. However, `GroundedSessionAdapter` defaults `pointer_head=None` unless explicitly passed.
- **Resolution for Batches 221–230**: Verify runtime initialization with both loaded checkpoint and safe fallback paths.
