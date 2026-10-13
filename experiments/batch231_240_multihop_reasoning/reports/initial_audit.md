# Batch 231: Independent Audit of Prior Results & Starting Invariants

**Date**: 2026-10-13  
**Auditor**: Principal Autonomous AI / Systems Engineer  
**Workspace**: `d:\Project\ChakrView`  
**Branch**: `dev/v0.6-v0.7-validation`  
**Delivered Commit Audited**: `7b3daf1` (`feat(batch221-230): improve relation-aware pointer generalization`)

---

## 1. Executive Summary & Verification of Invariants

| Invariant / Metric | Expected Value | Actual Verified Value | Status |
|---|---|---|---|
| **Git Branch** | `dev/v0.6-v0.7-validation` | `dev/v0.6-v0.7-validation` | **VERIFIED** |
| **Commit at HEAD** | `7b3daf1` | `7b3daf1` | **VERIFIED** |
| **Canonical Checkpoint Hash** | `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` | `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` | **VERIFIED INTACT** |
| **Canonical Unique Parameters** | 3,443,136 | 3,443,136 across 6 layers, $d_{model}=192$, vocab 4096, max seq 512 | **VERIFIED** |
| **Device Configuration** | Hardware-adaptive / CPU | Intel/AMD x86_64, Windows, `torch.cuda.is_available() == False` | **CONFIRMED CPU ONLY** |
| **Protected `session.py`** | Preserved user ChatML edit | Working tree modified: contains ChatML conversation formatting | **PRESERVED** |
| **Frozen Test Manifest** | `frozen_test_manifest_100.json` | Hash: `f345670a7b49ed1fd17b9c700fbcf468e597ee3035e50278dc439dbfdc0fd7aa` | **VERIFIED (100 items)** |

---

## 2. Reproduction of Reported Prior Results

We re-evaluated the controlled arms on the exact 100-item manifest (`f345670...`):
- **Arm A2 (Frozen Prior Pointer + TokenTrie)**: 53/100 (53.0%) EM.
- **Arm B2 (Relation-Aware Pointer + TokenTrie)**: 62/100 (62.0%) EM.
- **Arm C (Control Pointer + TokenTrie)**: 56/100 (56.0%) EM.
- **Arm D (Deterministic Slot Extractor)**: 48/100 (48.0%) EM.
- **Arm E (BM25 Lexical Grounding)**: 68/100 (68.0%) surface chunk grounding.

---

## 3. Investigation of Potential Reporting Inconsistencies

1. **`chakrview/runtime/session.py` Status**:
   - `git status --short` confirms ` M chakrview/runtime/session.py`. It is a tracked file with local working-tree modifications. It is neither committed nor untracked.
   - It remains strictly unstaged and preserved in its exact user-edited state.
2. **Diagnostic Slice Overlap in Manifest**:
   - The 100-item manifest contains multiple non-exclusive diagnostic tags: 10 Hindi items, 38 multi-token items, 62 single-token items, and 21 numeric distractor confusions.
   - For Batches 231–240, diagnostic slices must be explicitly reported with their exact subset sizes ($N_k$), while maintaining a single, fixed primary benchmark denominator.
3. **Candidate Generation & Oracle Check**:
   - Candidates are extracted exclusively from the input context text via n-gram ($n \le 3$) and whitespace/punctuation tokenization. No test-label metadata or global label dictionaries are leaked.
4. **Runtime Checkpoint Loading**:
   - In `run_runtime_regression_eval.py`, `ChakrRelationPointerHead` is explicitly loaded from `relation_aware_pointer_step1000.pt`. When absent, `GroundedSessionAdapter` safely defaults to deterministic extraction or neural core fallback.
