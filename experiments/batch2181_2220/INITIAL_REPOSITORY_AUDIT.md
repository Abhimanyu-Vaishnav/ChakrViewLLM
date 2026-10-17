# INITIAL REPOSITORY AUDIT

**Batch:** 2181–2220  
**Date:** October 2026  
**Auditor:** ChakrView Senior Systems & ML Research Engineer  
**Workspace:** `D:\Project\ChakrView`  
**Current Branch:** `dev/v0.6-v0.7-validation`  
**Starting Commit:** `574aa3d` (`feat(batch2141-2180): controlled CPT 3-seed run, source-aware evaluation, and feedback learning foundation`)  
**Parent Commit:** `6955858`  

---

## 1. Git Working Tree State

- **Branch Status:** Tracking `dev/v0.6-v0.7-validation` as expected.
- **Tracked Modifications:**
  - `chakrview/runtime/session.py` (Pre-existing modification; protected and untouched by this mission).
- **Untracked Directories & Files:**
  - `docs/sprints/` (historical sprint notes)
  - `experiments/batch103_106_slot_weighted_sft/` through `experiments/batch119_122_independent_replication/`
  - Historical test reports under `experiments/batch83_86...` and `experiments/batch91_94...`
  - `tests/test_batch103_...` through `tests/test_batch115_...`
- **Pre-existing Integrity Policy:** None of the pre-existing user files or modifications were reset, discarded, or overwritten.

---

## 2. Protected Artifact Verification

All protected reference artifacts reported in canonical project documentation were independently verified:

| Protected Artifact | Expected / Canonical Location | Observed Value | Verification Result |
| :--- | :--- | :--- | :---: |
| **Canonical Checkpoint (Step 10,000)** | `experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt` | SHA-256: `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` | **VERIFIED MATCH** |
| **Canonical Architecture** | `chakrview/brain/config.py` & `model.py` | 3,443,136 parameters, hash: `c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da` | **VERIFIED MATCH** |
| **Canonical Tokenizer** | `release/chakrview-v0.5.0/data/experiments/vocab_4096/` | 4,096 vocab size, `vocab.json` SHA-256: `1d59ee38...` | **VERIFIED MATCH** |
| **Cleaned Corpus Manifest** | `experiments/batch2101_2140/CORPUS_V2_CLEANED_MANIFEST.json` | SHA-256: `e3db033c5e886915b81a04e578f7db0d9472eb29ba0e5a1b32bb3da6cfbb7a69` | **VERIFIED MATCH** |
| **Historical Candidate (150 steps)** | `experiments/batch1861_1900/checkpoints/cpt_balanced_seed_42.pt` | SHA-256: `85753315bb55853f0e8bad91a916c3846c848333a1b8e3b9b02a5256b7aa7f1c` | **VERIFIED MATCH** |
| **Historical Candidate (300 steps)** | `experiments/batch1901_1940/checkpoints/cpt_balanced_seed_42_300steps.pt` | SHA-256: `2f7751511b623e1dd06bbe2d721df71e679da34d48d4b443f092600d768e7628` | **VERIFIED MATCH** |

---

## 3. Audit of Batch 2141–2180 Deliverables

1. **CPT Run Artifacts:** Verified in `experiments/batch2141_2180/cpt/`:
   - `seed_42`: `cpt_best_checkpoint.pt` (SHA-256: `d3268fc4...`), `cpt_final_checkpoint.pt` (SHA-256: `3039ded5...`).
   - `seed_1337`: `cpt_best_checkpoint.pt` (SHA-256: `05c021c3...`), `cpt_final_checkpoint.pt` (SHA-256: `07b514d2...`).
   - `seed_2026`: `cpt_best_checkpoint.pt` (SHA-256: `c47e826d...`), `cpt_final_checkpoint.pt` (SHA-256: `e6fbfdcf...`).
   - `cpt_master_summary.json`: Present and valid JSON.
2. **Evaluation Results:** Verified in `experiments/batch2141_2180/evaluation_results.json`:
   - Contains evaluation results across all 6 model configurations on 290 test documents (64,769 tokens).
3. **Feedback Engine:** `experiments/batch2141_2180/feedback_learning_engine.py`:
   - Verified existence of `ConversationMemoryStore`, `FeedbackExperienceLedger`, `ControlledLearningDatasetExporter`, and `SessionFeedbackAdapter`.
4. **Test Suite:** `tests/test_batch2141_2180_cpt_and_feedback.py`:
   - Executed via `pytest`; 4/4 tests passed.
5. **Initial Observations for Reconciliation:**
   - Text narratives in `MISSION_SUMMARY.md` and `CAPABILITY_MATRIX.md` cited "47 documents, 15,264 tokens", whereas `evaluation_results.json` and `CORPUS_V2_CLEANED_MANIFEST.json` evaluate 290 documents and 64,769 tokens.
   - The reported bootstrap confidence interval `[-0.2622, -0.2099]` represents the **unweighted paired document delta**, not the token-weighted delta `-0.1928`.
   - Seed completion counts (200, 125, 125) reflect the preregistered early stopping criterion (3 consecutive checkpoints without validation improvement), not run failures.

---

## 4. Hardware Environment Detection

- **OS / Platform:** Windows 11 / AMD64
- **Python Executable:** `D:\Project\ChakrView\.venv_gpu\Scripts\python.exe` (Python 3.14.7)
- **PyTorch Version:** `2.14.1+cu126`
- **CUDA Device:** `NVIDIA RTX A1000 6GB Laptop GPU` (Driver / CUDA 12.6 supported and functional)
- **System Memory:** 15.65 GB RAM (Current utilization: ~57%)
- **Finding:** A capable GPU is physically installed and recognized by PyTorch. The historical reliance on CPU was caused by explicit `device = "cpu"` hardcoding in the Batch 2141–2180 runner scripts.
