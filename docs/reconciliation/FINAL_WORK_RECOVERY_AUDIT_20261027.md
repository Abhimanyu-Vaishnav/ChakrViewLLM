# CHAKRVIEW — FINAL UNCOMMITTED WORK RECOVERY & VERIFICATION AUDIT

**Audit Date:** 2026-10-27T22:35:00+05:30
**Lead Auditor:** Senior Git Repository Maintainer & Release Integrity Engineer
**Repository Root:** `D:\Project\ChakrView`
**Remote Origin:** `https://github.com/Abhimanyu-Vaishnav/ChakrViewLLM.git`
**Current Branch:** `main` (commit: `f4f9a0551cb37b3717668a12075e767c9769adac`)
**Upstream Tracking:** `origin/main` (commit: `f4f9a0551cb37b3717668a12075e767c9769adac`)
**Sync Status:** **100% IDENTICAL & SYNCHRONIZED TO REMOTE**

---

## 1. Executive Summary & Verification Findings

This audit independently investigated the entire filesystem and Git index to verify whether any eligible pending work remained uncommitted or unsynchronized:

1. **Git Working Tree Status:**
   - Modified tracked files: **0**
   - Staged changes: **0**
   - Untracked changes: `experiments/overnight_runs/` (Preserved Tier 2 Local Research Archive).
   - Upstream divergence: **0 commits ahead, 0 commits behind** (`origin/main`).
2. **Recent Synchronization History:**
   - All eligible source, test, data, and documentation files across Experiments 31, 32, and 33, as well as the Governance PR #3 completion report and reconciliation matrix, were committed and pushed across three commits:
     - `a2d742a`: `docs(research): update experiment registry with Exp31-Exp33 and record governance completion report` (2 files, +110 lines)
     - `e996e5a`: `feat(research): archive Experiment 31-33 protocols, datasets, results, and relational pointer circuits` (54 files, +6,765 lines)
     - `f4f9a05`: `docs(reconciliation): record comprehensive repository inventory, pending changes matrix, and post-push report` (4 files, +281 lines)
   - Total files committed & pushed across the three operations: **60 files (+7,156 lines)**.
3. **Live Remote Verification:**
   - `git ls-remote origin refs/heads/main` returns `f4f9a0551cb37b3717668a12075e767c9769adac`.
   - All 60 files exist on GitHub on branch `main`.
4. **Preserved Local Research Artifacts:**
   - `experiments/overnight_runs/`: 25,470 total files (5.12 GB, 122 `.pt` weights) remain intact on disk.
   - Experiments 31–33 Checkpoints: 6 `.pt` weights (82.35 MB) remain intact in `experiments/exp3*/checkpoints/`.
   - Historical Checkpoints: 462 `.pt` weights (14.8 GB) remain intact in `artifacts/` and `checkpoints/`.
   - Excluded Scratch: Virtual environments (`.venv/`, `.venv_gpu/`) and bytecode (`__pycache__/`) are excluded by `.gitignore`.

---

## 2. Canonical Artifact Integrity Audit

Verified bit-exact against official release records:

| Canonical Artifact | Repository Path | Expected SHA-256 | Actual Observed SHA-256 | Verification Status |
|---|---|---|---|:---:|
| **Step 10,000 Checkpoint** | `experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt` | `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` | `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` | **BIT-EXACT MATCH** |
| **BPE Vocabulary** | `data/experiments/vocab_4096/vocab.json` | `db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a` | `db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a` | **BIT-EXACT MATCH** |
| **BPE Merges** | `data/experiments/vocab_4096/merges.json` | `ada4b1dcfbb3d962a2e60203089881872a5063de73b2fef9eeff81ce0016efa7` | `ada4b1dcfbb3d962a2e60203089881872a5063de73b2fef9eeff81ce0016efa7` | **BIT-EXACT MATCH** |
| **Model Core** | `chakrview/brain/model.py` | `1343d7d18224cc308465b7f07eb6d889532e7f17cee74b5f84f878e542f72361` | `1343d7d18224cc308465b7f07eb6d889532e7f17cee74b5f84f878e542f72361` | **BIT-EXACT MATCH** |
| **Runtime Session** | `chakrview/runtime/session.py` | `9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67` | `9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67` | **BIT-EXACT MATCH** |

- **Release Tag Target:** `refs/tags/v0.8.0` points to commit `ae96c645eb08c438440ffbe7546754341964a93e` (Zero modification).
- **Regression Tests:** `pytest -q tests/test_v060_release_smoke.py` $\to$ **7 passed in 2.20s (100% PASS)**.

---

## 3. Preregistration Invariance & Hash Verification

All pre-registered hashes recorded prior to training execution were verified bit-exact:
- **Exp31:** `c2ddd4c8408b352d6ab951d14561063f5df927f45a91243005a6ebb3cef077db`
- **Exp32:** `b47f2fd7b950513e6e1e7517e9c05243fed1fd34916bbaaf0a8eeb5b5b843ccf`
- **Exp33:** `f46862490cdbf9f63b9a93f3e532818dc168ea4d3e8e7b63f9c3971767078d7c`

---

## 4. Preservation & Storage Policy for Local Artifacts

In accordance with [`docs/development/EXPERIMENT_ARCHIVING_POLICY.md`](../development/EXPERIMENT_ARCHIVING_POLICY.md):
- **Tier 1 (Versioned in Git):** Full runnable source code, prompts, frozen dataset splits, evaluation metrics JSON, error traces, and markdown reports for Exp31–33 are safely synchronized on `main`.
- **Tier 2 (Preserved Locally):** Binary weights (`.pt`) and raw overnight run dumps (5.12 GB) remain preserved on local disk. They are omitted from `main` to keep repository clone size bounded for users, with all cryptographic hashes permanently tracked in versioned manifests.
- **Tier 3 (Excluded):** Ephemeral virtual environments (`.venv/`, `.venv_gpu/`) and python bytecode (`__pycache__/`) remain safely ignored.
