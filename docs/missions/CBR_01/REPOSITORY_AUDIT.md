# CBR-01 — WORKSTREAM A: REPOSITORY & CODE INTEGRITY AUDIT

**Audit Date:** 2026-11-04  
**Audit Scope:** Git history, working tree, canonical assets, branch structure, and protected cryptographic hashes.  

---

## 1. Git State & Branch Demarcation

- **Origin Remote:** `https://github.com/Abhimanyu-Vaishnav/ChakrViewLLM.git`
- **Initial HEAD:** `2b7c06f` (docs(release): publish complete v0.9.0 release documentation)
- **Active Experiment Branch:** `exp/cbr-01-conversational-breakthrough`
- **Protected Release Tag:** `v0.9.0` (pointing to commit `c8937a8`, immutable)

---

## 2. Cryptographic Protected File Audit

The canonical foundation files were hashed via SHA-256 prior to beginning operations:

| Component | Repository Path | Expected SHA-256 | Verified SHA-256 | Status |
| :--- | :--- | :--- | :--- | :---: |
| **Brain Architecture** | `chakrview/brain/model.py` | `1343d7d18224cc308465b7f07eb6d889532e7f17cee74b5f84f878e542f72361` | `1343d7d18224cc308465b7f07eb6d889532e7f17cee74b5f84f878e542f72361` | **PASS** |
| **Runtime Session** | `chakrview/runtime/session.py` | `9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67` | `9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67` | **PASS** |
| **Canonical Checkpoint**| `experiments/batch16_21_combined/.../training_checkpoint_step_10000.pt` | `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` | `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` | **PASS** |
| **User `.gitignore`** | `.gitignore` | `700f3d8b6d8c5c216884f1808f5d5c7da17282ce4d69f6e5d563f2ee49e4fda6` | `700f3d8b6d8c5c216884f1808f5d5c7da17282ce4d69f6e5d563f2ee49e4fda6` | **PASS** |
| **Extraction Decision**| `docs/experiments/CAND_01_STAGE_C_EXTRACTION_AUTHORIZATION_DECISION.md` | `e5b433adebb53b45e8250f07cae38282e1a06839bed15fa788d16d4db5192f00` | `e5b433adebb53b45e8250f07cae38282e1a06839bed15fa788d16d4db5192f00` | **PASS** |

---

## 3. Working Tree Invariants

- Unstaged user changes in `.gitignore` are preserved intact.
- Uncommitted local scripts (`scripts/generate_dataset_categories_*.py`) remain untouched.
- `experiments/overnight_runs/` remains completely excluded and unaccessed.
