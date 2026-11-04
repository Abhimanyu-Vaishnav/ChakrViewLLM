# CBR-SFT-05 PHASE 2 PREFLIGHT REPORT

**Date:** 2026-11-04  
**Repository Root:** `D:\Project\ChakrView`  
**Active Branch:** `exp/cbr-01-conversational-breakthrough`  
**Current HEAD Commit:** `257801b4c0fcc3017bb450b104baee5971ad530a`  

---

## 1. Protected Artifacts & Parent Checkpoint Integrity

All non-negotiable safety contracts were validated using bit-exact cryptographic hashing:

| Artifact / Checkpoint | Expected SHA-256 | Actual SHA-256 | Verification Result |
| :--- | :--- | :--- | :---: |
| `chakrview/brain/model.py` | `1343d7d18224cc308465b7f07eb6d889532e7f17cee74b5f84f878e542f72361` | `1343d7d18224cc308465b7f07eb6d889532e7f17cee74b5f84f878e542f72361` | **PASS** |
| `chakrview/runtime/session.py` | `9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67` | `9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67` | **PASS** |
| Canonical Step 10k Checkpoint | `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` | `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` | **PASS** |
| CBR-SFT-02 Step 50 Checkpoint | `592553c3e27b654997e8f5df61e94fe1de25bbf577391df6c1a87f974730bf9d` | `592553c3e27b654997e8f5df61e94fe1de25bbf577391df6c1a87f974730bf9d` | **PASS** |
| CBR-SFT-03B Step 200 Checkpoint | `297dcc60f39cadd6bae9a5bd969b1a7d831845d7a67823b30c194a67ef75b5ac` | `297dcc60f39cadd6bae9a5bd969b1a7d831845d7a67823b30c194a67ef75b5ac` | **PASS** |
| CBR-SFT-03B Step 300 Alternate | `e540bd2364113d7eaf271c3495d882921f4421e59659f19f06cd7d590f74710a` | `e540bd2364113d7eaf271c3495d882921f4421e59659f19f06cd7d590f74710a` | **PASS** |
| Canonical Random-Init Weight Hash | `c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da` | `c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da` | **PASS** |
| **Parent CBR-SFT-04 Step 200** | `327118ce7057af868d38c64ce3df35dba3b85b820b8022e18cb5be1f86cfa153` | `327118ce7057af868d38c64ce3df35dba3b85b820b8022e18cb5be1f86cfa153` | **PASS** |

---

## 2. Working Tree State & Uncommitted Files

`git status --short` confirms the exact set of pre-existing modified documentation files from prior CBR-01 missions and untracked Phase 1/evaluation files:
- Modified files: 7 documentation files under `docs/missions/CBR_01/experiments/CBR_SFT_03B/` (preserved untouched).
- Untracked directories:
  - `data/sft/cbr_sft_05/` (Phase 1 dataset)
  - `evaluation/cbr_sft_05_suite/` (Phase 1 evaluation suite)
  - `data/sft/cbr_sft_04/` and `evaluation/cbr_sft_04_suite/` (CBR-SFT-04 artifacts)
  - `experiments/cand01/`, `experiments/overnight_runs/` (preserved untouched)
- Zero untracked files deleted; zero git reset/clean operations executed.

---

## 3. Preflight Status Verdict

### Verdict: **`PREFLIGHT PASS — READY FOR AUDIT`**
