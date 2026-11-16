# UCIP-04: Phase Zero — Initial State & Repository Preservation

**Mission ID:** UCIP-04  
**Project:** ChakrView  
**Repository:** `D:\Project\ChakrView`  
**Active Branch:** `exp/cbr-01-conversational-breakthrough`  
**Starting Commit:** `51714cd` (`audit: independently verify UCIP-03 neural generation benchmarks and root cause`)  
**Date:** 2026-11-16  
**Auditor / Lead Engineer:** Autonomous Conversational Recovery Agent  

---

## 1. Git Repository Invariants

- **Branch:** `exp/cbr-01-conversational-breakthrough`
- **Head Commit:** `51714cd`
- **Remote Origin:** `https://github.com/Abhimanyu-Vaishnav/ChakrViewLLM.git`
- **Working Tree:** Clean, zero uncommitted modifications.

---

## 2. Checkpoint & Protected Artifact Verification

| Item | Path | Size (Bytes) | Authoritative SHA-256 | Verification Status |
| :--- | :--- | :--- | :--- | :--- |
| **Protected Runtime** | `chakrview/runtime/session.py` | 12,511 | `9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67` | **PASS (UNTOUCHED)** |
| **Canonical Stage B** | `experiments/batch16_21_combined/.../training_checkpoint_step_10000.pt` | 42,212,787 | `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` | **PASS (BIT-EXACT)** |
| **Parent Checkpoint (`cbr_sft_04`)** | `experiments/cbr01_sft/runs/cbr_sft_04/.../sft_checkpoint_step_200.pt` | 41,417,634 | `327118ce7057af868d38c64ce3df35dba3b85b820b8022e18cb5be1f86cfa153` | **PASS (BIT-EXACT)** |
| **v0.11 Release Model** | `releases/v1.0_unified_conversational_intelligence/chakrview_v011_model.pt` | 41,388,167 | `5ae98b369b08373df1222364d1b6165493b34287baf297ba32320c16b76f2754` | **PASS (BIT-EXACT)** |
| **13M Recovered Candidate** | `experiments/cognition/ucip_01r/training/ucip01r_13m_recovered_best.pt` | 51,757,989 | `19f248afbd5502372ccbb064d0d53fa41b362f391a2ad8f99e944fe0a036343d` | **PASS (BIT-EXACT)** |
| **UCIP-03 Released Checkpoint** | `releases/ucip_03_conversation_candidate/chakrview_ucip03_model.pt` | 13,795,053 | `b27313593e2b323781d65feeb8d4d7318fbfabe2eb53f56230bdf126c23f20c6` | **PASS (BIT-EXACT)** |
| **Canonical Tokenizer Vocab** | `releases/ucip_03_conversation_candidate/vocab.json` | 112,627 | `db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a` | **PASS (BIT-EXACT)** |
| **Canonical Tokenizer Merges** | `releases/ucip_03_conversation_candidate/merges.json` | 76,299 | `ada4b1dcfbb3d962a2e60203089881872a5063de73b2fef9eeff81ce0016efa7` | **PASS (BIT-EXACT)** |

---

## 3. Preserved UCIP-03R Insights
1. UCIP-03 benchmark awarded 100% credit for `len(gen_text) > 3`, which masked word salad generation.
2. Teacher-forced token accuracy (65.31%) was fundamentally dissociated from autoregressive sequence accuracy (2.24%).
3. The model formed an associative attractor basin blending unigrams from training proverbs into incoherent phrases like `"The planned of the bold."`.
4. UCIP-04 will establish genuine autoregressive generation evaluated at every training checkpoint using greedy rollouts and independent task-specific rubrics.
