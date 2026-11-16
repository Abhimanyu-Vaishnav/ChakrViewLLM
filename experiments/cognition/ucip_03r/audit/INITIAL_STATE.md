# UCIP-03R: Phase Zero — Initial State & Preservation Record

**Mission ID:** UCIP-03R  
**Project:** ChakrView  
**Repository:** `D:\Project\ChakrView`  
**Active Branch:** `exp/cbr-01-conversational-breakthrough`  
**Starting Commit:** `de2f512` (`feat(ucip-03): foundational neural language acquisition, controlled training and sealed evaluation`)  
**Date:** 2026-11-16  
**Auditor:** Autonomous Independent Verification Agent  

---

## 1. Git Repository State

- **Branch:** `exp/cbr-01-conversational-breakthrough` (Up to date with `origin/exp/cbr-01-conversational-breakthrough`)
- **Working Tree:** Clean at commit `de2f512`
- **Remote Origin:** `https://github.com/Abhimanyu-Vaishnav/ChakrViewLLM.git`

---

## 2. Cryptographic Integrity of Checkpoints & Protected Files

| Item | Target Path | File Size | Authoritative SHA-256 Digest | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Protected File** | `chakrview/runtime/session.py` | 12,511 bytes | `9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67` | **PASS (UNMODIFIED)** |
| **UCIP-03 Released Checkpoint** | `releases/ucip_03_conversation_candidate/chakrview_ucip03_model.pt` | 13,795,053 bytes | `b27313593e2b323781d65feeb8d4d7318fbfabe2eb53f56230bdf126c23f20c6` | **PASS (BIT-EXACT)** |
| **UCIP-03 Released Vocab** | `releases/ucip_03_conversation_candidate/vocab.json` | 112,627 bytes | `db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a` | **PASS (BIT-EXACT)** |
| **UCIP-03 Released Merges** | `releases/ucip_03_conversation_candidate/merges.json` | 76,299 bytes | `ada4b1dcfbb3d962a2e60203089881872a5063de73b2fef9eeff81ce0016efa7` | **PASS (BIT-EXACT)** |
| **UCIP-03 Released Config** | `releases/ucip_03_conversation_candidate/config.json` | 263 bytes | `e2a705fbc4f38476894e11712a65e16bbe4ed8e10c0f1615246bf65f6173bf47` | **PASS (BIT-EXACT)** |
| **v0.11 Release Model** | `releases/v1.0_unified_conversational_intelligence/chakrview_v011_model.pt` | 41,388,167 bytes | `5ae98b369b08373df1222364d1b6165493b34287baf297ba32320c16b76f2754` | **PASS (VERIFIED)** |
| **13M Recovered Best** | `experiments/cognition/ucip_01r/training/ucip01r_13m_recovered_best.pt` | 51,757,989 bytes | `19f248afbd5502372ccbb064d0d53fa41b362f391a2ad8f99e944fe0a036343d` | **PASS (VERIFIED)** |
| **UCIP-03 Micro Best** | `experiments/cognition/ucip_03/training/micro/candidate_best.pt` | 13,795,053 bytes | `b27313593e2b323781d65feeb8d4d7318fbfabe2eb53f56230bdf126c23f20c6` | **PASS (IDENTICAL TO RELEASE)** |
| **UCIP-03 Medium Best** | `experiments/cognition/ucip_03/training/medium/candidate_best.pt` | 51,757,073 bytes | `204f91bb292309cda37300059a4555ec99373b5ff8f801e2035e75bd0eeaab22` | **PASS (VERIFIED)** |

---

## 3. Preservation Commitments

1. No retraining or fine-tuning of any neural model.
2. No modification of model weights, tokenizer vocabularies, or merge tables.
3. No replacement of release candidate artifacts.
4. Protected file `chakrview/runtime/session.py` will remain untouched.
5. All audit findings will be documented with raw execution logs and deterministic reproduction scripts.
