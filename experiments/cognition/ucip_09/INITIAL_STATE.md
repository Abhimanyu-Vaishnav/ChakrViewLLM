# UCIP-09: Pre-Flight Forensic Audit & Baseline Integrity Manifest

**Mission:** UCIP-09 — Neural Language Intelligence Recovery & Architecture Evolution  
**Project:** ChakrView  
**Repository:** `D:\Project\ChakrView`  
**Active Branch:** `exp/cbr-01-conversational-breakthrough`  
**HEAD Commit:** `9b71ce1` (feat(cognition): complete UCIP-08 forensic audit and neural capability recovery)  
**Date:** November 17, 2026  

---

## 1. System & Execution Environment

- **Python Version:** 3.14.7 (64-bit AMD64)
- **PyTorch Version:** 2.14.0+cpu
- **Host OS:** Windows 11 Pro (10.0.26300-SP0)
- **Hardware Architecture:** Intel x86_64, GenuineIntel (i9 13th-Gen, 32 GB RAM)
- **Execution Mode:** 100% CPU-First Indigenous Compute (Zero GPU, Zero external APIs, Zero external pretrained weights)

---

## 2. Protected Artifact & Canonical Invariant Verification

| Artifact Name | Path | Expected SHA-256 | Actual SHA-256 | Verification Status |
|---|---|---|---|---|
| **Protected Runtime (`session.py`)** | `chakrview/runtime/session.py` | `9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67` | `9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67` | **100% BIT-EXACT MATCH** |
| **Canonical Baseline Init (`seed=42`)** | `ChakrMicro(seed=42)` (3,443,136 params) | `c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da` | `c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da` | **100% BIT-EXACT MATCH** |
| **Canonical Stage B (Step 10,000)** | `experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt` | `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` | `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` | **100% BIT-EXACT MATCH** |
| **Frozen Owner Reality 100** | `experiments/cognition/ucip_08/sealed_owner_reality_100.json` | `7607eaf4635232822cdd0b1faad15543664e7c7b339ecd0fda5a7a8e3766ae15` | `7607eaf4635232822cdd0b1faad15543664e7c7b339ecd0fda5a7a8e3766ae15` (embedded `sha256` field) | **100% BIT-EXACT MATCH** |

---

## 3. Parent Candidate Model Checkpoint Verification

| Checkpoint Name | Checkpoint Path | Architecture | Total Parameters | Verified SHA-256 Digest | Status |
|---|---|---|---|---|---|
| **UCIP-06 Medium** | `releases/ucip_06_candidate/chakrview_ucip06_medium_model.pt` | ChakrMedium | 12,932,736 | `be843bc2cbc08f3f8459b85d6eec2cdce534441ac5ccc797f0bc0d2be8aa1b72` | **VERIFIED** |
| **UCIP-07 Medium** | `releases/ucip_07_candidates/chakrview_ucip07_medium_model.pt` | ChakrMedium | 12,932,736 | `554c503e058311d8732216a3ad200cf1c7d60a98baa39f78c26328c66516d5b2` | **VERIFIED** |
| **UCIP-08 Medium** | `releases/ucip_08_candidates/chakrview_ucip08_medium_model.pt` | ChakrMedium | 12,932,736 | `fe6f381acf05662bcc6570e2ae4a06f3ecc89b669811e5340eebd84bb014f631` | **VERIFIED** |

---

## 4. Tokenizer Artifact Integrity

- **Artifact Directory:** `data/experiments/vocab_4096`
- **Class:** `BPETokenizer` (`chakrview/tokenizer/tokenizer.py`)
- **Vocab Size:** 4,096 total slots (4,093 active entries + 3 special tokens: `BOS_ID=0`, `EOS_ID=1`, `PAD_ID=2`)
- **Learned Merges:** 3,837 byte-level merges
- **Files & Hashes:**
  - `config.json`: `0fca72b6a70e9ae812589199c661ef6be20cb9973f06353db125a1ec7264e589`
  - `merges.json`: `ada4b1dcfbb3d962a2e60203089881872a5063de73b2fef9eeff81ce0016efa7`
  - `vocab.json`: `db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a`

---

## 5. Working Tree & Git Hygiene

- **Untracked Directories:**
  - `experiments/cognition/ucip_05/GIT_RELEASE_RECORD.md`
  - `experiments/cognition/ucip_06/`
  - `releases/ucip_06_candidate/`
- **Current Test Suite Status:**
  - `pytest tests/test_architecture_loader.py tests/test_inference_session.py tests/test_ucip07r_loader.py` -> **22 passed in 9.11s (100% PASS)**
