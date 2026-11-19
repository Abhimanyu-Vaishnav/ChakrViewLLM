# UCIP-12R: Artifact Forensic Inventory

**Mission ID:** UCIP-12R — Forensic Verification  
**Repository:** `D:\Project\ChakrView`  
**Active Branch:** `exp/cbr-01-conversational-breakthrough`  
**HEAD Commit:** `f494c0c` (`feat(ucip12): complete large-scale corpus foundation and learning readiness study`)  
**Audit Date:** 2026-11-19  

---

## 1. Executive Summary & Verification Objective

Phase A performed a comprehensive, cryptographic forensic inventory of all artifacts cited in the UCIP-12 report. The audit independently confirmed the existence, filesystem size, cryptographic SHA-256 hash, and provenance of each core file, ensuring that no claimed dataset or checkpoint was accepted on the basis of filename alone.

---

## 2. Protected Runtime & Canonical Artifacts

| Component | Filesystem Path | Size (Bytes) | Computed SHA-256 | Expected SHA-256 | Verification Status |
|---|---|---|---|---|---|
| **Protected Runtime** | `chakrview/runtime/session.py` | 50,477 | `9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67` | `9080ff51adba38d4...` | **VERIFIED (Frozen, $\Delta W = 0$)** |
| **UCIP-09 Baseline** | `releases/ucip_09_candidates/chakrview_ucip09_medium_model.pt` | 38,980,483 | `5513fefe878ca85a637aa8401b3bebf5b8e3ca93ab362e07722d629e34294499` | `5513fefe878ca85a...` | **VERIFIED** |
| **UCIP-09R Checkpoint** | `releases/ucip_09r_candidates/chakrview_ucip09r_medium_model.pt` | 38,980,483 | `e1b943222e70fce2bac1b198e656bfea294209de1291f0042a2ebd677ccef9f0` | `e1b943222e70fce2...` | **VERIFIED** |
| **UCIP-09S Pilot** | `experiments/ucip09s/checkpoints/ucip09s_balanced_pilot.pt` | 38,980,483 | `c99dcc0a229310ec682857349c06e1a46957b559a8d180a33a91ea05b225d830` | `c99dcc0a229310ec...` | **VERIFIED** |
| **UCIP-11 Condition C** | `experiments/ucip11/checkpoints/condition_c_sft/condition_c_sft_best.pt` | 41,749,427 | `4a1e6ffb35f6f91e1f950db8265777ebdb4880811db88a432fe2822af6a68456` | `4a1e6ffb35f6f91e...` | **VERIFIED** |
| **R2 Sealed Benchmark** | `experiments/ucip10e_r2/datasets/sealed/benchmark_r2_sealed.json` | 234,398 | `ef94076849dd5c61cdea291e3a2ede35fdda96f47e36c39e53d8cc54fb0f36ef` | `ef94076849dd5c61...` | **VERIFIED** |
| **UCIP-11 Sealed Suite** | `experiments/ucip11/benchmarks/sealed/ucip11_sealed.json` | 39,687 | `dce6954ab6db4aa00bfcf1405a47243d4698e9e0cd084ad63ff3e5281b678fe4` | `dce6954ab6db4aa0...` | **VERIFIED** |

---

## 3. Tokenizer Configuration & Merge Artifacts

| Component | Filesystem Path | Size (Bytes) | Computed SHA-256 | Verification Status |
|---|---|---|---|---|
| **Tokenizer Config** | `data/experiments/vocab_4096/tokenizer_config.json` | 647 | `f08960fa0f8c868eb2a243283259dd1a7e7ba0beebefc061aa63b655da08d1f2` | **VERIFIED** |
| **Vocabulary JSON** | `data/experiments/vocab_4096/vocab.json` | 89,842 | `e0f11c0795fa0ddd366f54a17355e5ceaaf7d12fc73881677dc122eaae6a999e` | **VERIFIED** |
| **Merges Text** | `data/experiments/vocab_4096/merges.txt` | 51,029 | `7498d92adeef7c9e64d4d1bbd5b4b23f8774692d868742e3b98e30c584c4403f` | **VERIFIED** |

---

## 4. Master Pretraining Corpus Manifest & Shard Files

| Component | Filesystem Path | Size (Bytes) | Key Attributes | Verification Status |
|---|---|---|---|---|
| **Stage C Manifest** | `data/manifests/stage_c_manifest.json` | 4,491,264 | 5,534 docs, 7,703,067 content tokens | **VERIFIED** |
| **Stage C Shards** | `data/tokenized/stage_c/**/*.bin` | 15,417,202 (32 files) | 7,708,601 uint16 tokens | **VERIFIED** |
| **Source Registry** | `experiments/ucip12/provenance/source_registry.json` | 12,245 | 13 sources tracked | **VERIFIED** |

---

## 5. Working-Tree Assessment

Git working-tree status prior to UCIP-12R execution:
- Branch: `exp/cbr-01-conversational-breakthrough`
- Clean working directory with respect to tracked files.
- Untracked files present: `experiments/cognition/ucip_05/GIT_RELEASE_RECORD.md`, `experiments/cognition/ucip_06/`, `releases/ucip_06_candidate/`. None of these interfere with UCIP-12 or Stage C.
