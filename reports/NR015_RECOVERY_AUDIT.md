# NR-015 Phase 0 — Recovery & Baseline Artifact Audit
**Mission ID:** CV-GENESIS-NR-015  
**Date:** 2026-11-25  
**Repository:** `D:\Project\ChakrView`  
**Active Branch:** `exp/cbr-01-conversational-breakthrough`  
**Starting Commit:** `5b88dd1` (CV-GENESIS-NR-014: causal LM objective forensic unit testing, expanded coding dataset, Stage-F balanced training, and 260-task generalization evaluation)  
**Remote Sync Status:** Verified clean, 0 ahead, 0 behind `origin/exp/cbr-01-conversational-breakthrough`.

---

## 1. Executive Summary & Audit Mandate
In accordance with CV-GENESIS-NR-015 Phase 0, a forensic audit was executed on repository status, Git tracking, commit lineage, canonical and experimental checkpoint cryptographic signatures, tokenizer assets, and evaluation artifacts.

All canonical and experimental checkpoints were verified bit-for-bit without corruption or missing components.

---

## 2. Cryptographic Checkpoint Inventory

| Checkpoint Name | Relative Path | File Size | SHA-256 Checksum | Lineage / Status |
| :--- | :--- | :--- | :--- | :--- |
| **Canonical Pretrained (5.18M)** | `experiments/project_genesis/pretraining/genesis_pretrain_best.pt` | 59.32 MB | `c5cf8becb87b3d9439861bf66736d18134edbc74322f77a3dfc35c8f8c129c9c` | Canonical 5.18M baseline |
| **NR-005 SFT Best (5.18M)** | `experiments/genesis_conversational_foundation_005/checkpoints/genesis_sft_best.pt` | 59.32 MB | `ee1b91e848db3ca6e25008724f9f4fe200f217b4707ef81964395676da3d4269` | Conversational foundation |
| **NR-010 Pretrain Best (17M)** | `experiments/genesis_pretraining_010/checkpoints/genesis_17m_pretrain_best.pt` | 65.13 MB | `96219f224e7c4a7909254a5b7f0fc33a649853ffa4264c576b0852d3caeed7c0` | 17M Pretrained foundation |
| **NR-010 SFT Best (17M)** | `experiments/genesis_pretraining_010/sft_checkpoints/genesis_17m_sft_best.pt` | 65.13 MB | `ac62c43fda2e33acb3a24cbdad1038d104d8f2a8653ac39bd6c081606aac56dd` | 17M SFT foundation |
| **ChakrGenesis-17M Stage-D Best** | `experiments/genesis_stage_d_011/checkpoints/genesis_17m_staged_best.pt` | 65.13 MB | `2c21632ebae4f0b05b9ff7091797ecc956ac0ad00ef6ec934381223bc273b269` | **PRIMARY CANONICAL FOUNDATION** |
| **ChakrGenesis-17M Stage-E Best** | `experiments/genesis_stage_e_013/checkpoints/genesis_17m_stage_e_best.pt` | 65.13 MB | `0ff238b6829ebd638bbb8cdefded86616c5e76c0180946e69da74a77ba3ac630` | Archived Experimental (NR-013) |
| **ChakrGenesis-17M Stage-F Best** | `experiments/genesis_stage_f_014/checkpoints/genesis_17m_stage_f_best.pt` | 65.13 MB | `454aa58af12cb915ecf9bf144366489eb56dcfa088e0845574bc24fcca6f6ab8` | Archived Experimental (NR-014) |

---

## 3. Production Tokenizer Artifacts (`data/experiments/vocab_4096`)

| File | File Size | SHA-256 Checksum | Status |
| :--- | :--- | :--- | :--- |
| `vocab.json` | 112,627 bytes | `db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a` | Verified Canonical Vocab |
| `merges.json` | 76,299 bytes | `ada4b1dcfbb3d962a2e60203089881872a5063de73b2fef9eeff81ce0016efa7` | Verified Canonical Merges |
| `config.json` | 1,130 bytes | `0fca72b6a70e9ae812589199c661ef6be20cb9973f06353db125a1ec7264e589` | Verified Config (`vocab_size=4096`) |

---

## 4. Evaluation and Dataset Integrity
- **NR-014 260-Task Benchmark:** `data/sealed_eval_014/sealed_benchmark_260.json` (62,118 bytes, SHA-256: `4520496abc10acb9f677485df244ab5d81e9d80f3cb4a5de9c18a82fd76e8beb`). Verified intact.
- **Git LFS Status:** Clean, zero untracked or dangling large objects.
- **Working Tree:** Pristine, ready for Phase 1.
