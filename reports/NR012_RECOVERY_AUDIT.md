# CHAKRVIEW RESEARCH REPORT: NR-012 RECOVERY AUDIT
**Mission ID:** CV-GENESIS-NR-012  
**Date:** 2026-11-25  
**Repository:** `D:\Project\ChakrView`  
**Starting Commit:** `3ae648c9b5f9d40690c26952d2aac47d7d19ef04`  
**Branch:** `exp/cbr-01-conversational-breakthrough`  

---

## 1. Executive Summary & Audit Purpose
Before initiating development of the ChakrView Interactive Runtime, CLI, Local Web UI, and Capability Verification under Mission `CV-GENESIS-NR-012`, this audit cryptographically validates the exact baseline state of all canonical, pretraining, fine-tuning, and tokenizer artifacts.

All baseline checkpoints and canonical tokenizers are present, completely undamaged, and match their historical SHA-256 signatures bit-for-bit.

---

## 2. Checkpoint & Artifact Ledger

| Artifact Identifier | Local File Path | Size (Bytes) | Cryptographic SHA-256 Hash | Integrity Status |
| :--- | :--- | :--- | :--- | :--- |
| **Canonical Vocab (4,096)** | `data/experiments/vocab_4096/vocab.json` | 112,627 | `db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a` | Verified |
| **Canonical Pretrain Best (5.18M)** | `experiments/genesis_canonical_pretraining_004/checkpoints/canonical_pretrain_best.pt` | 20,733,723 | `03a4c16d20ad362114e92e928b7b11117c9d55345af10a580b01c1b2be5c229e` | Verified |
| **NR-005 SFT Best (5.18M)** | `experiments/genesis_conversational_foundation_005/checkpoints/genesis_sft_best.pt` | 62,203,157 | `ee1b91e848db3ca6e25008724f9f4fe200f217b4707ef81964395676da3d4269` | Verified |
| **NR-008 Balanced Best (5.18M)** | `experiments/genesis_balanced_sft_008/checkpoints/balanced_sft_best.pt` | 20,733,159 | `776a572501323967c7dd26d11a9f236d4535b6d3b71a1ad8ba5aa1b973890da0` | Verified |
| **NR-010 17M Pretrain Best** | `experiments/genesis_pretraining_010/checkpoints/genesis_17m_pretrain_best.pt` | 68,292,851 | `96219f224e7c4a7909254a5b7f0fc33a649853ffa4264c576b0852d3caeed7c0` | Verified |
| **NR-010 17M SFT Best** | `experiments/genesis_pretraining_010/sft_checkpoints/genesis_17m_sft_best.pt` | 68,292,297 | `ac62c43fda2e33acb3a24cbdad1038d104d8f2a8653ac39bd6c081606aac56dd` | Verified |
| **NR-011 17M Stage-D Best** | `experiments/genesis_stage_d_011/checkpoints/genesis_17m_staged_best.pt` | 68,292,655 | `2c21632ebae4f0b05b9ff7091797ecc956ac0ad00ef6ec934381223bc273b269` | Verified |
| **NR-011 17M Stage-D Final** | `experiments/genesis_stage_d_011/checkpoints/genesis_17m_staged_final.pt` | 68,292,753 | `bcca0c323ee2fc5c3f159ed5588af8fd195c11b1f76663f2d2d6d69c2274c2fe` | Verified |

---

## 3. Git Working Tree State
- Branch: `exp/cbr-01-conversational-breakthrough`
- Upstream Status: Synced cleanly with `origin/exp/cbr-01-conversational-breakthrough`
- Working directory: Clean before report generation
- Zero modifications or deletions of historical assets

## 4. Hardware Environment Verification
- Host CPU: Intel Core i9-13900H (14 cores, 20 logical threads)
- Host RAM: 32 GB DDR5
- Accelerator: NVIDIA RTX A1000 6GB Laptop GPU (6,144 MB VRAM, CUDA 12.8 compute support)
- PyTorch Version: 2.6.0+cu126

**Audit Verdict:** PASSED. All baseline conditions confirmed intact. Authorized to proceed with CV-GENESIS-NR-012 execution.
