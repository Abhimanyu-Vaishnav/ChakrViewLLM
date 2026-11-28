# NR-057: REPOSITORY SAFETY & HARDWARE BASELINE

**Mission Identifier:** NR-057  
**Date:** November 28, 2026  
**Auditor:** ChakrView Engineering & Forensic Team  
**Scope:** Phase 0 repository safety verification, hardware environment profiling, and canonical checkpoint hash anchoring.

---

## 1. Executive Summary

Mission **NR-057** initiates the clean corpus expansion and training readiness engineering campaign for ChakrView. Repository safety was verified, confirming zero unstaged, uncommitted, or untracked changes on top of NR-056 commit `a00d754`.

- **Active Branch:** `exp/cbr-01-conversational-breakthrough`
- **Head Commit:** `a00d7543888ae185b37651a25fe38fa74780df6d` (`a00d754`)
- **Remote Synchronization:** Up to date with `origin/exp/cbr-01-conversational-breakthrough`.
- **Primary Disk (D:):** 83.1 GB free space available.
- **System Disk (C:):** 146.6 GB free space available.
- **Hardware Environment:** Intel Core i9-13900H (14C/20T) | 34.0 GB System RAM | NVIDIA RTX A1000 Laptop GPU (6.44 GB VRAM) | CUDA 12.6 | PyTorch 2.14.1+cu126.

---

## 2. Frozen Canonical Checkpoint & Tokenizer Registry

| Artifact | Relative Path | SHA-256 Hash |
| :--- | :--- | :--- |
| **Stage-D Canonical** | `experiments/genesis_stage_d_011/checkpoints/genesis_17m_staged_best.pt` | `2c21632ebae4f0b05b9ff7091797ecc956ac0ad00ef6ec934381223bc273b269` |
| **RC1 Release** | `release/v1.0.0-rc1/chakr_1_0_rc1_best.pt` | `cf585ebf9aa70f630516eec9c45e85f127e29ed91d6b6ae6a26d44fd20fcce4e` |
| **RC2 Candidate** | `experiments/genesis_stage_rc2_046/chakr_1_0_rc2_best.pt` | `33f93c75ba6c4ccd06a87ddb68ad300e11ce0f67aef45d6aa0a6ae7868c2be30` |
| **RC3 Milestone** | `experiments/fast_track_047_049/chakr_1_0_rc3_best.pt` | `61b39ae2bc55c116ed3ec8164497ec798d1fb6994a449d852248b266cd9dc73c` |
| **NR-052 17M Base** | `experiments/pretraining_nr052/chakr_17m_v4_base.pt` | `58781710bb4d85b40d77281eb6ccecb1754a3f3fadc468db8c1fa8b51a2dd062` |
| **NR-052 38M Base** | `experiments/pretraining_nr052/chakr_38m_v4_base.pt` | `7fb29b3519a957dbb4ffc12bdcc1cc50d6ea9bf49739458ebb9e3de031f504fb` |
| **NR-052 17M SFT** | `experiments/pretraining_nr052/chakr_17m_nr052_sft.pt` | `c13211bea859ab6ff8cb89c749b29ae57df638a85f2bef60908e23827fa5a4bb` |
| **NR-052 38M SFT** | `experiments/pretraining_nr052/chakr_38m_nr052_sft.pt` | `e08d363efdaac44cc62fa854ad1802de8ffb9de4060e090ad5373d66174c53ed` |
| **Tokenizer Vocab** | `data/experiments/vocab_4096/vocab.json` | `db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a` |
| **Tokenizer Merges**| `data/experiments/vocab_4096/merges.txt` | `961794b1a43a0d3a52bb6c8a77a98b4b1979b08537b01815e10cfb32eec6c3ea` |

All baseline hashes remain intact and unmodified.
