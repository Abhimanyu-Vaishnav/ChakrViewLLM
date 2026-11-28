# NR-056: REPOSITORY STATE & ARTIFACT SNAPSHOT

**Mission Identifier:** NR-056  
**Date:** November 28, 2026  
**Auditor:** ChakrView Independent Forensic Engineering Team  
**Scope:** Phase 0 repository integrity verification, commit tree validation, remote synchronization, and baseline artifact SHA-256 recording.

---

## 1. Executive Summary

Prior to initiating the NR-056 audit, repository safety was established. Working tree clean status and local-to-remote synchronization were verified with zero untracked modifications or uncommitted changes.

- **Active Branch:** `exp/cbr-01-conversational-breakthrough`
- **Remote URL:** `https://github.com/Abhimanyu-Vaishnav/ChakrViewLLM.git`
- **Verified Starting Head Commit:** `7429bf764f61f9dfa0ae58c89c8942b826b1c85d` (`7429bf7`)
- **Remote Synchronization:** Head commit `7429bf7` is confirmed present on both local and remote branches (`origin/exp/cbr-01-conversational-breakthrough`).

---

## 2. Key Artifact Inventory & Hash Registry

| Artifact Identifier | Canonical Relative Path | Size (Bytes) | SHA-256 Hash |
| :--- | :--- | :---: | :--- |
| **Tokenizer Vocab** | `data/experiments/vocab_4096/vocab.json` | 63,334 | `db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a` |
| **Tokenizer Merges** | `data/experiments/vocab_4096/merges.txt` | 51,079 | `961794b1a43a0d3a52bb6c8a77a98b4b1979b08537b01815e10cfb32eec6c3ea` |
| **Benchmark V2 JSON** | `data/evaluation/benchmark_v2/benchmark_v2_100.json` | 33,639 | `84c9c57d7689aa81f08cb3892f3c0516be702117a26f633d2629bbf7bfb0853d` |
| **Benchmark V2 Manifest** | `data/manifests/benchmark_v2_manifest.json` | 540 | `1a646d655f444cfa06b64b1ea95b5420e6fbb6b69cfc7ff8bfa9df6d7950c058` |
| **Stage-D Checkpoint** | `experiments/genesis_stage_d_011/checkpoints/genesis_17m_staged_best.pt` | 68,292,101 | `2c21632ebae4f0b05b9ff7091797ecc956ac0ad00ef6ec934381223bc273b269` |
| **RC1 Checkpoint** | `release/v1.0.0-rc1/chakr_1_0_rc1_best.pt` | 68,292,101 | `cf585ebf9aa70f630516eec9c45e85f127e29ed91d6b6ae6a26d44fd20fcce4e` |
| **RC2 Checkpoint** | `experiments/genesis_stage_rc2_046/chakr_1_0_rc2_best.pt` | 68,292,101 | `33f93c75ba6c4ccd06a87ddb68ad300e11ce0f67aef45d6aa0a6ae7868c2be30` |
| **RC3 Checkpoint** | `experiments/fast_track_047_049/chakr_1_0_rc3_best.pt` | 68,292,101 | `61b39ae2bc55c116ed3ec8164497ec798d1fb6994a449d852248b266cd9dc73c` |
| **NR051 17M Base** | `experiments/pretraining_nr051/chakr_17m_pretrained_base.pt` | 68,292,101 | `8d871ca0aff6e23398bdb17f86521eb26dfb4b778d227035d77e2fe44803c25f` |
| **NR051 38M Base** | `experiments/pretraining_nr051/chakr_38m_pretrained_base.pt` | 150,042,671 | `22306c9a479c80cd9ce3e937e3145fb1520241d96c3b17b3ff8aed2af4d7f304` |
| **NR051 17M SFT** | `experiments/pretraining_nr051/chakr_17m_sft_candidate.pt` | 68,292,101 | `9f63e097108f4c3ef2fdc882826d570845a101d02f036bc1c055ad66183600b4` |
| **NR051 38M SFT** | `experiments/pretraining_nr051/chakr_38m_sft_candidate.pt` | 150,042,671 | `a0639a3fba90352c47b151fb70e60209a1889cd3265b80c3cadcb751e3c5b31f` |
| **NR052 17M Base** | `experiments/pretraining_nr052/chakr_17m_v4_base.pt` | 68,292,101 | `58781710bb4d85b40d77281eb6ccecb1754a3f3fadc468db8c1fa8b51a2dd062` |
| **NR052 38M Base** | `experiments/pretraining_nr052/chakr_38m_v4_base.pt` | 150,042,671 | `7fb29b3519a957dbb4ffc12bdcc1cc50d6ea9bf49739458ebb9e3de031f504fb` |
| **NR052 17M SFT** | `experiments/pretraining_nr052/chakr_17m_nr052_sft.pt` | 68,292,101 | `c13211bea859ab6ff8cb89c749b29ae57df638a85f2bef60908e23827fa5a4bb` |
| **NR052 38M SFT** | `experiments/pretraining_nr052/chakr_38m_nr052_sft.pt` | 150,042,671 | `e08d363efdaac44cc62fa854ad1802de8ffb9de4060e090ad5373d66174c53ed` |

---

## 3. Safety Clearance

No modifications will be made to any of the checkpoints, corpora, or tokenizer files. All audit experiments will write exclusively to `experiments/forensics_nr056/`, `reports/`, and isolated evaluation directories.
