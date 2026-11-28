# NR-056: CHECKPOINT PROVENANCE & LINEAGE AUDIT

**Mission Identifier:** NR-056  
**Date:** November 28, 2026  
**Auditor:** ChakrView Independent Forensic Engineering Team  
**Scope:** Verification of parent checkpoint lineage, training scripts, corpora, and state dictionary metadata.

---

## 1. Executive Summary

Every primary candidate checkpoint was verified against its root initialization and training logs. Machine-readable lineage is recorded in `experiments/forensics_nr056/checkpoint_lineage.json`.

---

## 2. Checkpoint Provenance Registry

| Checkpoint Identifier | Canonical Repository Path | Parent Checkpoint | Training Corpus | Driver Script | Architecture Family | SHA-256 Hash |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Stage-D Canonical** | `experiments/genesis_stage_d_011/checkpoints/genesis_17m_staged_best.pt` | `genesis_17m_pretrain_best.pt` (Scratch 42) | Corpus V1 | `stage_d_qa_corpus_pipeline.py` | ChakrGenesis17M | `2c21632ebae4f0b05b9ff7091797ecc956ac0ad00ef6ec934381223bc273b269` |
| **RC1 Release** | `release/v1.0.0-rc1/chakr_1_0_rc1_best.pt` | `genesis_17m_staged_best.pt` | Corpus V1 (Unmasked) | `train_first_model.py` | ChakrGenesis17M | `cf585ebf9aa70f630516eec9c45e85f127e29ed91d6b6ae6a26d44fd20fcce4e` |
| **RC2 Experimental** | `experiments/genesis_stage_rc2_046/chakr_1_0_rc2_best.pt` | `genesis_17m_staged_best.pt` | Corpus V2 (Masked) | `train_rc2_sft.py` | ChakrGenesis17M | `33f93c75ba6c4ccd06a87ddb68ad300e11ce0f67aef45d6aa0a6ae7868c2be30` |
| **RC3 Baseline** | `experiments/fast_track_047_049/chakr_1_0_rc3_best.pt` | `chakr_1_0_rc2_best.pt` | Corpus V3 | `train_rc3_fast_track.py` | ChakrGenesis17M | `61b39ae2bc55c116ed3ec8164497ec798d1fb6994a449d852248b266cd9dc73c` |
| **NR051 17M Base** | `experiments/pretraining_nr051/chakr_17m_pretrained_base.pt` | Random Init (Seed 42) | Corpus V3 Pretrain | `run_controlled_pretraining_campaign.py` | ChakrGenesis17M | `8d871ca0aff6e23398bdb17f86521eb26dfb4b778d227035d77e2fe44803c25f` |
| **NR051 38M Base** | `experiments/pretraining_nr051/chakr_38m_pretrained_base.pt` | Random Init (Seed 42) | Corpus V3 Pretrain | `train_38m_pretrain.py` | ChakrGenesis38M | `22306c9a479c80cd9ce3e937e3145fb1520241d96c3b17b3ff8aed2af4d7f304` |
| **NR051 17M SFT** | `experiments/pretraining_nr051/chakr_17m_sft_candidate.pt` | `chakr_17m_pretrained_base.pt` | Corpus V3 SFT | `run_controlled_sft_campaign.py` | ChakrGenesis17M | `9f63e097108f4c3ef2fdc882826d570845a101d02f036bc1c055ad66183600b4` |
| **NR051 38M SFT** | `experiments/pretraining_nr051/chakr_38m_sft_candidate.pt` | `chakr_38m_pretrained_base.pt` | Corpus V3 SFT | `run_controlled_sft_campaign.py` | ChakrGenesis38M | `a0639a3fba90352c47b151fb70e60209a1889cd3265b80c3cadcb751e3c5b31f` |
| **NR052 17M Base** | `experiments/pretraining_nr052/chakr_17m_v4_base.pt` | Random Init (Seed 42) | Corpus V4 Pretrain | `run_controlled_pretraining_v4.py` | ChakrGenesis17M | `58781710bb4d85b40d77281eb6ccecb1754a3f3fadc468db8c1fa8b51a2dd062` |
| **NR052 38M Base** | `experiments/pretraining_nr052/chakr_38m_v4_base.pt` | Random Init (Seed 42) | Corpus V4 Pretrain | `run_controlled_pretraining_v4.py` | ChakrGenesis38M | `7fb29b3519a957dbb4ffc12bdcc1cc50d6ea9bf49739458ebb9e3de031f504fb` |
| **NR052 17M SFT** | `experiments/pretraining_nr052/chakr_17m_nr052_sft.pt` | `chakr_17m_v4_base.pt` | Corpus V4 SFT | `run_controlled_sft_campaign.py` | ChakrGenesis17M | `c13211bea859ab6ff8cb89c749b29ae57df638a85f2bef60908e23827fa5a4bb` |
| **NR052 38M SFT** | `experiments/pretraining_nr052/chakr_38m_nr052_sft.pt` | `chakr_38m_v4_base.pt` | Corpus V4 SFT | `run_controlled_sft_campaign.py` | ChakrGenesis38M | `e08d363efdaac44cc62fa854ad1802de8ffb9de4060e090ad5373d66174c53ed` |

---

## 3. Machine-Readable Lineage Export

The complete JSON structure with parent pointers, script references, training configurations, and exact file hashes has been recorded in `experiments/forensics_nr056/checkpoint_lineage.json`.
