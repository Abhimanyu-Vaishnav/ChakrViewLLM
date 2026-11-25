# NR-015 Phase 10 — Artifact Inventory & Verification Signatures
**Mission ID:** CV-GENESIS-NR-015  
**Date:** 2026-11-25  
**Repository:** `D:\Project\ChakrView`  
**Active Branch:** `exp/cbr-01-conversational-breakthrough`  

---

## 1. Checkpoint Cryptographic Signatures

| Artifact Name | Path | Size | SHA-256 Checksum | Classification |
| :--- | :--- | :--- | :--- | :--- |
| **Stage-D Best** | `experiments/genesis_stage_d_011/checkpoints/genesis_17m_staged_best.pt` | 65.13 MB | `2c21632ebae4f0b05b9ff7091797ecc956ac0ad00ef6ec934381223bc273b269` | **PRIMARY CANONICAL** |
| **Stage-E Best** | `experiments/genesis_stage_e_013/checkpoints/genesis_17m_stage_e_best.pt` | 65.13 MB | `0ff238b6829ebd638bbb8cdefded86616c5e76c0180946e69da74a77ba3ac630` | Archived Experimental |
| **Stage-F Best** | `experiments/genesis_stage_f_014/checkpoints/genesis_17m_stage_f_best.pt` | 65.13 MB | `454aa58af12cb915ecf9bf144366489eb56dcfa088e0845574bc24fcca6f6ab8` | Archived Experimental |
| **NR-010 Pretrain Best** | `experiments/genesis_pretraining_010/checkpoints/genesis_17m_pretrain_best.pt` | 65.13 MB | `96219f224e7c4a7909254a5b7f0fc33a649853ffa4264c576b0852d3caeed7c0` | Historical Pretrain |
| **NR-005 SFT Best** | `experiments/genesis_conversational_foundation_005/checkpoints/genesis_sft_best.pt` | 59.32 MB | `ee1b91e848db3ca6e25008724f9f4fe200f217b4707ef81964395676da3d4269` | Historical 5.18M SFT |
| **Canonical Pretrained** | `experiments/project_genesis/pretraining/genesis_pretrain_best.pt` | 59.32 MB | `c5cf8becb87b3d9439861bf66736d18134edbc74322f77a3dfc35c8f8c129c9c` | Canonical 5.18M Base |

---

## 2. Experimental Data Suites & Benchmark Manifests

| Dataset / Benchmark File | Path | Tasks / Samples | SHA-256 Checksum | Purpose |
| :--- | :--- | :---: | :--- | :--- |
| **NR-015 Sealed Benchmark** | `data/sealed_eval_015/sealed_benchmark_300.json` | 325 tasks | Generated | 12-family generalization suite |
| **Synthetic Binding Train** | `data/synthetic_binding_015/train_binding.json` | 1,200 | `eee02b7f3db88c1d4506f6813be7e09709836a56de0cde2d3f0f1aec62883902` | Controlled variable binding |
| **Synthetic Binding Val (OOD)**| `data/synthetic_binding_015/val_binding.json` | 200 | `8cceaac3ee00964ef77a6c14bf0afcb2f02fbde815dfe57e2b43e1979f178c37` | Unseen variables & larger numbers |
| **Curriculum Datasets** | `data/curriculum_015/` | 1,000 | Multiple | Easy/Med/Hard staged progression |

---

## 3. Mission Research Reports (`reports/`)
- `NR015_RECOVERY_AUDIT.md`: Phase 0 recovery & artifact verification
- `NR015_LEARNING_FORENSICS.md`: Phase 1 causal invariants & controlled memorization
- `NR015_GENERALIZATION_BENCHMARK.md`: Phase 2 sealed 325-task inductive benchmark design
- `NR015_CONTROLLED_EXPERIMENT_REPORT.md`: Phase 3 synthetic binding pilot findings
- `NR015_REPRESENTATION_ANALYSIS.md`: Phase 4 layer-wise representations & renaming sensitivity
- `NR015_CURRICULUM_REPORT.md`: Phase 5 easy-to-hard vs randomized curriculum evaluation
- `NR015_ARCHITECTURE_EXPERIMENTS.md`: Phase 6 38M scaled architecture profiling
- `NR015_CODING_GENERALIZATION.md`: Phase 7 40-task execution benchmark results
- `NR015_LANGUAGE_EVALUATION.md`: Phase 8 multilingual conversational evaluation
- `NR015_ARTIFACT_INVENTORY.md`: Comprehensive inventory & SHA-256 signatures
- `NR015_FINAL_REPORT.md`: Master synthesis, scientific verdict, and roadmap
