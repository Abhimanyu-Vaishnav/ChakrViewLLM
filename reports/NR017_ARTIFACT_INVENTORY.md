# NR-017 Phase 10 — Artifact Inventory & Verification Signatures
**Mission ID:** CV-GENESIS-NR-017  
**Date:** 2026-11-26  
**Repository:** `D:\Project\ChakrView`  
**Active Branch:** `exp/cbr-01-conversational-breakthrough`  

---

## 1. Checkpoint Cryptographic Signatures

| Artifact Name | Path | Size | SHA-256 Checksum | Classification |
| :--- | :--- | :--- | :--- | :--- |
| **Stage-D Best (17M)** | `experiments/genesis_stage_d_011/checkpoints/genesis_17m_staged_best.pt` | 68,292,655 B | `2c21632ebae4f0b05b9ff7091797ecc956ac0ad00ef6ec934381223bc273b269` | **PRIMARY CANONICAL** |
| **Stage-E Best (17M)** | `experiments/genesis_stage_e_013/checkpoints/genesis_17m_stage_e_best.pt` | 68,292,561 B | `0ff238b6829ebd638bbb8cdefded86616c5e76c0180946e69da74a77ba3ac630` | Historical Milestone |
| **Stage-F Best (17M)** | `experiments/genesis_stage_f_014/checkpoints/genesis_17m_stage_f_best.pt` | 68,292,561 B | `454aa58af12cb915ecf9bf144366489eb56dcfa088e0845574bc24fcca6f6ab8` | Historical Milestone |
| **NR-010 Pretrain Best** | `experiments/genesis_pretraining_010/checkpoints/genesis_17m_pretrain_best.pt` | 68,292,851 B | `96219f224e7c4a7909254a5b7f0fc33a649853ffa4264c576b0852d3caeed7c0` | Historical Foundation |
| **NR-005 SFT Best** | `experiments/genesis_conversational_foundation_005/checkpoints/genesis_sft_best.pt` | 62,203,157 B | `ee1b91e848db3ca6e25008724f9f4fe200f217b4707ef81964395676da3d4269` | Historical 5.18M SFT |
| **Canonical Pretrained** | `experiments/project_genesis/pretraining/genesis_pretrain_best.pt` | 62,204,051 B | `c5cf8becb87b3d9439861bf66736d18134edbc74322f77a3dfc35c8f8c129c9c` | Canonical 5.18M Base |

---

## 2. Experimental Data Suites & Benchmark Manifests

| Dataset / Benchmark File | Path | Tasks / Samples | SHA-256 Checksum | Purpose |
| :--- | :--- | :---: | :--- | :--- |
| **Stage-G Training Corpus** | `data/stage_g_017/stage_g_train_1800.json` | 1,800 tasks | `1da85a77f2200beea3f2d5b37c7878e40c61bc102acb41c25e3af05cb14ab81c` | 10-family training corpus (3 conditions) |
| **Stage-G Sealed Benchmark** | `data/stage_g_017/stage_g_sealed_benchmark_400.json` | 400 tasks | `8bf8ff917118307f3c72687e67d3d75b81388b632830bb3c7c55bd28712a1bcc` | 400-task sealed benchmark (200 In-Dist / 200 OOD) |
| **Training Comparison Log** | `data/stage_g_017/training_comparison_results.json` | 6 runs | Generated | 3 conditions $\times$ 2 seeds training log |
| **Trace Verification Log** | `data/stage_g_017/trace_verification_results.json` | 400 tasks | Generated | Replay verification of Condition B |
| **Detailed Breakdown Log** | `data/stage_g_017/detailed_generalization_breakdown.json` | 400 tasks | Generated | Family-wise & difficulty-wise metrics |
| **Counterfactual Log** | `data/stage_g_017/compositional_counterfactual_results.json` | 4 pairs | Generated | Invariance & sensitivity probes |
| **Regression Results Log** | `data/stage_g_017/regression_evaluation_results.json` | 26 tasks | Generated | Coding & trilingual regression log |

---

## 3. Mission Research Reports (`reports/`)
- `NR017_RECOVERY_AUDIT.md`: Phase 0 recovery & hash audit
- `NR017_TASK_GENERATOR.md`: Phase 1 & 2 task generator & 3 condition specifications
- `NR017_OBJECTIVE_AND_TOKENIZER_AUDIT.md`: Phase 3 tokenizer & objective verification
- `NR017_TRAINING_COMPARISON.md`: Phase 4 controlled 3 conditions $\times$ 2 seeds training comparison
- `NR017_TRACE_VERIFICATION.md`: Phase 5 independent trace verifier & transition replay
- `NR017_GENERALIZATION_RESULTS.md`: Phase 6 400-task family & difficulty breakdown
- `NR017_COMPOSITIONAL_ANALYSIS.md`: Phase 7 counterfactual invariance & sensitivity tests
- `NR017_REGRESSION_EVALUATION.md`: Phase 8 coding & multilingual regression testing
- `NR017_ARTIFACT_INVENTORY.md`: Comprehensive cryptographic inventory & manifests
- `NR017_FINAL_REPORT.md`: Master synthesis, scientific verdict, and roadmap
