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
| **Stage-G Cond A Seed42**| `experiments/genesis_stage_g_017/Condition_A_Seed42.pt` | 68,292,449 B | Verified | Experimental Condition A |
| **Stage-G Cond B Seed42**| `experiments/genesis_stage_g_017/Condition_B_Seed42.pt` | 68,292,449 B | Verified | Experimental Condition B |
| **Stage-G Cond C Seed42**| `experiments/genesis_stage_g_017/Condition_C_Seed42.pt` | 68,292,449 B | Verified | Experimental Condition C |
| **Stage-G Cond A Seed101**| `experiments/genesis_stage_g_017/Condition_A_Seed101.pt` | 68,292,449 B | Verified | Experimental Condition A |
| **Stage-G Cond B Seed101**| `experiments/genesis_stage_g_017/Condition_B_Seed101.pt` | 68,292,449 B | Verified | Experimental Condition B |
| **Stage-G Cond C Seed101**| `experiments/genesis_stage_g_017/Condition_C_Seed101.pt` | 68,292,449 B | Verified | Experimental Condition C |

---

## 2. Experimental Data Suites & Benchmark Manifests

| Dataset / Benchmark File | Path | Tasks / Samples | SHA-256 Checksum | Purpose |
| :--- | :--- | :---: | :--- | :--- |
| **Stage-G Training 1800** | `data/stage_g_017/stage_g_train_1800.json` | 1,800 tasks | `1da85a77f2200beea3f2d5b37c7878e40c61bc102acb41c25e3af05cb14ab81c` | 10-family training tasks |
| **Stage-G Sealed Benchmark** | `data/stage_g_017/stage_g_sealed_benchmark_400.json` | 400 tasks | `8bf8ff917118307f3c72687e67d3d75b81388b632830bb3c7c55bd28712a1bcc` | In-dist (200) & OOD (200) benchmark |
| **Training Comparison Log** | `data/stage_g_017/training_comparison_results.json` | 6 runs | Generated | 3 conditions x 2 seeds metrics |
| **Trace Verification Results**| `data/stage_g_017/trace_verification_results.json` | 400 tasks | Generated | Transition replay & audit |
| **Compositional Results** | `data/stage_g_017/compositional_counterfactual_results.json` | 4 pairs | Generated | Renaming & perturbation tests |
| **Regression Results** | `data/stage_g_017/regression_evaluation_results.json` | Multiple | Generated | Python & multilingual check |

---

## 3. Mission Research Reports (`reports/`)
- `NR017_RECOVERY_AUDIT.md`: Phase 0 recovery & hash audit
- `NR017_TASK_GENERATOR.md`: Phase 1 & 2 task generator & 3 condition specifications
- `NR017_OBJECTIVE_AND_TOKENIZER_AUDIT.md`: Phase 3 token & loss invariant audit
- `NR017_TRAINING_COMPARISON.md`: Phase 4 3-condition x 2-seed training comparison
- `NR017_TRACE_VERIFICATION.md`: Phase 5 trace replay verification results
- `NR017_GENERALIZATION_RESULTS.md`: Phase 6 400-task held-out benchmark evaluation
- `NR017_COMPOSITIONAL_ANALYSIS.md`: Phase 7 counterfactual paired test analysis
- `NR017_REGRESSION_EVALUATION.md`: Phase 8 coding and multilingual regression report
- `NR017_ARTIFACT_INVENTORY.md`: Cryptographic inventory and manifests
- `NR017_FINAL_REPORT.md`: Master synthesis, scientific verdict, and roadmap
