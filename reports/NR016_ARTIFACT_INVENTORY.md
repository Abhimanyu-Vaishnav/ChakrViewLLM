# NR-016 Phase 9 — Artifact Inventory & Verification Signatures
**Mission ID:** CV-GENESIS-NR-016  
**Date:** 2026-11-25  
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
| **Scientific Benchmark 320** | `data/scientific_eval_016/scientific_benchmark_320.json` | 320 tasks | `48e97598ee3fd64a1fca69bf160a58211e8c9b6d2ceb0f712d900142bbc57209` | 8-family In-Dist vs OOD benchmark |
| **Scientific Training 1600** | `data/scientific_eval_016/scientific_train_1600.json` | 1,600 samples | `b2fdc0dac114058e71dc30865f4cf5b33abfbf683593b281c5987b86883b74e6` | Matched scratch training corpus |
| **Capacity Comparison Log** | `data/scientific_eval_016/capacity_comparison_results.json` | 4 model runs | Generated | 17M vs 38M Seed 42 & 101 records |
| **Coding Benchmark 50** | `data/coding_eval_016/coding_benchmark_50.json` | 50 tasks | Generated | Hidden unit tests suite |
| **Coding Eval Results 50** | `data/coding_eval_016/coding_eval_results_50.json` | 50 tasks | Generated | Sandbox execution results |
| **Recovered 325 Results** | `data/sealed_eval_015/recovered_benchmark_results.json` | 325 tasks | Generated | Turn-by-turn family results |
| **Language Regression Log** | `data/scientific_eval_016/language_regression_results.json` | 12 prompts | Generated | Multilingual & cognitive results |

---

## 3. Mission Research Reports (`reports/`)
- `NR016_RECOVERY_AND_INTEGRITY_AUDIT.md`: Phase 0 recovery & hash audit
- `NR016_NR015_REPRODUCTION_AUDIT.md`: Phase 1 independent reproduction results
- `NR016_SEALED_BENCHMARK_RECOVERY.md`: Phase 2 complete 325-task recovery results
- `NR016_GENERALIZATION_BENCHMARK.md`: Phase 3 scientific task grammar & generator
- `NR016_CAPACITY_COMPARISON.md`: Phase 4 17M vs 38M matched comparison across seeds
- `NR016_REPRESENTATION_STUDY.md`: Phase 5 depth & counterfactual representation study
- `NR016_CODING_GENERALIZATION.md`: Phase 6 50-task coding benchmark results
- `NR016_LANGUAGE_AND_REGRESSION.md`: Phase 7 trilingual evaluation & regression check
- `NR016_ARTIFACT_INVENTORY.md`: Cryptographic inventory & manifests
- `NR016_FINAL_REPORT.md`: Master synthesis, scientific verdict, and gate decision
