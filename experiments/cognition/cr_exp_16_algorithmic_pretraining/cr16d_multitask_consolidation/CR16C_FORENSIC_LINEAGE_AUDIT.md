# CR-16C Forensic Lineage Audit & Benchmark Verification

### 1. Lineage & Checkpoint Hash Verification
Every artifact utilized as a parent or baseline for CR-EXP-16D was audited against repository records before initiating experiments:

| Artifact | Reported Path | Expected / Recorded Hash | Verification Status |
|---|---|---|---|
| **Stage B Parent** | `checkpoints/stage_b_checkpoint.pt` | `963577053e1987ba42a49f5068407421cbff7973c178229a43a0544f808f9024` | **VERIFIED MATCH** |
| **CR-16 Copying Benchmark** | `experiments/cognition/cr_exp_16_algorithmic_pretraining/benchmark_data/cr16_synthetic_benchmark_1000.json` | `0b61e73d31b089c8a816c4f0da99e900c42732952402fe34d8721c56ef98f5da` | **VERIFIED MATCH** |
| **CR-16A Arm C Checkpoint** | `.../cr16a_copying_recovery/checkpoints/arm_c_multi_template/arm_c_multi_template_best.pt` | `ac4beb4cae8f6e8c715c0245a4918e9f2ee727829d67ef07a6eec967cb3d4b68` | **VERIFIED MATCH** |
| **CR-16A Arm D Checkpoint (Parent)** | `.../cr16a_copying_recovery/checkpoints/arm_d_generalization/arm_d_generalization_best.pt` | `9d609a770c748f9321d06713093620da0fe6caabc04fea2401b37403d33de1a5` | **VERIFIED MATCH** |
| **CR-16C Benchmark (1,500 probes)** | `.../cr16c_key_value_retrieval/data/cr16c_sealed_benchmark_1500.json` | `033f807b539c0fae3b7b2049f5ce624f7e2501a4e101f2f87a3b4e2860d5bfa7` | **VERIFIED MATCH** |
| **CR-16C Retrieval Arm A s42** | `.../cr16c_key_value_retrieval/checkpoints/retrieval_arm_a_target_masked_seed42/retrieval_arm_a_target_masked_best.pt` | `375df4cefdfe7fbe5349f280a566ee2b2c32cf97b1029c7ffc82c610e75a507d` | **VERIFIED MATCH** |
| **Protected Runtime** | `chakrview/runtime/session.py` | `9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67` | **VERIFIED UNTOUCHED** |

---

### 2. Forensic Recomputation of CR-16C Experimental Claims
To guarantee scientific integrity, the raw evaluation predictions from CR-16C were independently re-parsed and re-scored against ground truth before CR-16D training:

| Model Evaluated in CR-16C | Reported Retrieval Score | Recomputed Retrieval Score | Match Status | Reported Copying Score |
|---|---|---|---|---|
| Stage B Baseline | 0 / 1500 (0.0%) | 0 / 1500 (0.00%) | **EXACT MATCH** | 0 / 100 (0.0%) |
| CR-16A Arm D Parent | 4 / 1500 (0.27%) | 4 / 1500 (0.27%) | **EXACT MATCH** | 66 / 100 (66.0%) |
| Arm A (Target Masked) Seed 42 | 885 / 1500 (59.00%) | 885 / 1500 (59.00%) | **EXACT MATCH** | 0 / 100 (0.0%) |
| Arm A (Target Masked) Seed 101 | 868 / 1500 (57.87%) | 868 / 1500 (57.87%) | **EXACT MATCH** | 0 / 100 (0.0%) |
| Arm B (Full Sequence) Seed 42 | 730 / 1500 (48.67%) | 730 / 1500 (48.67%) | **EXACT MATCH** | 0 / 100 (0.0%) |
| Arm B (Full Sequence) Seed 101 | 772 / 1500 (51.47%) | 772 / 1500 (51.47%) | **EXACT MATCH** | 0 / 100 (0.0%) |

### 3. Conclusion of the Audit
1. CR-16C's reported scores were 100% reproducible and confirmed bit-for-bit from source predictions.
2. The catastrophic forgetting of sequence copying upon learning retrieval was confirmed: CR-16A Arm D scored 66/100 copying accuracy prior to retrieval training, and collapsed to 0/100 copying accuracy after pure retrieval fine-tuning across all seeds.
3. CR-16D correctly builds directly on verified parent artifacts without synthetic data contamination.

