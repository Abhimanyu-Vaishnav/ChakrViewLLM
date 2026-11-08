# CR-EXP-13 ARTIFACT VERIFICATION REPORT
**Auditor:** Autonomous Scientific Auditor & Neural Research Scientist  
**Audit Date:** 2026-11-08T10:05:00+05:30  
**Repository State:** Verified on branch `exp/cbr-01-conversational-breakthrough` at commit `3814f09`  

---

## 1. CRYPTOGRAPHIC VERIFICATION MATRIX

| Artifact Description | File Path | Expected SHA-256 | Actual SHA-256 | Verification Status | Notes |
|---|---|---|---|---|---|
| **Protected Runtime** | `chakrview/runtime/session.py` | `9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67` | `9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67` | **VERIFIED** | Byte-for-byte untouched |
| **Tokenizer Vocab** | `data/experiments/vocab_4096/vocab.json` | `db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a` | `db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a` | **VERIFIED** | Canonical 4,096 BPE |
| **Tokenizer Merges** | `data/experiments/vocab_4096/merges.json` | `ada4b1dcfbb3d962a2e60203089881872a5063de73b2fef9eeff81ce0016efa7` | `ada4b1dcfbb3d962a2e60203089881872a5063de73b2fef9eeff81ce0016efa7` | **VERIFIED** | Canonical Merges |
| **CR-10 Arm B Checkpoint**| `experiments/cognition/cr_exp_10_readiness/checkpoints/candidate_arm_b/cr_exp_10_arm_b_best_val.pt` | `8c15c2cb8b4753f24931fd0cf9a76d8b629d7738c9a4f1869ef65b509af0d488` | `8c15c2cb8b4753f24931fd0cf9a76d8b629d7738c9a4f1869ef65b509af0d488` | **VERIFIED** | Baseline reference |
| **CR-11 Candidate A Checkpoint**| `experiments/cognition/cr_exp_11_entity_binding/checkpoints/candidate_a/cr_exp_11_candidate_a_best_val.pt` | `4d14cf05df395608ed8d1ab02eeca571b6bf049574785a3d16a7ec2112a64bc5` | `4d14cf05df395608ed8d1ab02eeca571b6bf049574785a3d16a7ec2112a64bc5` | **VERIFIED** | CR-11 Champion |
| **CR-11 Candidate B Checkpoint**| `experiments/cognition/cr_exp_11_entity_binding/checkpoints/candidate_b/cr_exp_11_candidate_b_best_val.pt` | `b41bebc65017fc6918bf44fd65664b22b93a6d89f412a97d79b718f3e5453fce` | `b41bebc65017fc6918bf44fd65664b22b93a6d89f412a97d79b718f3e5453fce` | **VERIFIED** | CR-11 Working Memory Challenger |
| **CR-12 Candidate A Best**| `experiments/cognition/cr_exp_12_conversational_consolidation/checkpoints/candidate_a/cr_exp_12_candidate_a_best_val.pt` | `cfb99ad7651bf64303a4fd8ab562ab9d7b0b0ed008fb3389957e97c76a05a8a2` | `cfb99ad7651bf64303a4fd8ab562ab9d7b0b0ed008fb3389957e97c76a05a8a2` | **VERIFIED** | CR-12 Memory Specialist |
| **CR-12 Candidate A Final**| `experiments/cognition/cr_exp_12_conversational_consolidation/checkpoints/candidate_a/cr_exp_12_candidate_a_final.pt` | `f6e59aa1459878b2a547b42c87e08076a30d6ab0c435695dfa50bb088098dcb2` | `f6e59aa1459878b2a547b42c87e08076a30d6ab0c435695dfa50bb088098dcb2` | **VERIFIED** | Step 250 checkpoint |
| **CR-12 Candidate B Best**| `experiments/cognition/cr_exp_12_conversational_consolidation/checkpoints/candidate_b/cr_exp_12_candidate_b_best_val.pt` | `60271c8b5e3d26b76237dc8f986e0d8d6ada2f08815cea57ec5d9a37102d9d2e` | `60271c8b5e3d26b76237dc8f986e0d8d6ada2f08815cea57ec5d9a37102d9d2e` | **VERIFIED** | CR-12 Champion |
| **CR-12 Candidate B Final**| `experiments/cognition/cr_exp_12_conversational_consolidation/checkpoints/candidate_b/cr_exp_12_candidate_b_final.pt` | `bffd8c565ce22216c5da231802e977d5287006d913f5409b4268ff653365668e` | `bffd8c565ce22216c5da231802e977d5287006d913f5409b4268ff653365668e` | **VERIFIED** | Step 250 checkpoint |
| **Frozen 1,500 Benchmark**| `experiments/cognition/cr_exp_12_conversational_consolidation/benchmark_data/benchmark_1500_suite.json` | `19565749c65dbe91552caa337997eaa4052e6ed9c46a8bfa8825e70cfd65184c` | `19565749c65dbe91552caa337997eaa4052e6ed9c46a8bfa8825e70cfd65184c` | **VERIFIED** | Frozen 1,500 probes |

---

## 2. BENCHMARK INTEGRITY AUDIT
- Total Probes: Exactly 1,500 probes.
- Unique IDs: 1,500 unique IDs verified (zero duplicate IDs).
- Category Totals:
  - `answerability_calibration`: 150
  - `conditional_logic`: 100
  - `constraint_satisfaction`: 75
  - `counterfactual_updates`: 200
  - `elementary_arithmetic`: 75
  - `multi_entity_distractor`: 200
  - `multiturn_working_memory`: 300
  - `natural_conversation`: 200
  - `single_hop_retrieval`: 200
  - Sum: Exactly 1,500 probes.
- **Reproducibility of Reported Scores**:
  All reported overall accuracies (CR-10 Arm B: $17.33\%$, CR-11 Candidate A: $29.33\%$, CR-11 Candidate B: $28.33\%$, CR-12 Candidate A: $31.87\%$, CR-12 Candidate B: $34.13\%$) are **100% numerically reproduced to the exact integer probe count**.
