# CR-EXP-12 REPOSITORY, LINEAGE & CHECKPOINT AUDIT REPORT
**Audit Date:** 2026-11-07T22:29:30+05:30  
**Host Architecture:** ChakrMicro v0.1 ($3,443,136$ parameters, $d_{\text{model}}=192, n_{\text{layers}}=6, n_{\text{heads}}=6$, Vocab 4096, Rotary Position Embeddings)  
**Host Environment:** Python 3.14.7, PyTorch 2.14.0+cpu, Windows 11 (AMD64)  
**Git Branch:** `exp/cbr-01-conversational-breakthrough` | **Base Commit HEAD:** `b9a3b46`  

---

## 1. CRYPTOGRAPHIC VERIFICATION MATRIX

| Artifact Key | Relative File Path | Expected SHA-256 | Actual SHA-256 | Status | Notes |
|---|---|---|---|---|---|
| **CR-09 Candidate 3** | `experiments/cognition/cr_exp_09_autonomous_mission/checkpoints/candidate_3/cr_exp_09_candidate_3_best_val.pt` | `de871613aa28264014f22e227beea8976c359dca6f7026ed4221703ac7a3e592` | `de871613aa28264014f22e227beea8976c359dca6f7026ed4221703ac7a3e592` | **VERIFIED** | CR-09 champion baseline |
| **CR-10 Arm A** | `experiments/cognition/cr_exp_10_readiness/checkpoints/candidate_arm_a/cr_exp_10_arm_a_best_val.pt` | `dc51e4ab338006ff7380bb7500963a17a8740b9b5eab75bb452166463892e74c` | `dc51e4ab338006ff7380bb7500963a17a8740b9b5eab75bb452166463892e74c` | **VERIFIED** | CR-10 comparison candidate |
| **CR-10 Arm B** | `experiments/cognition/cr_exp_10_readiness/checkpoints/candidate_arm_b/cr_exp_10_arm_b_best_val.pt` | `8c15c2cb8b4753f24931fd0cf9a76d8b629d7738c9a4f1869ef65b509af0d488` | `8c15c2cb8b4753f24931fd0cf9a76d8b629d7738c9a4f1869ef65b509af0d488` | **VERIFIED** | CR-10 parent baseline |
| **CR-11 Candidate A** | `experiments/cognition/cr_exp_11_entity_binding/checkpoints/candidate_a/cr_exp_11_candidate_a_best_val.pt` | `4d14cf05df395608ed8d1ab02eeca571b6bf049574785a3d16a7ec2112a64bc5` | `4d14cf05df395608ed8d1ab02eeca571b6bf049574785a3d16a7ec2112a64bc5` | **VERIFIED** | CR-11 Candidate A baseline |
| **CR-11 Candidate B** | `experiments/cognition/cr_exp_11_entity_binding/checkpoints/candidate_b/cr_exp_11_candidate_b_best_val.pt` | `b41bebc65017fc6918bf44fd65664b22b93a6d89f412a97d79b718f3e5453fce` | `b41bebc65017fc6918bf44fd65664b22b93a6d89f412a97d79b718f3e5453fce` | **VERIFIED** | CR-11 Candidate B baseline |
| **CR-12 Baseline A Copy** | `experiments/cognition/cr_exp_12_conversational_consolidation/checkpoints/baseline_cr11_a/cr_exp_11_candidate_a_best_val_copy.pt` | `4d14cf05df395608ed8d1ab02eeca571b6bf049574785a3d16a7ec2112a64bc5` | `4d14cf05df395608ed8d1ab02eeca571b6bf049574785a3d16a7ec2112a64bc5` | **VERIFIED** | Working immutable baseline copy |
| **CR-12 Baseline B Copy** | `experiments/cognition/cr_exp_12_conversational_consolidation/checkpoints/baseline_cr11_b/cr_exp_11_candidate_b_best_val_copy.pt` | `b41bebc65017fc6918bf44fd65664b22b93a6d89f412a97d79b718f3e5453fce` | `b41bebc65017fc6918bf44fd65664b22b93a6d89f412a97d79b718f3e5453fce` | **VERIFIED** | Working immutable baseline copy |
| **Tokenizer Vocab** | `data/experiments/vocab_4096/vocab.json` | `db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a` | `db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a` | **VERIFIED** | Canonical 4096 ByteBPE |
| **Tokenizer Merges** | `data/experiments/vocab_4096/merges.json` | `ada4b1dcfbb3d962a2e60203089881872a5063de73b2fef9eeff81ce0016efa7` | `ada4b1dcfbb3d962a2e60203089881872a5063de73b2fef9eeff81ce0016efa7` | **VERIFIED** | Canonical ByteBPE merges |
| **Protected Session Runtime** | `chakrview/runtime/session.py` | `9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67` | `9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67` | **VERIFIED** | Byte-for-byte untouched |

---

## 2. LINEAGE AUDIT CONCLUSION
All parent lineage checkpoints and canonical resources are intact and bit-for-bit cryptographically validated. Phase 0 repository audit is complete with zero blockers.

