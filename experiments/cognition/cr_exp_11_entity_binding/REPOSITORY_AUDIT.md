# CR-EXP-11 REPOSITORY, LINEAGE & CHECKPOINT AUDIT REPORT
**Audit Date:** 2026-11-07T21:52:00+05:30  
**Host Architecture:** ChakrMicro v0.1 ($3,443,136$ parameters, $d_{\text{model}}=192, n_{\text{layers}}=6, n_{\text{heads}}=6$, Vocab 4096, RoPE)  
**Host Environment:** Python 3.14.7, PyTorch 2.14.0+cpu, Windows 11 (AMD64)  
**Git Branch:** `exp/cbr-01-conversational-breakthrough` | **HEAD:** `6a13aea`  

---

## 1. CRYPTOGRAPHIC VERIFICATION MATRIX

| Artifact Key | Relative File Path | Expected SHA-256 | Actual SHA-256 | Status | Notes |
|---|---|---|---|---|---|
| **CR-10 Arm B (Champion)** | `experiments/cognition/cr_exp_10_readiness/checkpoints/candidate_arm_b/cr_exp_10_arm_b_best_val.pt` | `8c15c2cb8b4753f24931fd0cf9a76d8b629d7738c9a4f1869ef65b509af0d488` | `8c15c2cb8b4753f24931fd0cf9a76d8b629d7738c9a4f1869ef65b509af0d488` | **VERIFIED** | CR-10 primary baseline |
| **CR-10 Arm A** | `experiments/cognition/cr_exp_10_readiness/checkpoints/candidate_arm_a/cr_exp_10_arm_a_best_val.pt` | `dc51e4ab338006ff7380bb7500963a17a8740b9b5eab75bb452166463892e74c` | `dc51e4ab338006ff7380bb7500963a17a8740b9b5eab75bb452166463892e74c` | **VERIFIED** | CR-10 comparison candidate |
| **CR-09 Candidate 3 (Champion)** | `experiments/cognition/cr_exp_09_autonomous_mission/checkpoints/candidate_3/cr_exp_09_candidate_3_best_val.pt` | `de871613aa28264014f22e227beea8976c359dca6f7026ed4221703ac7a3e592` | `de871613aa28264014f22e227beea8976c359dca6f7026ed4221703ac7a3e592` | **VERIFIED** | CR-09 comparison baseline |
| **CR-11 Arm B Working Copy** | `experiments/cognition/cr_exp_11_entity_binding/checkpoints/baseline_arm_b/cr_exp_10_arm_b_best_val_copy.pt` | `8c15c2cb8b4753f24931fd0cf9a76d8b629d7738c9a4f1869ef65b509af0d488` | `8c15c2cb8b4753f24931fd0cf9a76d8b629d7738c9a4f1869ef65b509af0d488` | **VERIFIED** | Immutable working copy |
| **CR-11 Cand 3 Working Copy** | `experiments/cognition/cr_exp_11_entity_binding/checkpoints/baseline_cand3/cr_exp_09_candidate_3_best_val_copy.pt` | `de871613aa28264014f22e227beea8976c359dca6f7026ed4221703ac7a3e592` | `de871613aa28264014f22e227beea8976c359dca6f7026ed4221703ac7a3e592` | **VERIFIED** | Immutable working copy |
| **Tokenizer Vocab** | `data/experiments/vocab_4096/vocab.json` | `db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a` | `db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a` | **VERIFIED** | Canonical 4096 ByteBPE |
| **Tokenizer Merges** | `data/experiments/vocab_4096/merges.json` | `ada4b1dcfbb3d962a2e60203089881872a5063de73b2fef9eeff81ce0016efa7` | `ada4b1dcfbb3d962a2e60203089881872a5063de73b2fef9eeff81ce0016efa7` | **VERIFIED** | Canonical ByteBPE merges |
| **Protected Session Runtime** | `chakrview/runtime/session.py` | `9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67` | `9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67` | **VERIFIED** | Protected file untouched |

---

## 2. LINEAGE AUDIT
- `sft_checkpoint_step_200.pt` (`327118ce...`) $\to$ `conversational_pilot_best_val.pt` (`ea6ee570...`) $\to$ `arm_b_best_val.pt` (`98bd76f9...`) $\to$ `cr_exp_09_candidate_3_best_val.pt` (`de871613...`) $\to$ `cr_exp_10_arm_b_best_val.pt` (`8c15c2cb...`).
- Full lineage chain is cryptographically intact and validated.

---

## 3. PHASE 0 CONCLUSION
All baseline models and canonical resources are intact and bit-for-bit verified. Immutable working copies are staged under `experiments/cognition/cr_exp_11_entity_binding/checkpoints/`. Phase 0 is complete.

