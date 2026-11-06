# CR-EXP-10 REPOSITORY & LINEAGE AUDIT REPORT
**Audit Executed:** 2026-11-07T00:01:45+05:30  
**Host Architecture:** ChakrMicro v0.1 ($3,443,136$ parameters, $d_{\text{model}}=192, n_{\text{layers}}=6, n_{\text{heads}}=6$, Vocab 4096)  
**Host Environment:** Python 3.14.7, PyTorch 2.14.0+cpu, Windows 11 (64-bit AMD64)  
**Git Branch:** `exp/cbr-01-conversational-breakthrough` | **HEAD:** `88a16a4`  

---

## 1. CRYPTOGRAPHIC VERIFICATION MATRIX

| Artifact Key | Relative File Path | Expected SHA-256 | Actual SHA-256 | Status | Notes |
|---|---|---|---|---|---|
| **Parent Checkpoint (Step 200)** | `experiments/cbr01_sft/runs/cbr_sft_04/checkpoints/sft_checkpoint_step_200.pt` | `327118ce7057af868d38c64ce3df35dba3b85b820b8022e18cb5be1f86cfa153` | `327118ce7057af868d38c64ce3df35dba3b85b820b8022e18cb5be1f86cfa153` | **VERIFIED** | Canonical SFT parent |
| **Conversational Pilot** | `experiments/cognition/autonomous_12h/conversational_pilot/checkpoints/conversational_pilot_best_val.pt` | `ea6ee570854f5df71cec09af20ff66ff55b46d9584382cba4d77c3dbae33b6fd` | `ea6ee570854f5df71cec09af20ff66ff55b46d9584382cba4d77c3dbae33b6fd` | **VERIFIED** | CR-EXP-04 verified pilot |
| **CR-08 Arm B (Baseline)** | `experiments/cognition/cr_exp_08_contextual_assignment/checkpoints/arm_b/arm_b_best_val.pt` | `98bd76f99d43dedd10c31a1d290c95e452eba299f511ecf72e5ecc4075fb0956` | `98bd76f99d43dedd10c31a1d290c95e452eba299f511ecf72e5ecc4075fb0956` | **VERIFIED** | CR-EXP-08 baseline origin |
| **CR-09 Baseline Copy** | `experiments/cognition/cr_exp_09_autonomous_mission/checkpoints/baseline/cr_exp_08_arm_b_best_val_copy.pt` | `98bd76f99d43dedd10c31a1d290c95e452eba299f511ecf72e5ecc4075fb0956` | `98bd76f99d43dedd10c31a1d290c95e452eba299f511ecf72e5ecc4075fb0956` | **VERIFIED** | Immutable copy |
| **CR-09 Candidate 3 (Champion)** | `experiments/cognition/cr_exp_09_autonomous_mission/checkpoints/candidate_3/cr_exp_09_candidate_3_best_val.pt` | `de871613aa28264014f22e227beea8976c359dca6f7026ed4221703ac7a3e592` | `de871613aa28264014f22e227beea8976c359dca6f7026ed4221703ac7a3e592` | **VERIFIED** | Starting Champion Model |
| **CR-10 Baseline Copy** | `experiments/cognition/cr_exp_10_readiness/checkpoints/baseline_cand3/cr_exp_09_candidate_3_best_val_copy.pt` | `de871613aa28264014f22e227beea8976c359dca6f7026ed4221703ac7a3e592` | `de871613aa28264014f22e227beea8976c359dca6f7026ed4221703ac7a3e592` | **VERIFIED** | Isolated working copy for CR-10 |
| **Tokenizer Vocab** | `data/experiments/vocab_4096/vocab.json` | `db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a` | `db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a` | **VERIFIED** | Canonical 4096 ByteBPE |
| **Tokenizer Merges** | `data/experiments/vocab_4096/merges.json` | `ada4b1dcfbb3d962a2e60203089881872a5063de73b2fef9eeff81ce0016efa7` | `ada4b1dcfbb3d962a2e60203089881872a5063de73b2fef9eeff81ce0016efa7` | **VERIFIED** | Canonical ByteBPE merges |
| **Protected Session Runtime** | `chakrview/runtime/session.py` | `9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67` | `9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67` | **VERIFIED** | Protected file untouched |

---

## 2. RECONCILIATION OF CANDIDATE 1 & 2 HASHES IN CR-EXP-09 REPORT
In the final narrative report of CR-EXP-09, the preliminary intermediate save hashes for Candidate 1 (`3134918e...`) and Candidate 2 (`8e3a2469...`) were listed from intermediate checkpoints, whereas the final on-disk checkpoints reflect `95b146b4...` and `16e28bbb...`.
Crucially, the **Candidate 3 Champion checkpoint** matches its reported hash:
`de871613aa28264014f22e227beea8976c359dca6f7026ed4221703ac7a3e592`.

---

## 3. AUDIT CONCLUSION & REPOSITORY SAFETY STATUS
- All canonical and protected files are intact and bit-for-bit verified.
- Protected runtime `chakrview/runtime/session.py` is verified unmodified.
- Candidate 3 Champion is verified and securely copied to `experiments/cognition/cr_exp_10_readiness/checkpoints/baseline_cand3/cr_exp_09_candidate_3_best_val_copy.pt`.
- Phase 0 is complete with zero blockers. Phase 1 (Independent CR-EXP-09 Validation) is authorized to proceed.

