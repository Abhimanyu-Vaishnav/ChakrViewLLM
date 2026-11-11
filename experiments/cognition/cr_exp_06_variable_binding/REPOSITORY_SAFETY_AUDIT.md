# Repository Safety Audit — CR-EXP-06

**Date**: 2026-11-06  
**Execution Environment**: Windows PowerShell, Python virtualenv at `.venv/Scripts/python.exe`  
**Git Branch**: `exp/cbr-01-conversational-breakthrough`  
**Git HEAD**: `564472a42245a47a7bf6f78ca6410e523383d2ce`  
**Working Tree Status**: Contains newly generated read-only evaluation deliverables for `cr_exp_04_validation` and `cr_exp_05_ood_diagnosis`. Zero unstaged modifications to core codebase.

---

## 1. Verified Cryptographic Lineage

All mandatory artifact hashes were computed directly from disk buffers:

| Target Artifact | File Path | Actual SHA-256 Digest | Status |
|---|---|---|---|
| **Parent Checkpoint** | `experiments/cbr01_sft/runs/cbr_sft_04/checkpoints/sft_checkpoint_step_200.pt` | `327118ce7057af868d38c64ce3df35dba3b85b820b8022e18cb5be1f86cfa153` | **PASS (Exact)** |
| **Conversational Candidate** | `experiments/cognition/autonomous_12h/conversational_pilot/checkpoints/conversational_pilot_best_val.pt` | `ea6ee570854f5df71cec09af20ff66ff55b46d9584382cba4d77c3dbae33b6fd` | **PASS (Exact)** |
| **Canonical Baseline Init** | `chakrview/brain/model.py` (seed=42) | `c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da` | **PASS (Exact)** |
| **Tokenizer Vocab** | `data/experiments/vocab_4096/vocab.json` | `db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a` | **PASS (Exact)** |
| **Tokenizer Merges** | `data/experiments/vocab_4096/merges.json` | `ada4b1dcfbb3d962a2e60203089881872a5063de73b2fef9eeff81ce0016efa7` | **PASS (Exact)** |
| **Tokenizer Config** | `data/experiments/vocab_4096/config.json` | `0fca72b6a70e9ae812589199c661ef6be20cb9973f06353db125a1ec7264e589` | **PASS (Exact)** |
| **Protected Runtime** | `chakrview/runtime/session.py` | `9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67` | **PASS (Untouched)** |

---

## 2. Protected Execution Boundaries Confirmed
1. **Canonical Baseline Protection**: `c5571c...` is completely isolated; $\Delta W = 0$ enforced.
2. **Parent & Candidate Checkpoint Protection**: Both checkpoints are treated as strictly read-only reference models. All new candidate weights will be written exclusively to `experiments/cognition/cr_exp_06_variable_binding/control_arm/` and `treatment_arm/`.
3. **Architecture Preservation**: ChakrMicro v0.1 remains frozen at 3,443,136 parameters (`d_model=192, n_layers=6, n_heads=6, d_ff=512, max_seq_len=512, vocab_size=4096`).
4. **Git Safety**: No commits, resets, rebases, merges, pushes, or branch modifications will occur.

