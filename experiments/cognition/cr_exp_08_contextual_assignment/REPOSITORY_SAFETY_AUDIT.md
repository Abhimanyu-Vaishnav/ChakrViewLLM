# CR-EXP-08: REPOSITORY SAFETY AUDIT
## Preflight Verification & Resource Safeguards

- **Date**: 2026-11-06
- **Working Directory**: `D:\Project\ChakrView`
- **Branch**: `exp/cbr-01-conversational-breakthrough`
- **HEAD Commit**: `564472a42245a47a7bf6f78ca6410e523383d2ce`

---

### 1. Environment & Compute Hardware Specifications

| Component | Verified Specification |
|---|---|
| **Operating System** | Windows 11 Enterprise (10.0.26200) |
| **Python Runtime** | 3.14.7 |
| **PyTorch Version** | 2.14.0+cpu |
| **Processor (CPU)** | Intel64 Family 6 Model 154 Stepping 3, GenuineIntel |
| **GPU Acceleration** | None (Deterministic CPU Execution) |
| **Available RAM** | > 16.0 GB |
| **Available Disk** | 382.47 GB free on volume `D:` |

---

### 2. Cryptographic Checkpoint & Artifact Hashes

All protected repository artifacts were verified bit-exact prior to any operation:

| Artifact Target | File Path / Specification | Expected SHA-256 | Actual SHA-256 | Status |
|---|---|---|---|---|
| **Canonical Baseline** | In-Memory `ChakrMicro` (Seed 42) | `c5571c9c5cb77386...` | `c5571c9c5cb77386...` | **MATCH** ($\Delta W = 0$) |
| **Frozen Parent** | `experiments/cbr01_sft/runs/cbr_sft_04/.../sft_checkpoint_step_200.pt` | `327118ce7057af86...` | `327118ce7057af86...` | **MATCH** |
| **Conversational Pilot** | `experiments/.../conversational_pilot_best_val.pt` | `ea6ee570854f5df7...` | `ea6ee570854f5df7...` | **MATCH** |
| **CR-EXP-06 Arm A** | `experiments/.../control_arm_best_val.pt` | `729d5812c1ec02d2...` | `729d5812c1ec02d2...` | **MATCH** |
| **CR-EXP-06 Arm B** | `experiments/.../treatment_arm_best_val.pt` | `2d8cea93ba8dbe06...` | `2d8cea93ba8dbe06...` | **MATCH** |
| **Tokenizer Vocab** | `data/experiments/vocab_4096/vocab.json` | `db4d6119feddbcc0...` | `db4d6119feddbcc0...` | **MATCH** |
| **Tokenizer Merges** | `data/experiments/vocab_4096/merges.json` | `ada4b1dcfbb3d962...` | `ada4b1dcfbb3d962...` | **MATCH** |
| **Protected Runtime** | `chakrview/runtime/session.py` | `9080ff51adba38d4...` | `9080ff51adba38d4...` | **MATCH** |

---

### 3. Execution Safety Commitments

1. **Isolation Guarantee**: Training artifacts for Arm A, Arm B, and Arm C will be isolated under `experiments/cognition/cr_exp_08_contextual_assignment/checkpoints/`.
2. **Read-Only Preservation**: No files in `chakrview/`, `checkpoints/`, `data/`, or prior experiment folders (`cr_exp_01` through `cr_exp_07.2`) will be modified.
3. **No External Intelligence**: Zero external API calls, zero downloaded pretrained weights, zero Hugging Face weights.
4. **Independent Weight Copies**: Each training arm starts from an independent copy of `conversational_pilot_best_val.pt` to prevent state contamination.

