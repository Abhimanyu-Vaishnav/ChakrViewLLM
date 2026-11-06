# CR-EXP-09: REPOSITORY & HARDWARE PREFLIGHT AUDIT
## Authoritative Lineage, Cryptographic Verification & Environment Inventory

- **Mission ID**: CR-EXP-09
- **Audit Date**: 2026-11-06T23:37:00+05:30
- **Auditor Role**: Autonomous AI Research & Development Lead

---

### 1. Environment & Compute Hardware Specifications

| Component | Verified Specification |
|---|---|
| **Operating System** | Windows 11 Enterprise (10.0.26200) |
| **Python Runtime** | Python 3.14.7 |
| **PyTorch Version** | 2.14.0+cpu |
| **CPU Architecture** | Intel64 Family 6 Model 154 Stepping 3, GenuineIntel |
| **GPU Acceleration** | None (CPU Deterministic Execution) |
| **Available RAM** | > 16.0 GB |
| **Available Storage** | 382.47 GB free on volume `D:` |
| **Working Directory** | `D:\Project\ChakrView` |
| **Current Git Branch** | `exp/cbr-01-conversational-breakthrough` |
| **Current Git HEAD** | `564472a42245a47a7bf6f78ca6410e523383d2ce` |
| **Working Tree Status** | Clean (Zero uncommitted tracked changes; new experiment directories isolated) |

---

### 2. Cryptographic Checkpoint & Artifact Lineage Verification

All 11 critical project artifacts were audited and verified bit-exact against their expected cryptographic digests:

| Artifact Name | Path / Origin | Expected SHA-256 Digest | Actual SHA-256 Digest | Status |
|---|---|---|---|---|
| **Canonical Baseline** | In-Memory `ChakrMicro` (Seed 42) | `c5571c9c5cb77386...` | `c5571c9c5cb77386...` | **VERIFIED** ($\Delta W = 0$) |
| **Frozen Parent** | `experiments/cbr01_sft/runs/cbr_sft_04/.../sft_checkpoint_step_200.pt` | `327118ce7057af86...` | `327118ce7057af86...` | **VERIFIED** |
| **Conversational Pilot** | `experiments/.../conversational_pilot_best_val.pt` | `ea6ee570854f5df7...` | `ea6ee570854f5df7...` | **VERIFIED** |
| **CR-EXP-06 Control** | `experiments/.../control_arm_best_val.pt` | `729d5812c1ec02d2...` | `729d5812c1ec02d2...` | **VERIFIED** |
| **CR-EXP-06 Treatment**| `experiments/.../treatment_arm_best_val.pt` | `2d8cea93ba8dbe06...` | `2d8cea93ba8dbe06...` | **VERIFIED** |
| **CR-EXP-08 Arm A** | `experiments/.../arm_a/arm_a_best_val.pt` | `42f64002b5e1a369...` | `42f64002b5e1a369...` | **VERIFIED** |
| **CR-EXP-08 Arm B** | `experiments/.../arm_b/arm_b_best_val.pt` | `98bd76f99d43dedd...` | `98bd76f99d43dedd...` | **VERIFIED** |
| **CR-EXP-08 Arm C** | `experiments/.../arm_c/arm_c_best_val.pt` | `0311c6dcf403c7e4...` | `0311c6dcf403c7e4...` | **VERIFIED** |
| **Tokenizer Vocab** | `data/experiments/vocab_4096/vocab.json` | `db4d6119feddbcc0...` | `db4d6119feddbcc0...` | **VERIFIED** |
| **Tokenizer Merges** | `data/experiments/vocab_4096/merges.json` | `ada4b1dcfbb3d962...` | `ada4b1dcfbb3d962...` | **VERIFIED** |
| **Protected Runtime** | `chakrview/runtime/session.py` | `9080ff51adba38d4...` | `9080ff51adba38d4...` | **VERIFIED** |

---

### 3. Safety & Architectural Boundaries Reaffirmed

1. **No External Weights / APIs**: Strict local CPU-first neural execution.
2. **Protected File Invariant**: `chakrview/runtime/session.py` remains untouched.
3. **No Production Overwrite**: Canonical baselines and previous experiment artifacts are strictly immutable.
4. **Isolated Workspaces**: All CR-EXP-09 training runs, datasets, and checkpoints reside strictly under `experiments/cognition/cr_exp_09_autonomous_mission/`.

