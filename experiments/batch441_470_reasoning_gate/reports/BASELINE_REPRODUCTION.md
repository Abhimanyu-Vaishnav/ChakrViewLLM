# Baseline Reproduction & Audit Report
**Location:** `experiments/batch441_470_reasoning_gate/reports/BASELINE_REPRODUCTION.md`  
**Milestone:** Batches 441–470  
**Target:** Commit `4ee1260` and `candidate_step150_paraphrase_augmented.pt`

---

## 1. Verified Artifact Checksums

| Artifact | Path | Measured SHA-256 | Status |
| :--- | :--- | :--- | :--- |
| **Canonical Checkpoint** | `experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt` | `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` | Verified / Immutable |
| **Candidate Step 150 (Corrected EOS)** | `experiments/batch381_410_integrity/checkpoints/candidate_step150_corrected_eos.pt` | `27812873dde718485e70825d7be101ede2d1f54d8dd60e3f6a169c51e91a8f9e` | Verified Reference |
| **Candidate Paraphrase Augmented** | `experiments/batch411_440_paraphrase/checkpoints/candidate_step150_paraphrase_augmented.pt` | `b8b82aaf2b713ea73ff4337a1db7a5b2cce7efd24ae875c70934b2555e0c48b0` | Verified Starting State |

---

## 2. Empirical Benchmark Reproduction

| Benchmark Manifest | Total Items | Prior Reported Exact Match | Measured Re-evaluation | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Held-Out Paraphrase** | 20 | 65.0% (13/20) | **65.0% (13/20)** | **100% REPRODUCED** |
| **Diagnostic Retention** | 15 | 66.7% (10/15) | **66.7% (10/15)** | **100% REPRODUCED** |
| **Held-Out Compositional** | 4 | 0.0% (0/4) | **0.0% (0/4)** | **100% REPRODUCED** |

---

## 3. Analysis of Compositional Defect

On compound queries like `"Question: Calculate (14 plus 28) minus 10.\nAnswer:"`, the model predicts `42` because `14 plus 28` strongly activates the learned addition shortcut, completely ignoring the subsequent `- 10` subtraction operation.

Without explicit step-by-step intermediate token supervision (e.g. `Step 1: 14 + 28 = 42. Step 2: 42 - 10 = 32. Final: 32`), an autoregressive 3.44M model cannot perform mental multi-digit arithmetic in a single forward pass.
