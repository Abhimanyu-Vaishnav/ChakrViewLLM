# Checkpoint Dynamics and Provenance Audit

**Target Experiment**: `exp19_debiased_sft_cpu_600_20261024_125700`  
**Diagnosis Directory**: `experiments/overnight_runs/exp19_output_diagnosis_20261024_131331`  

---

## 1. Checkpoint Provenance Verification

- **Starting Base Checkpoint**:
  - Expected: `experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt`
  - Canonical SHA-256: `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee`
  - Isolated Copy: `checkpoints/training_checkpoint_start.pt` (`SHA-256: 96357705...`, bit-exact match).
  - Unmodified Verification: Canonical checkpoint was verified untouched before and after the run.

---

## 2. Selection of Step 340 vs Step 600

### Empirical Loss Curve
- **Initial Validation Loss**: 4.5025 (Perplexity: 90.24)
- **Minimum Validation Loss**: **2.8786** (Perplexity: 17.79) at **Step 340**.
- **Post-Step 340 Validation Loss Trajectory**:
  - Step 360: 2.8912
  - Step 400: 2.9463
  - Step 460: 3.0118
  - Step 500: 3.0857
  - Step 600: 3.3124 (Final step)
- **Training Loss Trajectory**:
  - Step 340: 1.8622
  - Step 400: 1.8499
  - Step 500: 1.6210
  - Step 600: 1.3414

### Over-Fitting Diagnosis
Between Step 340 and Step 600, training loss continued to fall monotonically from 1.8622 to 1.3414, while validation loss increased monotonically from 2.8786 to 3.3124 (+0.4338 nats). This demonstrates clear, classic over-fitting on the 1,200 training examples.

### Checkpoint File Verification
- Checkpoint `sft_best_checkpoint.pt` (`SHA-256: 997bf2f3da8652c20ea8f7372969d845ac956292b200ae9ca161069943ea9284`) contains:
  - `sft_step = 340`
  - `best_step = 340`
  - `best_val_loss = 2.8786`
- State dict comparison between `sft_best_checkpoint.pt` and `sft_checkpoint_step_340.pt`: **0 differences across all 57 tensor parameters**.
- State dict comparison between `sft_best_checkpoint.pt` and `sft_checkpoint_step_600.pt`: **57 tensor parameter differences**.

**Finding**: The checkpoint evaluated for Exp19 was unambiguously the genuine best validation checkpoint from Step 340, not the over-fitted Step 600 checkpoint.

---

## 3. Supplementary Inference: Step 340 vs Step 600 Generation Quality

Inference was run across a representative test subset using identical decoding parameters ($T=0.0$, max tokens 45, seed 42):
- In `factual_qa`: Both checkpoints produced repetition loops on `"city of the city"`.
- In `basic_arithmetic`: Step 340 produced `"The difference of 2005."` while Step 600 produced `"The difference of 170."` (both incorrect).
- In `structured_formatting`: Step 340 produced `'{"style={"style="}'` while Step 600 produced `'{"session_pathsvpx"}'`.
- In `short_explanation`: Both checkpoints produced identical output: `"The difference is the planets is the planet."`

**Conclusion**: Over-training beyond Step 340 slightly altered token choices but did not rescue semantic correctness or escape code/attribute attractor basins.
