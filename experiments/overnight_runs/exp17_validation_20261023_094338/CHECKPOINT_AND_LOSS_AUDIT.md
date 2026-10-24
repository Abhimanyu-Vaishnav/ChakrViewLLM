# ChakrView Exp17 Validation — Checkpoint & Loss Trajectory Audit

**Audit Target Checkpoints**:
1. `experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt` (Canonical Base)
2. `experiments/overnight_runs/mission_20261023_003512/sft_run/sft_best_checkpoint.pt` (Best SFT Step 225)
3. `experiments/overnight_runs/mission_20261023_003512/sft_run/sft_checkpoint_step_400.pt` (Final SFT Step 400)

---

## 1. Checkpoint Inventory & Integrity Metadata

| Checkpoint Name | Relative Path | SHA-256 | Internal `sft_step` | Best Step | Best Val Loss |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Canonical Base** | `experiments/batch16_21_combined/.../step_10000.pt` | `96357705...` | Pretrained | — | — |
| **SFT Best** | `experiments/overnight_runs/.../sft_best_checkpoint.pt` | `fa9b5e8c...` | 225 | 225 | 1.9602 |
| **SFT Step 400** | `experiments/overnight_runs/.../sft_checkpoint_step_400.pt` | `fc912247...` | 400 | 225 | 1.9602 |

- **Weight Hash Verification**:
  - `sft_best_checkpoint.pt`: Expected weight digest `276ba09b80faaf40e82ebdbade5884b77f4132f3454afc8f06b2bab1635e248b`.
  - `sft_checkpoint_step_400.pt`: Expected weight digest `dcf6f548c5c71f85c479b9fe858feb5e81d91c1233079fcafc5239f28a943f89`.
- **Finding**: The best checkpoint and step 400 checkpoint are distinct state dicts. The best checkpoint correctly captures the Step 225 model state rather than being an alias of the final step.

---

## 2. Validation Trajectory & Perplexity Mathematics

The SFT engine evaluates prompt-masked validation loss across 40 validation sequences at 25-step intervals.

### Trajectory Table (Full 400 Steps from Step 400 Checkpoint History):

| Step | Train Loss | Validation Loss | Perplexity ($\exp(\mathcal{L})$) | Best Flag | Checkpoint Saved |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **0** | — | 4.8618 | 129.25 | — | Baseline initialization |
| **25** | 4.6541 | 3.6746 | 39.43 | True | `sft_checkpoint_step_25.pt` |
| **50** | 2.9893 | 3.7808 | 43.85 | False | — |
| **75** | 3.0256 | 3.1423 | 23.16 | True | `sft_checkpoint_step_75.pt` |
| **100** | 2.1563 | 2.8433 | 17.17 | True | `sft_checkpoint_step_100.pt` |
| **125** | 2.6729 | 2.8686 | 17.61 | False | — |
| **150** | 4.0878 | 3.1112 | 22.45 | False | — |
| **175** | 3.4072 | 2.8966 | 18.11 | False | — |
| **200** | 1.3694 | 2.2236 | 9.24 | True | `sft_checkpoint_step_200.pt` |
| **225** | 2.3736 | **1.9602** | **7.10** | True | **`sft_best_checkpoint.pt` / `step_225.pt`** |
| **250** | 2.2705 | 2.4273 | 11.33 | False | — |
| **275** | 0.7883 | 2.1066 | 8.22 | False | — |
| **300** | 2.8818 | 2.3050 | 10.02 | False | `sft_checkpoint_step_300.pt` |
| **325** | 2.4622 | 2.1355 | 8.46 | False | — |
| **350** | 1.1217 | 2.1631 | 8.70 | False | — |
| **375** | 0.9260 | 2.8326 | 16.99 | False | — |
| **400** | 1.1529 | 2.6766 | 14.54 | False | `sft_checkpoint_step_400.pt` |

### Math Verification:
- Perplexity Formula: $\text{PPL} = \exp(\mathcal{L}_{\text{val}})$ where $\mathcal{L}$ is token-weighted cross-entropy loss over active response positions ($\mathcal{A}_i$).
- At Step 225: $\exp(1.9602) = 7.1007 \approx \mathbf{7.10}$.
- At Step 0: $\exp(4.8618) = 129.256 \approx \mathbf{129.25}$.
- Finding: **Perplexity math is 100% verified and mathematically consistent**.

---

## 3. Discrepancy Note in `TRAINING_REPORT.md`

In the overnight mission's markdown report `TRAINING_REPORT.md`, the author transcribed train/validation numbers from an earlier draft rather than copying the exact raw step history from `sft_checkpoint_step_400.pt`.
- Reported in `TRAINING_REPORT.md`: Step 25 Val Loss = 4.2180, Step 225 Train Loss = 1.3120.
- Actual Raw State Dict in Checkpoint: Step 25 Val Loss = 3.6746, Step 225 Train Loss = 2.3736.
- Critical Finding: Despite the markdown transcription mismatch, **the global minimum validation loss of 1.9602 at Step 225 is 100% corroborated by the actual raw checkpoint binary**.
