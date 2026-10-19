# Training Configuration Reconciliation — Batch 3021–3060
## Step Limit Resolution, Execution Path Parity & Cadence Alignment

### 1. Executive Summary

A discrepancy was identified between:
- The proposed text in `experiments/batch2941_2980/NEXT_TRAINING_PROPOSAL.md` (which cited **160 steps** for roughly 2 epochs of 80 batches).
- The actual configuration file `experiments/batch2941_2980/SFT_EXPERIMENT_CONFIG.json` (which specified **150 steps**).

This audit reconciles all parameters across the proposal, the experiment config, the training engine, and the preflight script, selecting a single unified pilot limit.

---

### 2. Discrepancy Analysis & Step Limit Selection

- **Training Split Size:** 320 records
- **Batch Size:** 4 records per batch
- **Batches per Epoch:** `320 / 4 = 80 batches`
- **Candidate Limits:**
  - `150 steps`: Exactly 1.875 epochs. Saves checkpoints at steps 25, 50, 75, 100, 125, 150.
  - `160 steps`: Exactly 2.0 full epochs (`2 * 80`). Periodic saves at steps 20, 40, 60, 80, 100, 120, 140, 160.
- **Decision:** **Adopt 160 steps** as the authoritative pilot limit:
  - Completes exactly 2 full epochs over all 320 training examples.
  - Prevents an uneven fractional pass where the first 70% of the dataset would receive 2 passes while the remaining 30% receives only 1.
  - Aligns cleanly with a validation cadence of every 10 steps (16 validation checks) and periodic checkpoint cadence of every 20 steps (8 periodic checkpoints + best checkpoint).

---

### 3. Reconciled Configuration Matrix

| Configuration Parameter | Reconciled Value | Source / Enforcement Mechanism |
| :--- | :--- | :--- |
| **Experiment ID** | `batch3021_3060_balanced_sft_pilot` | `SFT_EXPERIMENT_CONFIG.json` |
| **Model Architecture** | ChakrMicro v0.1 (`vocab=4096, d_model=192, layers=6, heads=6, ffn=512, ctx=512`) | Frozen specification (`chakrview/brain/config.py`) |
| **Parent Checkpoint** | `experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt` | Verified SHA-256: `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` |
| **Tokenizer Directory** | `release/chakrview-v0.5.0/data/experiments/vocab_4096` | Verified vocab size 4096, BOS=0, EOS=1, PAD=2 |
| **Dataset Paths** | `experiments/batch3021_3060/train.jsonl` (320), `val.jsonl` (40), `test.jsonl` (40) | Strict document-isolated dataset |
| **Optimizer Steps** | **160 steps** (Exactly 2 full epochs) | Enforced by `config.steps = 160` |
| **Batch Size** | 4 | 80 batches per epoch |
| **Learning Rate** | `5.0e-5` (min `5.0e-6`) | Cosine decay with warmup |
| **Weight Decay** | 0.01 | AdamW standard |
| **Gradient Clipping** | 1.0 | Standard norm clip |
| **Validation Cadence**| Every 10 steps | Evaluates all 40 validation records |
| **Checkpoint Cadence**| Every 20 steps | Saves steps 20, 40, 60, 80, 100, 120, 140, 160 |
| **Best-Checkpoint Saving** | Enabled (`sft_best_checkpoint.pt`) | Atomic safe replacement via `.tmp` file |
| **Random Seed** | 42 | Full determinism |
| **Execution Hardware**| CPU (`torch.float32`) | Standard CPU training environment |
| **Output Directory** | `experiments/batch3021_3060/balanced_sft_pilot_run` | Isolated from canonical artifacts |

---

### 4. Verification of Fail-Closed Preflight Protection

The preflight engine enforces that:
1. Every gate must evaluate to `PASS`.
2. Any single gate failure sets `overall_pass = False`.
3. If `overall_pass == False`, the script emits `FINAL_PREFLIGHT_REPORT.json` with status `FAIL` and calls `sys.exit(1)`.
4. The execution flag `--execute-authorized-training` is evaluated only after all gates pass.
5. In this batch, `--execute-authorized-training` has **never been invoked**.
