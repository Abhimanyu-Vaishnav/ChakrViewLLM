# Final SFT Invocation Verification Audit — Batch 3021–3060
## End-to-End Call Chain & Training Safety Verification (Zero Training Executed)

### Executive Summary

This audit independently traces and validates the future training invocation path for the balanced SFT experiment:
- Preflight Script: `experiments/batch3021_3060/run_preflight.py`
- Configuration File: `experiments/batch3021_3060/SFT_EXPERIMENT_CONFIG.json`
- Dataset Manifest: `experiments/batch3021_3060/SPLIT_ASSIGNMENTS.json`
- Execution Engine: `chakrview/training/sft_engine.py`

**Mandatory Guarantee:** Zero training, fine-tuning, optimizer steps, or background tasks were executed.

---

### Detailed Verification Checklist

#### 1. Dataset Loading Isolation
- **Verified Files:**
  - `train_file`: `experiments/batch3021_3060/train.jsonl` (320 records: 224 instruction, 96 replay).
  - `val_file`: `experiments/batch3021_3060/val.jsonl` (40 records: 28 instruction, 12 replay).
  - `test_file`: `experiments/batch3021_3060/test.jsonl` (40 records: 28 instruction, 12 replay).
  - `dataset_full`: `experiments/batch3021_3060/BALANCED_SFT_DATASET.jsonl` (400 total records).
- **Audit Finding:** The configuration file and engine strictly load paths located inside `experiments/batch3021_3060/`. The older dataset in `experiments/batch2941_2980/` (which had replay document crossings and benchmark overlap) is **not loaded**. Zero fallback mechanisms exist.

#### 2. Reconciled Optimizer & Training Hyperparameters
- **Optimizer Steps:** Exactly **160 steps** (2.0 full epochs at batch size 4 across 320 training examples).
- **Batch Size:** 4.
- **Random Seed:** 42 (reproducible NumPy, PyTorch CPU, and Python random seeds).
- **Learning Rate:** `5.0e-5` with cosine decay to `5.0e-6` and 10-step warmup.
- **Regularization:** AdamW with weight decay `0.01` and gradient clipping `1.0`.
- **Validation Cadence:** Evaluates all 40 validation records every 10 steps (16 validation checks).
- **Periodic Checkpoint Cadence:** Saves checkpoints every 20 steps (steps 20, 40, 60, 80, 100, 120, 140, 160).
- **Best-Checkpoint Saving:** Atomic saving to `sft_best_checkpoint.pt` whenever validation loss achieves a new global minimum.

#### 3. Strict Held-Out Evaluation Set Isolation
- In `sft_engine.py`:
  - `self.train_dataset` is constructed from `config.train_file` (`train.jsonl`).
  - `self.val_dataset` is constructed from `config.val_file` (`val.jsonl`).
  - `evaluate()` evaluates exclusively over `self.val_dataset`.
  - The held-out test split `experiments/batch3021_3060/test.jsonl` is **never loaded or accessed** during training, validation loss calculation, early stopping, or checkpoint selection.

#### 4. Checkpoint & Artifact Output Isolation
- Target Output Directory: `experiments/batch3021_3060/balanced_sft_pilot_run/`.
- Path safety guards in `sft_engine.py` and `run_preflight.py` explicitly reject paths pointing to canonical checkpoint directories (`main_training_10000`, `artifacts/step48`, `release/`).
- Prior run directories (such as `experiments/batch2821_2860/sft_pilot_run/`) cannot be overwritten.

#### 5. Pre-Execution Identity Verification
Before initializing model weights or reading tensors:
- SHA-256 of `training_checkpoint_step_10000.pt` is computed and matched against `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee`.
- Tokenizer files in `release/chakrview-v0.5.0/data/experiments/vocab_4096/` are loaded and confirmed to match `vocab_size = 4096`.
- Model state dict is strictly loaded with `strict=True`, matching all 2,510,400 parameters.

#### 6. Fail-Closed Gate Enforcement
In `run_preflight.py`:
- All 7 gates must pass with status `PASS`.
- Any single failure sets `overall_pass = False`, writes `FINAL_PREFLIGHT_REPORT.json` with status `FAIL`, and exits immediately with code 1.
- Without the mandatory CLI flag `--execute-authorized-training`, the script operates purely in read-only inspection mode.
- In this batch, `--execute-authorized-training` was **never invoked**.

#### 7. Best-Checkpoint and Periodic Checkpoint Wiring
In `sft_engine.py`:
- Initial best validation loss is initialized to `inf`.
- At each validation interval (`step % 10 == 0`), if `cur_vloss < self.best_val_loss` (strictly less than, resolving ties deterministically to the earlier checkpoint), the model state and optimizer state are written to `sft_best_checkpoint.pt.tmp` and atomically replaced to `sft_best_checkpoint.pt`.
- At each periodic interval (`step % 20 == 0`), periodic checkpoints `sft_checkpoint_step_{step}.pt` are written.

---

### Unresolved Risks & Operational Limitations

1. **Model Parameter Capacity:** ChakrMicro v0.1 has 2.5M parameters and a 512-token context window. While balanced 70/30 training mitigates premature `<EOS>` output on continuation prompts, out-of-distribution reasoning and open-ended factual knowledge remain strictly bounded by model scale.
2. **Execution Gate Requirement:** The future training run must not be launched without explicit operator authorization supplied directly at invocation time.

---

### Verdict

**`READY FOR EXPLICIT AUTHORIZATION`**
