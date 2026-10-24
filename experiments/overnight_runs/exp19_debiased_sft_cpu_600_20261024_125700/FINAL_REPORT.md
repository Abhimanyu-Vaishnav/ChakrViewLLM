# ChakrView Exp19 — Final Engineering & Evaluation Report

**Mission**: `exp19_debiased_sft_cpu_600`  
**Date**: October 24, 2026  
**Workspace**: `d:\Project\ChakrView`  
**Artifact Directory**: `experiments/overnight_runs/exp19_debiased_sft_cpu_600_20261024_125700/`  

---

## 1. Executive Summary

This report documents the authorized execution of **ChakrView Exp19** (`exp19_debiased_sft_cpu_600`), an isolated experimental fine-tuning run for ChakrMicro on CPU. The primary objective was to investigate whether deterministic response-template diversification resolves the severe template collapse observed in Exp18 (where 77.5% of responses began with `"The difference"`) while measuring its downstream impact on task accuracy, EOS termination, response length, latency, and structured JSON formatting.

### Primary Verdict: `COMPLETED_MIXED`
- **600 Steps Completed**: Exactly 600 optimizer steps completed on CPU in 47.24 seconds with zero loss explosions or gradient anomalies.
- **Repository Integrity Preserved**: All 9 protected repository assets bitwise match their audited SHA-256 signatures before and after the run. Canonical checkpoints and datasets were untouched.
- **Regression Suite Clean**: 126/126 unit tests passed (0 failures, 0 skipped).
- **Generation Behavior Materially Improved**:
  - EOS termination rate climbed from **45.00% (72/160)** to **54.37% (87/160)**, crossing the $\ge 50\%$ operational target.
  - In `structured_formatting`, EOS termination surged to **100% (20/20)** from 55% in Exp18.
  - Maximum opening prefix concentration collapsed from **77.50%** down to **38.12%**, effectively breaking the monoculture.
  - Average generated tokens decreased from ~41.0 to **29.71**, eliminating runaway generation loops and accelerating CPU inference latency by ~15% (282.78 ms vs 332.24 ms).
- **Task Correctness Did Not Improve**:
  - Scorable task accuracy measured **0.00% (0/140)** compared to **0.71% (1/140)** in Exp18. In a 140-sample benchmark, this delta represents a single sample shift ($\Delta = -0.71\%$), which is within expected statistical variation.
  - Structured JSON validity remained at **0.00% (0/20)** due to pre-training code priors (`import doctest`) dominating output sequences.

---

## 2. Preflight Gates & Audit Trail

All mandatory preflight checks passed with full bitwise verification:
1. **Source Checkpoint**:
   - `experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt`
   - Verified SHA-256: `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` (Match).
2. **Isolated Starting Copy**:
   - Copied to `checkpoints/training_checkpoint_start.pt`.
   - Verified copy SHA-256: `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` (Match).
3. **Protected Assets (Pre-Run & Post-Run)**:
   - Architecture Digest (seed 42): `c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da`
   - Tokenizer `vocab.json`: `db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a`
   - Tokenizer `merges.json`: `ada4b1dcfbb3d962a2e60203089881872a5063de73b2fef9eeff81ce0016efa7`
   - Protected `chakrview/runtime/session.py`: `9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67`
   - Model `chakrview/brain/model.py`: `1343d7d18224cc308465b7f07eb6d889532e7f17cee74b5f84f878e542f72361`
   - Exp17 Train Split: `074fafe5c1dbcbd7467430cb6f5eccd2c11f2f8679ff27e13c0d38cc90bac181`
   - Exp17 Val Split: `15d9c942c01fb6afaf6280b7515d9e621ff8df7740be8010daa2be98841f4564`
   - Exp17 Test Split: `be10afaa3957713cead5399c2b01e5ad2f4fb61d6355835432c7d6e0fdcc6430`
   - All 9 assets verified intact both before training and after final evaluation.
4. **Dataset Verification**:
   - `exp19_train_debiased_isolated.jsonl`: 1,200 records (`SHA-256: f0479f0ee95d4656f956b22dd38321ccf43ca822aeae999d90f512f0ed476e2b`).
   - `exp19_val_isolated.jsonl`: 160 records (`SHA-256: 1cd80c6034d96f2687dbcd715f04bae07d78cf405c91b5bc72a273698d4cdce0`).
   - `exp19_test_frozen_isolated.jsonl`: 160 records (`SHA-256: a3cc3c92cb9d4b59c983333abc941e68815af9fb30867c7c68bcb5951395bb68`).
   - Held-out test manifest content hash: `111db453d5ee52d436ef9a49f1ad6555ea572d118f84099ea0763e7f299ba047` (Exact match to frozen manifest).
   - Maximum 2-word opening concentration across all categories verified $\le 24.67\% \le 25.00\%$.

---

## 3. Training Execution Summary

- **Configuration**: 600 steps, batch size 2, cosine LR schedule ($5\times 10^{-5} \to 5\times 10^{-6}$), weight decay 0.01, gradient clip 1.0, seed 42.
- **Initial Validation Loss**: 4.5025 (PPL: 90.24)
- **Minimum Validation Loss**: **2.8786** (PPL: 17.79) achieved at **Step 340**.
- **Final Validation Loss**: 3.3124 (PPL: 27.45) at Step 600.
- **Best Checkpoint**: Automatically isolated and preserved at [checkpoints/sft_best_checkpoint.pt](file:///d:/Project/ChakrView/experiments/overnight_runs/exp19_debiased_sft_cpu_600_20261024_125700/checkpoints/sft_best_checkpoint.pt).

---

## 4. Evaluation & Paired Comparison Against Exp18

Evaluation performed using the frozen evaluator on all 160 held-out test records:

| Metric | Exp18 Baseline | Exp19 Debiased | Delta |
| :--- | :---: | :---: | :---: |
| **Total Test Records** | 160 | 160 | 0 |
| **Scorable Records** | 140 | 140 | 0 |
| **Correct Records** | 1 | 0 | -1 |
| **Scorable Accuracy** | 0.71% | **0.00%** | **-0.71%** |
| **EOS Terminations** | 72 | 87 | **+15** |
| **EOS Completion Rate** | 45.00% | **54.37%** | **+9.37%** |
| **Max Prefix Concentration** | 77.50% | **38.12%** | **-39.38%** |
| **Avg Tokens Generated** | ~41.0 | **29.71** | **-11.29** |
| **Avg Latency (CPU)** | 332.24 ms | **282.78 ms** | **-49.46 ms** |
| **Valid JSON (Structured)** | 0.00% (0/20) | **0.00%** (0/20) | 0.00% |

---

## 5. Artifact Index

All required experiment deliverables reside in:
`experiments/overnight_runs/exp19_debiased_sft_cpu_600_20261024_125700/`

- [RUN_MANIFEST.json](file:///d:/Project/ChakrView/experiments/overnight_runs/exp19_debiased_sft_cpu_600_20261024_125700/RUN_MANIFEST.json): Comprehensive machine-readable execution manifest.
- [PREFLIGHT_RESULTS.json](file:///d:/Project/ChakrView/experiments/overnight_runs/exp19_debiased_sft_cpu_600_20261024_125700/PREFLIGHT_RESULTS.json): Pre-run gate verifications.
- [DATASET_INTEGRITY_REPORT.json](file:///d:/Project/ChakrView/experiments/overnight_runs/exp19_debiased_sft_cpu_600_20261024_125700/DATASET_INTEGRITY_REPORT.json): Verification of split counts, isolation, and opening concentrations.
- [TRAINING_CONFIG.json](file:///d:/Project/ChakrView/experiments/overnight_runs/exp19_debiased_sft_cpu_600_20261024_125700/TRAINING_CONFIG.json): Exact hyperparameters and paths.
- [TRAINING_LOG.jsonl](file:///d:/Project/ChakrView/experiments/overnight_runs/exp19_debiased_sft_cpu_600_20261024_125700/TRAINING_LOG.jsonl): Step-by-step training and validation log (600 lines).
- [TRAINING_SUMMARY.md](file:///d:/Project/ChakrView/experiments/overnight_runs/exp19_debiased_sft_cpu_600_20261024_125700/TRAINING_SUMMARY.md): Detailed narrative on training dynamics and convergence.
- [checkpoints/training_checkpoint_start.pt](file:///d:/Project/ChakrView/experiments/overnight_runs/exp19_debiased_sft_cpu_600_20261024_125700/checkpoints/training_checkpoint_start.pt): Isolated starting checkpoint copy.
- [checkpoints/sft_best_checkpoint.pt](file:///d:/Project/ChakrView/experiments/overnight_runs/exp19_debiased_sft_cpu_600_20261024_125700/checkpoints/sft_best_checkpoint.pt): Selected best validation checkpoint (Step 340).
- [EVAL_EXP19_RESULTS.json](file:///d:/Project/ChakrView/experiments/overnight_runs/exp19_debiased_sft_cpu_600_20261024_125700/EVAL_EXP19_RESULTS.json): Summary statistics across 160 test records.
- [EVAL_EXP19_PER_RECORD.jsonl](file:///d:/Project/ChakrView/experiments/overnight_runs/exp19_debiased_sft_cpu_600_20261024_125700/EVAL_EXP19_PER_RECORD.jsonl): Full per-example test outputs and scoring annotations.
- [EXP18_VS_EXP19_COMPARISON.md](file:///d:/Project/ChakrView/experiments/overnight_runs/exp19_debiased_sft_cpu_600_20261024_125700/EXP18_VS_EXP19_COMPARISON.md): In-depth paired comparison report.
- [REGRESSION_TEST_RESULTS.txt](file:///d:/Project/ChakrView/experiments/overnight_runs/exp19_debiased_sft_cpu_600_20261024_125700/REGRESSION_TEST_RESULTS.txt): Full test suite output (126 passed, 0 failed).
- [POST_RUN_PROTECTED_HASHES.json](file:///d:/Project/ChakrView/experiments/overnight_runs/exp19_debiased_sft_cpu_600_20261024_125700/POST_RUN_PROTECTED_HASHES.json): Verification of all 9 protected asset hashes post-execution.
- [FINAL_REPORT.md](file:///d:/Project/ChakrView/experiments/overnight_runs/exp19_debiased_sft_cpu_600_20261024_125700/FINAL_REPORT.md): This document.
