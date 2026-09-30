# ChakrView Step 50: Checkpoint Audit & Selection Report

- **Date**: 2026-09-30
- **Phase**: Step 50 — Trained ChakrView Interactive Model Evaluation
- **Target Architecture**: ChakrMicro v0.1 (Decoder-only causal transformer)
- **Parameters**: 3,443,136
- **Vocabulary Size**: 4,096
- **Context Length**: 512
- **Frozen Baseline SHA-256**: `c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da`
- **Active Tokenizer Checksum**: `7498d92adeef7c9e64d4d1bbd5b4b23f8774692d868742e3b98e30c584c4403f`

---

## 1. Checkpoint Repository Audit

A comprehensive scan was conducted across `artifacts/`, `checkpoints/`, `models/`, and `data/` to discover all existing checkpoints.

### Discovered Checkpoints and Comparative Metrics

| Checkpoint Path | Type | Training Steps | Val Loss | Val PPL | Tokenizer Match | Selected |
|:---|:---:|---:|---:|---:|:---:|:---:|
| `artifacts/step48/run_seed_42/checkpoint_0000100.pt` | training | 100 | **2.5324** | **12.58** | **YES** | **SELECTED** |
| `artifacts/step48/run_seed_42/checkpoint_0000050.pt` | training | 50 | 3.7553 | 42.75 | YES | NO |
| `checkpoints/stage_b_learning_validation/checkpoint_0000200.pt` | untyped | 200 | 2.8915 | 18.02 | NO (missing metadata) | NO |
| `checkpoints/stage_b_resume_test/checkpoint_0000030.pt` | untyped | 30 | 4.0637 | 58.19 | NO (missing metadata) | NO |
| `checkpoints/stage_b_learning_validation/checkpoint_0000300.pt` | untyped | 300 | 4.7115 | 111.22 | NO (missing metadata) | NO |
| `checkpoints/stage_b_learning_validation/checkpoint_0000400.pt` | untyped | 400 | 5.0482 | 155.75 | NO (missing metadata) | NO |
| `checkpoints/stage_b_learning_validation/checkpoint_0000500.pt` | untyped | 500 | 5.5208 | 249.84 | NO (missing metadata) | NO |
| `checkpoints/stage_c_full_epoch/checkpoint_0005000.pt` | untyped | 5000 | 5.9571 | 386.47 | YES | NO |
| `checkpoints/stage_c_full_epoch/checkpoint_0006000.pt` | untyped | 6000 | 6.0348 | 417.73 | YES | NO |
| `checkpoints/stage_c_full_epoch/checkpoint_0006478.pt` | untyped | 6478 | 6.6316 | 758.69 | YES | NO |
| `checkpoints/stage_b_learning_validation/checkpoint_0000100.pt` | untyped | 100 | 7.3593 | 1570.73 | NO (missing metadata) | NO |
| `checkpoints/stage_c_baseline/checkpoint_0000500.pt` | untyped | 500 | 7.6013 | 2000.88 | YES | NO |
| `checkpoints/stage_c_full_epoch/checkpoint_0004500.pt` | untyped | 4500 | 7.6840 | 2173.35 | YES | NO |
| `checkpoints/stage_c_baseline/checkpoint_0000100.pt` | untyped | 100 | 7.7082 | 2226.52 | YES | NO |

---

## 2. Selection Rationale

The selected checkpoint for Step 50 interactive model evaluation is:
**`artifacts/step48/run_seed_42/checkpoint_0000100.pt`**

### Key Criteria Met:
1. **Best Validation Loss & PPL:** Achieved validation loss of **2.5324** and perplexity of **12.58** on the held-out validation corpus (down from pre-training uniform loss $8.3691$ / PPL $4311.62$).
2. **Strict Typology Adherence:** Contains explicit `checkpoint_type: "training"`, validated in Step 47 and ratified in Step 48.
3. **Exact Tokenizer Compatibility:** Checkpoint metadata explicitly embeds tokenizer checksum `7498d92adeef7c9e64d4d1bbd5b4b23f8774692d868742e3b98e30c584c4403f`, perfectly matching active tokenizer artifacts in `data/experiments/vocab_4096`.
4. **Architectural Invariance:** Exactly 3,443,136 unique parameters across 56 tensors, all with finite numerical values (no NaNs, no Infs).
5. **Distinct from Baseline:** Model weight SHA-256 is `85e1eb6cd3472d431692cde71cf58f705990fee50be91a2d70f46e0968481a64`, ensuring clear behavioral comparison against frozen baseline (`c5571c...`).
