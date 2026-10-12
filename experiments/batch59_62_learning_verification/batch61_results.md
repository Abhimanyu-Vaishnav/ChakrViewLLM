# Batch 60 & 61: Controlled Learning Experiment & Generalization Findings

## 1. Experimental Design (Batch 60)
To establish whether the ChakrMicro model exhibits genuine algorithmic generalization vs prompt-conditioned memorization, we created a strictly partitioned synthetic arithmetic suite:
- **Task**: Addition mapping `Compute: a + b = ?` $\to$ `ans<|im_end|>\n`
- **Total Unique Pairs**: 48 pairs ($1 \le a, b \le 20$)
- **Splits**:
  - `TRAIN`: 32 records
  - `VALIDATION`: 8 held-out records (0% overlap with TRAIN)
  - `HELD_OUT_BENCHMARK`: 8 held-out records (0% overlap with TRAIN or VALIDATION)
- **Generator**: Deterministic generator with fixed seed (`42`), recorded in `synthetic_dataset_manifest.json`.

## 2. Quantitative Evaluation: Baseline vs Trained Step 50 (Batch 61)

| Metric | Baseline Step 10,000 | Trained Step 50 |
|---|---|---|
| **TRAIN Exact Match** | 0 / 32 (0.0%) | 0 / 32 (0.0%) |
| **TRAIN Token Loss** | 6.9797 | 0.3823 |
| **VALIDATION Exact Match** | 0 / 8 (0.0%) | 0 / 8 (0.0%) |
| **VALIDATION Token Loss** | 6.9971 | 0.4642 |
| **BENCHMARK Exact Match** | 0 / 8 (0.0%) | 0 / 8 (0.0%) |
| **BENCHMARK Token Loss** | 7.0321 | 0.3612 |
| **EOS Completion Rate** | 0.0% | **100.0%** |
| **Repetition Rate** | 33.3% | **0.0%** |

## 3. Critical Scientific Discovery: Memorization vs Generalization
1. **Loss Reduction is NOT Generalization**:
   - Token-level cross-entropy loss dropped dramatically from ~7.00 to ~0.38 across all splits, and perplexity dropped from >1000 to ~1.50.
   - EOS emission achieved 100% on every single prompt.
   - However, **Exact Match remained 0.0%** across all splits.

2. **The "Narrow Memorization Shortcut" Dissected**:
   - In single-example training (`What is 2 + 3?` $\to$ `5`), the model output `5` with 99.92% probability.
   - When presented with the unseen input `What is 3 + 4?`, the model ALSO output `5` with **99.91% probability**.
   - This proves conclusively that the single-example experiment **did not learn addition**; it merely memorized a static mapping from the ChatML prompt template to the token `5`.

3. **Wikipedia Table Attractor Status**:
   - In the base model, token `127` (`|`) had 91.86% probability at the assistant turn.
   - After SFT on arithmetic, the model learned the ChatML transition and EOS termination (`<|im_end|>\n`), but when fine-tuned across diverse numeric answers without thousands of steps, the pretraining bias still pulled generation toward table formatting (`|277`).
