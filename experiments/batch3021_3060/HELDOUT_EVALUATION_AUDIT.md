# Held-Out Evaluation Set Audit — Batch 3021–3060
## Independence Verification, Benchmark Re-Classification & Contamination Report

### 1. Executive Summary

A core requirement of Batch 3021–3060 is to independently audit all existing and newly created evaluation benchmarks to distinguish genuine out-of-distribution evaluation from in-distribution regression testing.

Four distinct evaluation suites were audited:
1. **Historical 24-Case SFT Benchmark** (`experiments/batch2821_2860/SFT_EVALUATION_SET.json`)
2. **Historical 28-Case Generation Benchmark** (`experiments/batch2781_2820/GENERATION_BENCHMARK.json`)
3. **Current Balanced SFT 40-Case Test Split** (`experiments/batch2941_2980/test.jsonl`)
4. **New Document-Isolated 40-Case Test Split** (`experiments/batch3021_3060/test.jsonl`)

---

### 2. Comprehensive Benchmark Classification

| Evaluation Suite | Size | Composition | Overlap with Balanced Train Set | True Status / Role |
| :--- | :--- | :--- | :--- | :--- |
| **Historical SFT Benchmark** (`batch2821`) | 24 cases | 8 categories (3 cases each) | **21 of 24 cases (87.5%)** | **Historical Regression Suite Only**. Cannot serve as an independent held-out benchmark. |
| **Historical Generation Benchmark** (`batch2781`) | 28 cases | 14 raw completion, 14 plain instruction | **0 cases (0.0%)** | **Independent Behavioral Regression Suite**. Measures `<EOS>` regression and instruction format compliance. |
| **Batch 2941 Test Split** | 40 cases | 28 instruction (4 x 7 cats), 12 replay | **0 prompt overlaps**, but 9 replay cases came from `corpus_code_foundation` which crossed into `val`. | **Partially Isolated Test Set** (Replay document crossing defect). |
| **Batch 3021 Test Split** | 40 cases | 28 instruction (4 x 7 cats), 12 replay | **0 prompt overlaps, 0 document crossings** | **Fully Isolated Independent Test Set**. Strict document-level pre-split partition. |

---

### 3. Detailed Contamination Breakdown of the Historical 24-Case Suite

When Batch 2941–2980 created `build_balanced_dataset.py`, it imported all instruction cases from Batch 2821–2860 and assigned indices 0..31 of each category to `train.jsonl`.
Because cases 28..30 in Batch 2821 were the held-out test split, these cases were directly absorbed into the training set:
- `INST_028`, `INST_029`, `INST_030` -> In `train.jsonl`
- `EXTR_028`, `EXTR_029`, `EXTR_030` -> In `train.jsonl`
- `CQA_028`, `CQA_029`, `CQA_030` -> In `train.jsonl`
- `SUMM_028`, `SUMM_029`, `SUMM_030` -> In `train.jsonl`
- `STR_028`, `STR_029`, `STR_030` -> In `train.jsonl`
- `UNC_028`, `UNC_029`, `UNC_030` -> In `train.jsonl`
- `TRNS_028`, `TRNS_029`, `TRNS_030` -> In `train.jsonl`
- *(Note: `CONV_028..030` were omitted as conversational acknowledgement was excluded from the 7 primary instruction categories).*

**Conclusion:** The historical 24-case benchmark was not silently modified or overwritten. It remains in its original location as a historical regression record, but it **must never be reported as an independent validation metric for models trained on the balanced dataset**.

---

### 4. Establishment of the Trustworthy Batch 3021 Held-Out Test Set

The new test split in `experiments/batch3021_3060/test.jsonl` provides strict isolation:
1. **Instruction Portion (28 examples, 7 categories x 4 cases):**
   - Distinct prompts, distinct responses, and distinct underlying facts.
   - Zero overlap with `train.jsonl` or `val.jsonl`.
2. **Replay Portion (12 examples):**
   - Extracted exclusively from `corpus_code_foundation` and `corpus_math_foundation`.
   - Neither of these two source documents appears in `train.jsonl` or `val.jsonl`.
   - Proves whether raw continuation capability generalizes to completely unseen code and mathematical texts without producing premature `<EOS>`.
