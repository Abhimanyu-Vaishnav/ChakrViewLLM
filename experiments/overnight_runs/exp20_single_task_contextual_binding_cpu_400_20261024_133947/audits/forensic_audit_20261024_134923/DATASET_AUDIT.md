# Exp20 Dataset Audit & Parity Verification

**Audit Directory**: `experiments/overnight_runs/exp20_single_task_contextual_binding_cpu_400_20261024_133947/audits/forensic_audit_20261024_134923/`  

---

## 1. Dataset Integrity & Split Statistics

All records in the three Exp20 dataset files were parsed and audited:
- `data/exp20_train_context_extraction.jsonl`: Exactly **400 records**.
- `data/exp20_val_context_extraction.jsonl`: Exactly **50 records**.
- `data/exp20_test_frozen.jsonl`: Exactly **20 records**.

### Required Field & Category Validation
- Every record across all three splits contains all 9 required schema fields: `id`, `split`, `category`, `prompt_mode`, `prompt`, `context`, `formatted_prompt`, `response`, `expected_intent`.
- 100% of records in all splits have `category == "context_extraction"`.
- Zero malformed lines, missing fields, or empty strings were detected.

---

## 2. Split Isolation & Entity Overlap Audit

Entity names were programmatically extracted from the ground-truth responses:
- **Train Split**: 50 unique entity names.
- **Validation Split**: 10 unique entity names.
- **Test Split**: 10 unique entity names.

### Overlap Calculations
- **Train $\cap$ Test Entity Overlap**: **0 / 50 (0.00%)**
- **Train $\cap$ Val Entity Overlap**: **0 / 50 (0.00%)**
- **Val $\cap$ Test Entity Overlap**: **0 / 10 (0.00%)**
- **Train $\cap$ Test Exact Prompt Overlap**: **0 / 400 (0.00%)**

**Finding**: Strict, leak-free split isolation is confirmed. No entity or mission identifier leaked across dataset boundaries.

---

## 3. Field-by-Field Parity with Exp19 Frozen Test Set

The 20 context extraction records from Exp19 (`experiments/overnight_runs/exp19_debiased_sft_cpu_600_20261024_125700/data/exp19_test_frozen_isolated.jsonl`) were loaded and compared against `exp20_test_frozen.jsonl`:
- **ID Match**: 20 / 20
- **Prompt Match**: 20 / 20
- **Context Match**: 20 / 20
- **Expected Response Match**: 20 / 20
- **Prompt Mode Match**: 20 / 20 (`context_qa`)
- **Formatted Prompt Match**: 20 / 20
- **Expected Intent Match**: 20 / 20

**Finding**: Exp20 tested against the **exact bitwise identical context extraction benchmark** used in Exp18 and Exp19. The test set was not altered or eased.
