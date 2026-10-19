# ChakrView SFT V3 Grounded Dataset Builder & Preflight Validation

This directory contains the dataset generation pipeline, automated preflight validator, frozen dataset splits, and audit reports for the SFT V3 Grounded Curriculum milestone.

## Directory Structure & Files

- `README.md`: Overview, reproduction commands, output manifest, and safety rules.
- `SFT_V3_DATA_CONTRACT.md`: Versioned specification of record schemas, ground-truth rules, delimiter conventions, and loss-masking contracts.
- `DATASET_AUDIT_REPORT.md`: Comprehensive audit report detailing counts, leakage audits, preflight verification, and non-training candidate evaluations.
- `DATASET_MANIFEST.json`: Machine-readable metadata, seeds, file hashes, and category distributions.
- `build_sft_v3_dataset.py`: Fully deterministic, reproducible dataset generator.
- `validate_sft_v3_dataset.py`: Independent automated preflight validator.
- `evaluate_sft_v3_validation.py`: Non-training candidate evaluation harness measuring Native Generation vs Schema-Constrained Decoding.
- `train.jsonl`: 200 verified grounded training records (SHA-256: `57affaa613eaf0e31be752bc680dddcd3c748e1cee3752942db93e8dea270804`).
- `validation.jsonl`: 40 verified grounded validation records (SHA-256: `4c4c77d5db4884e06ed4e06550bd4badba6a4793f37a1d5a38d0c9512cbc1cf8`).
- `VALIDATION_CAPABILITY_EVALUATION.json`: Evaluation results of candidate checkpoint across native and constrained decoding.

## Commands to Reproduce Audits and Tests

```powershell
# 1. Run the builder to regenerate dataset
.\.venv\Scripts\python.exe experiments/sft_v3_dataset_builder/build_sft_v3_dataset.py

# 2. Run the automated preflight validator (must exit 0 with 0 errors)
.\.venv\Scripts\python.exe experiments/sft_v3_dataset_builder/validate_sft_v3_dataset.py

# 3. Run the unit and regression test suite (7/7 tests passed)
.\.venv\Scripts\python.exe -m pytest tests/test_sft_v3_dataset_builder.py

# 4. Run the non-training validation capability evaluation
.\.venv\Scripts\python.exe experiments/sft_v3_dataset_builder/evaluate_sft_v3_validation.py
```

## Hard Safety Confirmation
- **ZERO** model training, fine-tuning, optimizer steps, or backward passes were executed.
- `chakrview/runtime/session.py` has **ZERO** modifications.
- Canonical checkpoints and tokenizers remain 100% bit-identical.
- Held-out test split (`test.jsonl`) was **NEVER** accessed or inspected.
