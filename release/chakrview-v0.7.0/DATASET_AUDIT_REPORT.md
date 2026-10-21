# ChakrView v0.7.0 Coding Dataset Audit Report

**Component**: `chakrview/data/coding_pipeline.py`  
**Pipeline Status**: Verified & Leakage-Safe  

---

## 1. Schema Specifications

The pipeline defines the canonical `CodingRecord` schema:
```json
{
  "schema_version": "1.0.0",
  "task_id": "CODE-001",
  "task_type": "code_continuation",
  "language": "python",
  "context": "",
  "prompt": "def add(a, b): return ",
  "response": "a + b",
  "provenance": "synthetic_verified",
  "split": "train",
  "template_family": "math_operations",
  "metadata": {}
}
```

---

## 2. Validation & Contamination Guards

The pipeline enforces:
1. **Mandatory Field Checking**: Rejects records missing `task_id`, `task_type`, `language`, `prompt`, `response`, `provenance`, or `split`.
2. **Duplicate Detection**: Distinguishes exact duplicates from normalized duplicates (stripping comments and whitespace).
3. **Target Leakage Auditing**: Flags any response target in `train` that appears in `validation` or `test` splits.
4. **Template Family Overlap**: Flags template families shared across splits to prevent memorization artifacts.
5. **Sequence Length Enforcement**: Validates that prompt + completion token lengths fit within the model's 512-token context window using `BpeTokenizer`.
6. **Deterministic Manifests**: Writes data atomically and hashes output with SHA-256.

---

## 3. Data Integrity & Trustworthiness Policy

- **No Fabricated Provenance**: The pipeline strictly rejects unverified or synthetic data lacking clear provenance.
- **Missing Data Requirement**: Currently, the repository possesses the evaluation suite and experimental SFT JSON datasets. Large-scale high-diversity coding datasets are planned for collection in v0.8.0 under documented licensing.
