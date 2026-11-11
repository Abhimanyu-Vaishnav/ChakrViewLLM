# Dataset Integrity Report — CR-EXP-06

**Date**: 2026-11-06  
**Dataset Name**: `CR-EXP-06-Relational-Curriculum-v1`  
**Deterministic Generation Seed**: `20261106`  
**Manifest File**: `experiments/cognition/cr_exp_06_variable_binding/dataset_manifest.json`  

---

## 1. Split Counts and Cryptographic Checksums

| Split | File Name | Sample Count | SHA-256 Digest |
|---|---|---|---|
| **Training** | `train.jsonl` | 1,700 | Calculated in manifest |
| **Validation** | `validation.jsonl` | 200 | Calculated in manifest |
| **Test In-Distribution** | `test_in_distribution.jsonl` | 100 | Calculated in manifest |
| **Test OOD Entities** | `test_ood_entities.jsonl` | 100 | Calculated in manifest |
| **Test OOD Templates** | `test_ood_templates.jsonl` | 100 | Calculated in manifest |
| **Test Multi-Hop** | `test_multihop.jsonl` | 60 | Calculated in manifest |
| **TOTAL** | — | **2,260** | — |

---

## 2. Entity Isolation & Zero Contamination Guarantee

- **Training/Validation Entities**:
  - Persons (15): `Ravi`, `Priya`, `Amit`, `Kavita`, `Sunil`, `Neha`, `Vikram`, `Deepa`, `Sanjay`, `Ananya`, `Rohan`, `Pooja`, `Rajesh`, `Meera`, `Kabir`.
  - Cities (12): `Pune`, `Bengaluru`, `Mumbai`, `Delhi`, `Hyderabad`, `Chennai`, `Jaipur`, `Kolkata`, `Ahmedabad`, `Chandigarh`, `Nagpur`, `Indore`.
- **Strict OOD Evaluation Entities**:
  - Persons (12): `Aarav`, `Ishaan`, `Diya`, `Sanya`, `Tanya`, `Aditya`, `Rhea`, `Manish`, `Varun`, `Alok`, `Gaurav`, `Nisha`.
  - Cities (10): `Bhopal`, `Surat`, `Patna`, `Varanasi`, `Dehradun`, `Ranchi`, `Guwahati`, `Mysuru`, `Kochi`, `Coimbatore`.
  - Synthetic Objects (9): `Quasar`, `Rhodium`, `Platinum`, `Osmium`, `Titanium`, `Cobalt`, `Xylok`, `Zentrix`, `Pebble`.
- **Contamination Metrics**:
  - Overlap between ID persons and OOD persons: **0 (Exact 0%)**
  - Overlap between ID cities and OOD cities: **0 (Exact 0%)**
  - Prompt text duplicate count between Train and any Test split: **0 (Exact 0%)**

---

## 3. Span Annotation Verification
All relational samples contain `target_spans: [[start, end], ...]` indexing the exact character boundaries of the relational target entity in the `response` string. Non-relational conversational preservation samples have `target_spans: []`.

