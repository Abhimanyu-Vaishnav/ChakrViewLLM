# ChakrView v0.9.0 — Dataset Provenance and Governance Status

## 1. Governance Classification Framework

In accordance with `docs/data/DATASET_GOVERNANCE_INDEX.md`, datasets are audited under four operational classifications:
1. `INTERNAL_ONLY`: Valid for local experimentation; barred from public distribution.
2. `HOLD_PROVENANCE`: Barred from release pending source attribution closure.
3. `HOLD_LICENSE`: Barred from release pending copyright dedication deed.
4. `ELIGIBLE_FOR_RELEASE_REVIEW`: All six governance gates passed.

---

## 2. Dataset Status Audit

| Dataset ID | Canonical Relative Path | Records | Operational Status | Release Clearance |
| :--- | :--- | :---: | :---: | :---: |
| **DS-EXP34-V1** | `data/sft/exp34_instruction_mixture_train.jsonl` | 1,000 | `INTERNAL_ONLY` | Restricted to research repository |
| **DS-EXP34-V2** | `data/sft/exp34_candidate_v2/exp34_instruction_mixture_train_v2.jsonl` | 1,000 | `HOLD_PROVENANCE` | Blocked from standalone distribution |
| **DS-M3-FULL** | `data/sft/m3_conversational/m3_conversational_mixture_600.jsonl` | 600 | `INTERNAL_ONLY` | Restricted to research repository |
| **DS-M3-TRAIN** | `data/sft/m3_conversational/splits/m3_train_split_540.jsonl` | 540 | `INTERNAL_ONLY` | Restricted to research repository |
| **DS-M3-VAL** | `data/sft/m3_conversational/splits/m3_val_split_60.jsonl` | 60 | `INTERNAL_ONLY` | Restricted to research repository |
| **DS-EXP35-FULL**| `experiments/exp35_dpce/data/exp35_full_dataset.jsonl` | 1,000 | `HOLD_LICENSE` | Blocked from standalone distribution |
| **Stage C Train**| `data/tokenized/stage_c/train/*.bin` | 6.65M tok | `VERIFIED_INTERNAL` | Permitted for model pretraining |

---

## 3. Policy Adherence
- No raw or unlicensed datasets are packaged for external public release.
- Only reproducible model weights, configuration, and tokenizer metadata are included in release deliverables.
