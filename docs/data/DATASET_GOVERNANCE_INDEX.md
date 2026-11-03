# ChakrView Dataset Governance Index

**Repository:** `ChakrView`  
**Location:** `docs/data/DATASET_GOVERNANCE_INDEX.md`  
**Purpose:** Master governance registry for all active training, validation, and capability evaluation datasets.  
**Version:** 1.0.0  
**Effective Date:** 2026-10-30  
**Status:** Living Governance Document  

---

## 1. Overview and Operational Scope

This index establishes the central record of authority for all instruction-tuning, conversational, and capability datasets developed or utilized in ChakrView. It serves as the primary governance gateway connecting data artifacts to their respective generators, source dependencies, forensic licensing audits, and public release eligibility gates.

### Governance Status Taxonomy
Every dataset registered in this index is evaluated against four orthogonal governance dimensions:
- **Provenance Status:** Verification level of the complete chain of origin, author identities, input texts, and generation code (`VERIFIED`, `PARTIAL_CHAIN`, or `INCOMPLETE`).
- **Reproducibility Status:** Verification level of bit-exact or deterministic regeneration from source (`GENERATION_DETERMINISTIC`, `SPLIT_DETERMINISTIC`, `BIT_EXACT_VERIFIED`, or `REPRODUCIBILITY INCOMPLETE`).
- **Declared License:** The self-asserted or manifest-declared license key (`CC0-1.0`, `CC0-1.0 / MIT Compatible`, or `UNDECLARED`).
- **Publication Eligibility:** Strict legal and technical clearance status for public repository branches or external redistribution:
  - `VERIFIED — PUBLICATION ELIGIBLE` / `ELIGIBLE_FOR_RELEASE_REVIEW`: All technical, provenance, and legal gates passed with documented authorization.
  - `INTERNAL — NOT CLEARED FOR PUBLICATION` / `INTERNAL_ONLY`: Cleared for internal research, training, and benchmarking only; barred from public distribution.
  - `HOLD — PROVENANCE REQUIRED` / `HOLD_PROVENANCE`: Publication blocked pending formal source text attribution or edition clarification.
  - `HOLD — LICENSE EVIDENCE REQUIRED` / `HOLD_LICENSE`: Publication blocked pending resolution of missing, incomplete, or contradictory license declarations.
  - `REPRODUCIBILITY INCOMPLETE` / `HOLD_REPRODUCIBILITY`: Publication blocked due to absent generator scripts or inability to verify deterministic regeneration.

#### Machine-Readable Status Mapping
| Human-Readable Display Status | Audit Machine-Readable Status | Operational Meaning |
| :--- | :--- | :--- |
| `INTERNAL — NOT CLEARED FOR PUBLICATION` | `INTERNAL_ONLY` | Internal experimental use only; barred from public remotes |
| `HOLD — PROVENANCE REQUIRED` | `HOLD_PROVENANCE` | Publication blocked pending classical text attribution |
| `HOLD — LICENSE EVIDENCE REQUIRED` | `HOLD_LICENSE` | Publication blocked pending formal dedication deed or schema fix |
| `REPRODUCIBILITY INCOMPLETE` | `HOLD_REPRODUCIBILITY` | Publication blocked pending generator script recovery |
| `VERIFIED — PUBLICATION ELIGIBLE` | `ELIGIBLE_FOR_RELEASE_REVIEW` | All 6 gates passed; pending project owner release sign-off |

---

## 2. Master Dataset Register

The following table provides the comprehensive registry of evaluated datasets across Milestone M2 (Exp34), Milestone M3 (Conversational Intelligence), and Milestone M5 (Exp35 DPCE).

| Dataset ID | Canonical Relative Path | Records | Primary Generator / Script | Direct Source Dependencies | Declared License | Provenance Status | Reproducibility Status | Publication Eligibility | Related Experiment |
| :--- | :--- | :---: | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **DS-EXP34-V1** | `data/sft/exp34_instruction_mixture_train.jsonl` | 1,000 | `scripts/compile_exp34_dataset.py` | `scripts/generate_full_1000_dataset.py`<br>`scripts/generate_dataset_categories_2_to_5.py`<br>`scripts/generate_dataset_categories_6_to_10.py` | `CC0-1.0` *(in records)* | `PARTIAL_CHAIN` | `GENERATION_DETERMINISTIC` | `INTERNAL_ONLY` (`INTERNAL — NOT CLEARED`) | Exp34 (M2) |
| **DS-EXP34-V2** | `data/sft/exp34_candidate_v2/exp34_instruction_mixture_train_v2.jsonl` | 1,000 | `scripts/compile_exp34_candidate_v2.py` | `scripts/generate_candidate_v2_*.py`<br>`data/raw/english/*.txt`<br>`data/raw/hindi/*.txt`<br>`data/raw/hinglish/*.txt` | `CC0-1.0` *(in records)* | `PARTIAL_CHAIN` | `GENERATION_DETERMINISTIC` | `HOLD_PROVENANCE` (`HOLD — PROVENANCE REQUIRED`) | Exp34 Remediation (M2) |
| **DS-M3-FULL** | `data/sft/m3_conversational/m3_conversational_mixture_600.jsonl` | 600 | `scripts/compile_m3_dataset.py` | Hardcoded seed templates in script body | `CC0-1.0` *(in records)* | `VERIFIED` | `GENERATION_DETERMINISTIC` | `INTERNAL_ONLY` (`INTERNAL — NOT CLEARED`) | M3 Conversational |
| **DS-M3-TRAIN** | `data/sft/m3_conversational/splits/m3_train_split_540.jsonl` | 540 | `scripts/compile_m3_dataset.py` | `DS-M3-FULL` (90% stratified split) | `CC0-1.0` *(in records)* | `VERIFIED` | `SPLIT_DETERMINISTIC` | `INTERNAL_ONLY` (`INTERNAL — NOT CLEARED`) | M3 Conversational |
| **DS-M3-VAL** | `data/sft/m3_conversational/splits/m3_val_split_60.jsonl` | 60 | `scripts/compile_m3_dataset.py` | `DS-M3-FULL` (10% stratified split) | `CC0-1.0` *(in records)* | `VERIFIED` | `SPLIT_DETERMINISTIC` | `INTERNAL_ONLY` (`INTERNAL — NOT CLEARED`) | M3 Conversational |
| **DS-EXP35-FULL** | `experiments/exp35_dpce/data/exp35_full_dataset.jsonl` | 1,000 | *Unavailable on disk* | Procedural entity pools (fictional slots) | `UNDECLARED` *(in records)*<br>`CC0-1.0` *(in MD report)* | `INCOMPLETE` | `REPRODUCIBILITY INCOMPLETE` | `HOLD_LICENSE` / `HOLD_REPRODUCIBILITY` | Exp35 DPCE (M5) |
| **DS-EXP35-TRAIN** | `experiments/exp35_dpce/data/exp35_train.jsonl` | 900 | *Unavailable on disk* | `DS-EXP35-FULL` (90% stratified split) | `UNDECLARED` *(in records)*<br>`CC0-1.0` *(in MD report)* | `INCOMPLETE` | `REPRODUCIBILITY INCOMPLETE` | `HOLD_LICENSE` / `HOLD_REPRODUCIBILITY` | Exp35 DPCE (M5) |
| **DS-EXP35-VAL** | `experiments/exp35_dpce/data/exp35_validation.jsonl` | 100 | *Unavailable on disk* | `DS-EXP35-FULL` (10% stratified split) | `UNDECLARED` *(in records)*<br>`CC0-1.0` *(in MD report)* | `INCOMPLETE` | `REPRODUCIBILITY INCOMPLETE` | `HOLD_LICENSE` / `HOLD_REPRODUCIBILITY` | Exp35 DPCE (M5) |
| **DS-EXP35-EVAL** | `experiments/exp35_dpce/benchmark/exp35_benchmark_200.json` | 200 | *Unavailable on disk* | Procedural entity combinations | `UNDECLARED` *(in file)* | `INCOMPLETE` | `REPRODUCIBILITY INCOMPLETE` | `HOLD_LICENSE` / `HOLD_REPRODUCIBILITY` | Exp35 Evaluation (M5) |

---

## 3. Cross-Document Navigation

To review specific governance domains, navigate to the dedicated companion registers:
- **Detailed Lineage & Transformations:** [`DATASET_PROVENANCE_REGISTER.md`](DATASET_PROVENANCE_REGISTER.md)
- **Legal Grounds & Licensing Evidence:** [`DATASET_LICENSE_REGISTER.md`](DATASET_LICENSE_REGISTER.md)
- **Deterministic Recipes & Hashes:** [`DATASET_REPRODUCIBILITY.md`](DATASET_REPRODUCIBILITY.md)
- **Publication Release Gates & Audit Checklist:** [`DATASET_RELEASE_CHECKLIST.md`](DATASET_RELEASE_CHECKLIST.md)

