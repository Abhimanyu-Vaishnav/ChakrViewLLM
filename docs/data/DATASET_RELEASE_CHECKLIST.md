# ChakrView Dataset Release Checklist

**Repository:** `ChakrView`  
**Location:** `docs/data/DATASET_RELEASE_CHECKLIST.md`  
**Purpose:** Mandatory gating protocol required before staging or publishing datasets to public repositories.  
**Version:** 1.0.0  
**Effective Date:** 2026-10-30  
**Status:** Living Governance Document  

---

## 1. Release Gating Protocol

No dataset, fine-tuning split, or evaluation benchmark may be transitioned from internal research status to public Git tracking or external distribution without satisfying **every gate** on this checklist. 

All reviews must be signed off by a project maintainer with technical and legal governance authority.

---

## 2. Mandatory Release Gates

### Gate 1: Provenance and Authorship Verification
- [ ] **1.1 Origin Established:** Authorship of all code, templates, and text is definitively documented.
- [ ] **1.2 Ingested Sources Identified:** Every upstream file (e.g. from `data/raw/*`) has an identified title, author, and date.
- [ ] **1.3 Public Domain Verification:** For classical or historical texts, translators and critical editions have been checked to confirm that copyright protection has expired.

### Gate 2: License and Legal Clearance
- [ ] **2.1 Documented License Basis:** A legally recognized license or dedication deed (e.g. CC0-1.0, MIT) is formally backed by author affirmations.
- [ ] **2.2 Separation of Code and Data:** The dataset's license is clearly distinguished from the generator script's source license.
- [ ] **2.3 Attribution Statements Drafted:** Any attribution notices required by upstream third-party sources are prepared and placed in documentation.

### Gate 3: Reproducibility and Code Lineage
- [ ] **3.1 Executable Compiler Available:** The compiler script exists in the repository and runs cleanly without manual interventions.
- [ ] **3.2 Deterministic Random Seed:** A frozen seed is used, and consecutive runs produce bit-exact matches.
- [ ] **3.3 Cryptographic Hashes Documented:** Full SHA-256 hashes are verified and recorded in [`DATASET_PROVENANCE_REGISTER.md`](DATASET_PROVENANCE_REGISTER.md).
- [ ] **3.4 Automated Regression Tests:** Pytest suites verify record count, schema fields, and non-empty completions.

### Gate 4: Technical Integrity and Contamination Audit
- [ ] **4.1 Schema Uniformity:** Every JSONL record contains required fields (`id`, `prompt`, `response`, `category`, `source_type`, `license`).
- [ ] **4.2 Duplicate Record Check:** Zero exact duplicate records exist within the dataset.
- [ ] **4.3 Response Diversity Audited:** The ratio of unique completions is calculated and declared.
- [ ] **4.4 Cross-Split Leakage Check:** Zero exact prompt or prompt-response pair overlaps between train, validation, and evaluation benchmark splits.
- [ ] **4.5 Template Contamination Review:** OOD evaluation benchmarks do not share trivial template patterns with training splits.

### Gate 5: Safety and Content Hygiene
- [ ] **5.1 PII & Private Data:** No personal identifiable information (real names, addresses, API keys, credentials) is embedded.
- [ ] **5.2 Harmful or Unsafe Text:** Content has been screened to ensure compliance with project safety invariants.
- [ ] **5.3 Binary Weight Exclusion:** Checkpoint weights (`*.pt`) are strictly excluded from the release commit.

### Gate 6: Project Owner Authorization
- [ ] **6.1 Formal Sign-Off:** Written confirmation from the project owner approving the public transition.
- [ ] **6.2 Staging Review:** Changes are staged using explicit, single-file paths (`git add <file>`), completely avoiding `git add .` or wildcard operators.

---

## 3. Dataset Gate Status Summary (As of 2026-10-30)

| Dataset Identifier | Gate 1 (Provenance) | Gate 2 (Licensing) | Gate 3 (Reproducibility) | Gate 4 (Integrity) | Gate 5 (Safety) | Gate 6 (Sign-Off) | Current Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **DS-EXP34-V1** | Passed | Partial | Passed | Passed (91.7% div) | Passed | Pending | `INTERNAL — NOT CLEARED` |
| **DS-EXP34-V2** | Blocked | Blocked | Passed | Passed (100% div) | Passed | Pending | `HOLD — PROVENANCE REQUIRED` |
| **DS-M3-FULL** | Passed | Partial | Passed | Passed (20.8% div) | Passed | Pending | `INTERNAL — NOT CLEARED` |
| **DS-M3-TRAIN** | Passed | Partial | Passed | Passed (22.4% div) | Passed | Pending | `INTERNAL — NOT CLEARED` |
| **DS-M3-VAL** | Passed | Partial | Passed | Passed (83.3% div) | Passed | Pending | `INTERNAL — NOT CLEARED` |
| **DS-EXP35-FULL** | Blocked | Blocked | Blocked | Passed (100% div) | Passed | Pending | `HOLD — LICENSE EVIDENCE REQUIRED` |
| **DS-EXP35-TRAIN** | Blocked | Blocked | Blocked | Passed (100% div) | Passed | Pending | `HOLD — LICENSE EVIDENCE REQUIRED` |
| **DS-EXP35-VAL** | Blocked | Blocked | Blocked | Passed (100% div) | Passed | Pending | `HOLD — LICENSE EVIDENCE REQUIRED` |
| **DS-EXP35-EVAL** | Blocked | Blocked | Blocked | Passed (85.0% div) | Passed | Pending | `HOLD — LICENSE EVIDENCE REQUIRED` |

