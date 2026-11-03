# ChakrView Dataset Licensing and Attribution Register

**Repository:** `ChakrView`  
**Location:** `docs/data/DATASET_LICENSE_REGISTER.md`  
**Purpose:** Legal governance register separating code licenses, data rights, upstream sources, and attribution obligations.  
**Version:** 1.0.0  
**Effective Date:** 2026-10-30  
**Status:** Living Governance Document  

---

## 1. Principles of Dataset Licensing Governance

1. **Separation of Code and Data:** The license governing an executable compiler script (e.g., MIT or Apache-2.0) does not automatically dictate the copyright status of generated data. Data outputs are evaluated under copyright principles applicable to compilations, databases, and synthetic text.
2. **Legal Basis of Public Domain Dedication:** Stating `"license": "CC0-1.0"` in a JSON dictionary is a technical metadata declaration, not a self-executing legal deed. A valid CC0-1.0 dedication requires that the copyright holder (or legally authorized project entity) holds copyright title and formally executes the dedication.
3. **Factual and Non-Copyrightable Material:** Pure mathematical facts, common scientific constants (e.g. speed of light, planetary order), and simple code snippets lacking sufficient creative originality may fall outside copyright protection. However, specific creative expressions, curated instruction orderings, and prose explanations are subject to copyright.
4. **No Presumption of Clearance:** No dataset will be certified as publication-ready or distributed under open-source licenses until upstream copyright boundaries are documented with verifiable evidence.

---

## 2. Licensing Audit Matrix

| Dataset ID | Declared Record License | Generator Code License | Upstream Source Ingestions | Upstream License Basis | Attribution Obligations | Contributor Dedication Authority | Clearance Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **DS-EXP34-V1** | `CC0-1.0` *(asserted)* | Project repository standard | Hardcoded prompts & algorithms | Indigenous author creation | None identified | Needs formal contributor confirmation | `INTERNAL — NOT CLEARED FOR PUBLICATION` |
| **DS-EXP34-V2** | `CC0-1.0` *(asserted)* | Project repository standard | `data/raw/english/*.txt`<br>`data/raw/hindi/*.txt`<br>`data/raw/hinglish/*.txt` | `CC0-1.0 / MIT Compatible` *(asserted in raw manifest)* | Specific edition/translation citations required | Incomplete chain of title for classical texts | `HOLD — PROVENANCE REQUIRED` |
| **DS-M3-FULL** | `CC0-1.0` *(asserted)* | Project repository standard | Hardcoded Q&A seeds | Indigenous author creation | None identified | Needs formal contributor confirmation | `INTERNAL — NOT CLEARED FOR PUBLICATION` |
| **DS-M3-TRAIN** | `CC0-1.0` *(asserted)* | Project repository standard | Derived from `DS-M3-FULL` | Indigenous author creation | None identified | Needs formal contributor confirmation | `INTERNAL — NOT CLEARED FOR PUBLICATION` |
| **DS-M3-VAL** | `CC0-1.0` *(asserted)* | Project repository standard | Derived from `DS-M3-FULL` | Indigenous author creation | None identified | Needs formal contributor confirmation | `INTERNAL — NOT CLEARED FOR PUBLICATION` |
| **DS-EXP35-FULL** | `UNDECLARED` *(in records)*<br>`CC0-1.0` *(in report)* | *Unavailable* | Fictional entity pools | Indigenous synthetic generation | None identified | Absent from schema; report ungrounded | `HOLD — LICENSE EVIDENCE REQUIRED` |
| **DS-EXP35-TRAIN** | `UNDECLARED` *(in records)*<br>`CC0-1.0` *(in report)* | *Unavailable* | Derived from `DS-EXP35-FULL` | Indigenous synthetic generation | None identified | Absent from schema; report ungrounded | `HOLD — LICENSE EVIDENCE REQUIRED` |
| **DS-EXP35-VAL** | `UNDECLARED` *(in records)*<br>`CC0-1.0` *(in report)* | *Unavailable* | Derived from `DS-EXP35-FULL` | Indigenous synthetic generation | None identified | Absent from schema; report ungrounded | `HOLD — LICENSE EVIDENCE REQUIRED` |
| **DS-EXP35-EVAL** | `UNDECLARED` *(in file)* | *Unavailable* | Fictional entity combinations | Indigenous synthetic generation | None identified | Unspecified | `HOLD — LICENSE EVIDENCE REQUIRED` |

---

## 3. Detailed Attribution & Legal Boundary Analysis

### 3.1 Upstream Raw Corpus Attribution (`data/raw/*`)
The project's Stage B manifest (`data/raw/manifest.json`) categorizes 27 files totaling 1.57 MB across code, English, Hindi, Hinglish, mathematics, and Sanskrit.
- **The Stated Claim:** Files are declared as `"Project-authored, indigenous synthesis & public domain classical texts"`.
- **The Legal Risk:** While ancient classical texts (e.g. Vedic hymns, classical Sanskrit literature) are in the public domain, modern English or Hindi translations, annotations, and critical editions published within copyright terms (typically life of the author plus 60 to 70 years) remain protected by copyright.
- **Action Required:** Before datasets derived from `data/raw/*` (specifically `DS-EXP34-V2` Categories 6, 9, 10) can be published, a formal attribution register must record the publication year and translator for all ingested classical works.

### 3.2 Contributor CC0 Dedication Authority
Datasets `DS-EXP34-V1`, `DS-EXP34-V2`, and `DS-M3-*` embed `"license": "CC0-1.0"` directly in their JSON lines.
- **Current Evidence:** The dedication string was introduced programmatically by the script author.
- **Governance Standard:** In accordance with Creative Commons protocols, dedication under CC0-1.0 requires an unambiguous statement of copyright waiver by the copyright owner.
- **Remediation:** Project contributors will execute a formal project-level Data Rights Deed dedicating all original training prompts and responses to CC0-1.0 prior to public release.

### 3.3 Exp35 License Schema Discrepancy
- **Current Evidence:** The JSONL files for Exp35 (`exp35_full_dataset.jsonl`, `exp35_train.jsonl`, `exp35_validation.jsonl`) do not contain a `"license"` field in their schema dictionaries. Conversely, `experiments/exp35_dpce/data/exp35_dataset_quality_report.md` states:
  > *"License Designation: CC0-1.0 (Public Domain Dedication) for project-generated synthetic text."*
- **Governance Standard:** Technical schema documentation and governance quality reports must not contradict one another.
- **Remediation:** Harmonize metadata schemas upon future script recovery.

