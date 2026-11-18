# UCIP-12: Source Licensing & Legal Provenance Audit

**Mission:** UCIP-12 — Large-Scale Corpus Foundation & Learning Readiness  
**Target File:** `experiments/ucip12/provenance/source_registry.json`  
**Standard:** Open-Source Commercial Permissiveness, Strict Attribution & Purity Invariant  

---

## 1. Statutory Licensing Framework

ChakrView requires absolute legal traceability and permissive licensing for all pretraining corpora. Datasets with non-commercial restrictions (e.g. `CC-BY-NC`), share-alike copyleft viral propagation into model weights without commercial exemption, or unverified web-scrapes (e.g., Common Crawl unfiltered, Books3) are strictly prohibited.

---

## 2. Source Registry Audit (13 Sources)

The repository's master source registry (`scripts/stage_c/sources.json`, mirrored to `experiments/ucip12/provenance/source_registry.json`) tracks 13 candidate sources.

### A. Approved Sources (9 Sources)

| Source ID | Name | License | Upstream Source / URL | Status | Pretraining Inclusion |
|---|---|---|---|---|---|
| `cpython_lib` | CPython Standard Library (3.12.5) | PSF-2.0 | `python/cpython` (GitHub) | **APPROVED** | Ingested (787,190 tokens) |
| `thealgorithms_python` | TheAlgorithms - Python Implementations | MIT | `TheAlgorithms/Python` (GitHub) | **APPROVED** | Ingested (798,832 tokens) |
| `simple_english_wikipedia` | Simple English Wikipedia Official Dump | CC-BY-SA-4.0 | Wikimedia Foundation Official Dumps | **APPROVED** | Ingested (2,801,696 tokens) |
| `hindi_wikipedia` | Hindi Wikipedia Official Dump | CC-BY-SA-4.0 | Wikimedia Foundation Official Dumps | **APPROVED** | Ingested (2,400,182 tokens) |
| `sanskrit_canonical_classics` | Sanskrit Classical & Wikipedia Texts | Public Domain / CC-BY-SA | Wikimedia Foundation / Canonical Classics | **APPROVED** | Ingested (407,107 tokens) |
| `gsm8k_reasoning` | GSM8k Grade School Math Reasoning | MIT | `openai/grade-school-math` | **APPROVED** | Ingested (500,122 tokens) |
| `openstax_math` | OpenStax College Algebra & Calculus Excerpts | CC-BY-4.0 | OpenStax CNX / Rice University | **APPROVED** | Ingested (4,411 tokens) |
| `gutenberg_hindi_premchand` | Project Gutenberg Classical Hindi Works | Public Domain | Project Gutenberg (Author died 1936) | **APPROVED** | Pinned for Stage C.2 ingestion |
| `ogd_india_data` | Open Government Data Platform India | GODL-India | `data.gov.in` | **APPROVED** | Ingested (2,422 tokens) |

### B. Rejected Sources (3 Sources)

| Source ID | Upstream License | Rejection Rationale |
|---|---|---|
| `iitb_parallel_hinglish` | CC-BY-NC-SA-4.0 | **Non-Commercial Restriction:** The `-NC` clause legally prohibits downstream commercial or unrestricted weights distribution. |
| `common_crawl_unfiltered` | Unspecified / Web Scraped | **Corpus Purity Invariant:** Unfiltered web scrapes contain toxic content, unvetted copyrighted text, and private information. |
| `books3_pile` | Unlicensed Copyright Infringement | **Direct Copyright Infringement:** Contains unlicensed proprietary publications posing severe legal liability. |

### C. Conditional Sources (1 Source)

| Source ID | License | Condition for Admission |
|---|---|---|
| `l3cube_mahahinglish` | CC-BY-SA-4.0 | Requires automated scrubbing of personal handles (`@`), email addresses, phone numbers, and offensive conversational content before shard incorporation. 10 curated documents (1,105 tokens) admitted. |

---

## 3. License Compatibility Matrix

```
Permissive Code (MIT / PSF-2.0):    1,586,022 tokens (20.59%)
Knowledge / Encyclopedia (CC-BY-SA): 5,608,985 tokens (72.82%)
Reasoning & Education (MIT/CC-BY):    504,533 tokens ( 6.55%)
Public Domain & Open Gov (GODL):        3,527 tokens ( 0.05%)
-------------------------------------------------------------
TOTAL VERIFIED & COMPLIANT:         7,703,067 tokens (100.0%)
```

## 4. Compliance Verification
- **Zero non-commercial data** exists in Stage C.
- **Zero copyrighted book dump data** exists in Stage C.
- All attribution requirements are preserved via `data/manifests/stage_c_manifest.json` and mirrored in `experiments/ucip12/manifests/`.
