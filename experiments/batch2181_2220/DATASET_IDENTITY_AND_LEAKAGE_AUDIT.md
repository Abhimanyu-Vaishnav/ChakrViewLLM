# DATASET IDENTITY AND LEAKAGE AUDIT

**Batch:** 2181–2220  
**Target:** Cleaned Multilingual Corpus v2 (`CORPUS_V2_CLEANED_MANIFEST.json`, SHA-256: `e3db033c...`)  
**Status:** `AUDITED — STRICT SEPARATION VERIFIED`  

---

## 1. Split Allocation and Document Counts

The cleaned manifest comprises **1,258 documents** partitioned into three disjoint splits:

| Split | Document Count | Token Count | Character Count | Percentage |
| :--- | :---: | :---: | :---: | :---: |
| **Train** | 782 | 269,726 | 1,460,248 | 62.16% |
| **Validation** | 186 | 60,828 | 328,154 | 14.79% |
| **Test** | 290 | 64,769 | 350,912 | 23.05% |
| **Total** | **1,258** | **395,323** | **2,139,314** | **100.0%** |

---

## 2. Leakage and Overlap Verification

An exhaustive set-intersection audit was executed across document IDs and SHA-256 content hashes:

1. **Document ID Overlap:**
   - Train $\cap$ Validation: **0 documents**
   - Train $\cap$ Test: **0 documents**
   - Validation $\cap$ Test: **0 documents**
2. **Content Hash (SHA-256) Overlap:**
   - Train $\cap$ Validation: **0 hashes**
   - Train $\cap$ Test: **0 hashes**
   - Validation $\cap$ Test: **0 hashes**
3. **Exact and Near-Duplicate Matching:**
   - No cross-split duplicate text segments detected.

---

## 3. Source-Family Partitioning Analysis

Unlike naive random splitting that scatters documents from the same book or source across splits, Corpus v2 enforces source-family boundaries:

| Source Identifier | Domain / Description | Assigned Split |
| :--- | :--- | :---: |
| `gutenberg_tagore_gitanjali` | English literature (Rabindranath Tagore) | **Train** |
| `cpython_algorithms_heapq` | Python source code (algorithms) | **Train** |
| `ietf_rfc793_tcp` | Networking specifications | **Train** |
| `wikipedia_hi_premchand` | Hindi literature | **Train** |
| `wikipedia_hi_chemistry` | Hindi scientific text | **Train** |
| `wikipedia_en_turing_machine` | Theoretical computer science | **Train** |
| `cpython_algorithms_bisect` | Python source code (bisect algorithm) | **Validation** |
| `wikipedia_hi_ganit` | Hindi mathematics & reasoning | **Validation** |
| `google_hinglish_top_dialogues` | Conversational Hindi-English dialogues | **Validation** |
| `gutenberg_loney_coordinate_geometry`| Classical geometry (S.L. Loney) | **Test** |
| `wikipedia_en_information_theory` | Mathematics & Information Theory | **Test** |
| `wikipedia_hi_sanskrit_sahitya` | Hindi essays on classical literature | **Test** |
| `wikipedia_sa_bhagavad_gita` | Classical Sanskrit philosophy | **Test** |

### Evaluation Integrity Implications
- The held-out test split evaluates **zero-shot domain generalization** to unseen technical and classical texts (`loney_coordinate_geometry` and `information_theory`).
- Training on the training partition never exposes the model to test documents or test source families.
- **Historical Pretraining Contamination:** The canonical step-10,000 checkpoint was originally pretrained on a broader corpus in Batches 16–21. While Corpus v2 test documents were isolated during CPT, general pretraining overlap from public domain sources (e.g. Gutenberg) cannot be completely ruled out for the canonical baseline. However, because both Baseline and Candidate models are evaluated on the exact same held-out benchmark, candidate improvements reflect genuine learning rather than baseline contamination.
