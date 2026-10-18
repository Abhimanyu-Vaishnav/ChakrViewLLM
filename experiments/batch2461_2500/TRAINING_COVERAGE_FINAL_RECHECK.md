# TRAINING COVERAGE FINAL RECHECK REPORT

**Batch:** 2461–2500  
**Audit Phase:** Phase 4 — Independent Verification of CPT Training Coverage & Sanskrit Representation  
**Training Split Manifest:** [`experiments/batch2021_2060/SPLIT_V2_MANIFEST.json`](file:///d:/Project/ChakrView/experiments/batch2021_2060/SPLIT_V2_MANIFEST.json)  
**Cleaned Corpus Manifest:** [`experiments/batch2101_2140/CORPUS_V2_CLEANED_MANIFEST.json`](file:///d:/Project/ChakrView/experiments/batch2101_2140/CORPUS_V2_CLEANED_MANIFEST.json)  
**Canonical Tokenizer:** [`release/chakrview-v0.5.0/data/experiments/vocab_4096`](file:///d:/Project/ChakrView/release/chakrview-v0.5.0/data/experiments/vocab_4096)  
**Machine-Readable Summary:** [`experiments/batch2461_2500/TRAINING_COVERAGE_FINAL_RECHECK.json`](file:///d:/Project/ChakrView/experiments/batch2461_2500/TRAINING_COVERAGE_FINAL_RECHECK.json)  
**Status:** **INDEPENDENTLY VERIFIED — ZERO CLASSICAL SANSKRIT TRAINING EXPOSURE**  

---

## 1. Executive Summary & Verification Protocol

Phase 4 mandates independent verification of actual CPT training inputs to confirm or refute the claim of zero Classical Sanskrit training coverage. We avoided superficial source-name checks and executed a three-layer audit:
1. **Source Label Layer:** Inspecting declared source IDs.
2. **Language Metadata Layer:** Checking ISO 639 language codes assigned in the corpus manifest.
3. **Text and Token Layer:** Tokenizing all 782 raw training documents using the canonical BPE tokenizer and calculating exact target tokens using the standard sequence-chunking convention ($max\_seq = 256$, targets = $\text{len}(chunk) - 1$).

---

## 2. Input Resolution and Linkage Integrity

- **Train Document IDs in Split Manifest:** 782
- **Unique Train Document IDs:** 782 (0 duplicates)
- **Resolved Against Cleaned Corpus:** 782 / 782 (100% resolution rate, 0 missing IDs)
- **Total Training Documents Analyzed:** 782

---

## 3. Layered Coverage Analysis

### A. Source-Level Breakdown
| Source ID | Declared Language | Documents | Raw Encoded Tokens | Target Tokens ($max\_seq=256$) |
| :--- | :---: | :---: | :---: | :---: |
| `ietf_rfc793_tcp` | `en` | 367 | 61,905 | 61,538 |
| `gutenberg_tagore_gitanjali` | `en` | 195 | 34,007 | 33,812 |
| `wikipedia_en_turing_machine` | `en` | 74 | 16,678 | 16,604 |
| `wikipedia_hi_chemistry` | `hi` | 58 | 14,885 | 14,827 |
| `cpython_algorithms_heapq` | `code` | 56 | 9,814 | 9,758 |
| `wikipedia_hi_premchand` | `hi` | 32 | 10,187 | 10,155 |
| **Total** | **3 distinct** | **782** | **147,476** | **146,694** |

### B. Language Metadata Distribution
- `en` (English): 636 documents (81.33% of train docs)
- `hi` (Hindi): 90 documents (11.51% of train docs)
- `code` (Python Code): 56 documents (7.16% of train docs)
- `sa` (Classical Sanskrit): **0 documents (0.0%)**

### C. Corpus-Wide Sanskrit Document Allocation
In [`CORPUS_V2_CLEANED_MANIFEST.json`](file:///d:/Project/ChakrView/experiments/batch2101_2140/CORPUS_V2_CLEANED_MANIFEST.json), exactly 70 documents belong to Sanskrit-related sources:
- `wikipedia_sa_bhagavad_gita` (61 documents, language: `sa`, Classical Sanskrit): **100% assigned to `test` split**.
- `wikipedia_hi_sanskrit_sahitya` (9 documents, language: `hi`, Hindi commentary): **100% assigned to `test` split**.
- **`train` split Sanskrit count:** Exactly 0 documents, exactly 0 target tokens.
- **`validation` split Sanskrit count:** Exactly 0 documents.

---

## 4. Audit Conclusion & Boundary Conditions
1. **Fact:** The CPT training set contained zero Classical Sanskrit documents and zero Classical Sanskrit target tokens.
2. **Fact:** The test set contained 19,577 target tokens of Classical Sanskrit (`wikipedia_sa_bhagavad_gita`) on which the model regressed by +0.0776 CE (+8.07% PPL).
3. **Causality Boundary:** This observation supports the hypothesis that the regression is an unreinforced domain drift. However, establishing that adding Sanskrit training material will repair this drift without harming English, Hindi, or Code performance requires a controlled curriculum intervention experiment.
