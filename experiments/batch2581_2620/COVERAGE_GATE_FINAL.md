# TRAINING COVERAGE FINAL AUDIT & GATE ENFORCEMENT REPORT
**Batch:** 2581–2620  
**Repository:** ChakrViewLLM (`dev/v0.6-v0.7-validation`)  
**Primary Manifests:** `experiments/batch2021_2060/SPLIT_V2_MANIFEST.json` and `experiments/batch2101_2140/CORPUS_V2_CLEANED_MANIFEST.json`  

---

### 1. Invariant Enforcement Architecture

In Batch 2581–2620, the training-coverage evaluation gate is explicitly tied to 10 non-negotiable invariants. Every invariant is evaluated dynamically from ground-truth manifests, and any single failure causes the gate status to evaluate to `FAILED` with specific failing element identifiers.

| Invariant ID | Description | Enforced Requirement | Status |
| :--- | :--- | :--- | :--- |
| **INV-01** | Duplicate Training IDs | $\text{Count}(\text{duplicates in train}) = 0$ | **PASSED** (0 dups) |
| **INV-02** | Missing Training IDs | $\text{train\_ids} \subseteq \text{corpus\_ids}$ | **PASSED** (0 missing) |
| **INV-03** | Unexpected Document IDs | No foreign IDs outside corpus namespace | **PASSED** (0 unexpected) |
| **INV-04** | Duplicate Corpus IDs | $\text{Count}(\text{duplicates in corpus}) = 0$ | **PASSED** (0 dups) |
| **INV-05** | Split Disjointness | $\text{train} \cap \text{val} = \emptyset$, $\text{train} \cap \text{test} = \emptyset$, $\text{val} \cap \text{test} = \emptyset$ | **PASSED** (All 0) |
| **INV-06** | Partition Exhaustiveness | $\text{train} \cup \text{val} \cup \text{test} = \text{corpus\_ids}$ ($1258 = 1258$) | **PASSED** (Exact match) |
| **INV-07** | Source/Language Fidelity | Corpus `source_id` and `language` match split assignments | **PASSED** (0 mismatches) |
| **INV-08** | Classical Sanskrit Absence | $\text{Classical Sanskrit docs in train} = 0$, $\text{tokens} = 0$ | **PASSED** (0 docs, 0 toks) |
| **INV-09** | Per-Doc / Aggregate Total Consistency | $\sum \text{doc\_tokens} = \text{aggregate\_tokens}$ | **PASSED** (Exact match) |
| **INV-10** | Manifest Semantics Consistency | Declared `document_count` equals length of `document_ids` | **PASSED** (All 3 match) |

---

### 2. Empirical Verification Results

1. **Partition Composition:**
   - Train Split: 782 documents ($62.16\%$)
   - Validation Split: 186 documents ($14.79\%$)
   - Test Split: 290 documents ($23.05\%$)
   - Total Documents: 1,258 documents
2. **Sanskrit Absence in Training:**
   - 100% of the 70 corpus Sanskrit documents (`wikipedia_sa_bhagavad_gita`: 61, `wikipedia_hi_sanskrit_sahitya`: 9) are assigned to the `test` split. Exactly 0 are in `train`, and 0 in `validation`.
3. **Derived Gate Verdict:**
   - Total Invariants Evaluated: 10
   - Invariants Passed: 10
   - Invariants Failed: 0
   - **Gate Status:** **`VERIFIED`**
