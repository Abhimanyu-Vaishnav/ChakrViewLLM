# SOURCE IDENTITY RECONCILIATION

**Batch:** 2341–2380  
**Focus:** Reconstruction and Reconciliation of Source-Level Document and Token Counts  
**Status:** FULLY RECONCILED  

---

## 1. Discrepancy Overview

The audit prompt identified a conflict between two historical source-allocation tables:

- **Earlier Table (Draft Pre-Cleansing Allocation):**
  - `gutenberg_loney`: 134 documents, 26,052 tokens
  - `wikipedia_en_information_theory`: 51 documents, 10,750 tokens
  - `wikipedia_hi_sanskrit_sahitya`: 44 documents, 8,390 tokens
  - `wikipedia_sa_bhagavad_gita`: 61 documents, 19,577 tokens
  - Total: 290 documents, 64,769 tokens

- **Latest Table (Actual Ground-Truth Evaluator Allocation):**
  - `gutenberg_loney_coordinate_geometry`: 150 documents, 24,482 tokens
  - `wikipedia_en_information_theory`: 70 documents, 15,912 tokens
  - `wikipedia_hi_sanskrit_sahitya`: 9 documents, 4,798 tokens
  - `wikipedia_sa_bhagavad_gita`: 61 documents, 19,577 tokens
  - Total: 290 documents, 64,769 tokens

Both tables total 290 documents and 64,769 tokens, but differ significantly in how the non-Sanskrit documents are partitioned.

---

## 2. Primary Source-of-Truth Investigation

We inspected the primary data manifest used by the actual evaluator:
`experiments/batch2101_2140/CORPUS_V2_CLEANED_MANIFEST.json` (SHA-256: `e3db033c83f356f653b10d7eb00994da31d6e0738ab4177faddb133e5b8da380`).

Querying all documents where `split == "test"` directly yielded:
- Total test documents in manifest: **290**
- Source breakdown:
  1. `gutenberg_loney_coordinate_geometry`: **150 documents** (24,482 evaluated tokens)
  2. `wikipedia_en_information_theory`: **70 documents** (15,912 evaluated tokens)
  3. `wikipedia_hi_sanskrit_sahitya`: **9 documents** (4,798 evaluated tokens)
  4. `wikipedia_sa_bhagavad_gita`: **61 documents** (19,577 evaluated tokens)
- Total evaluated tokens under canonical tokenizer: **64,769 tokens**.

### Cross-Verification with Preceding Corpus Manifest
Checking the precursor manifest `experiments/batch2021_2060/CORPUS_V2_MANIFEST.json` and `experiments/batch2021_2060/SPLIT_V2_MANIFEST.json` confirms that the test document ID set (`NATURAL_V2_TEST_00001` through `NATURAL_V2_TEST_00290`) has **always** mapped to 150 Loney, 70 Information Theory, 9 Sanskrit Sahitya, and 61 Bhagavad Gita documents.

---

## 3. Origin of the Earlier Conflicting Table

The earlier table (134 Loney, 51 Info Theory, 44 Sanskrit Sahitya, 61 Gita) originated from an **unverified draft planning table** created during exploratory analysis before the formal data cleansing in Batch 2101–2140. In that preliminary draft, certain multi-passage documents from Hindi literature and English texts were counted using preliminary chunk boundaries rather than the frozen document ID registry.

### Conclusion
The true, active, and verified source-family partition across the 290 test documents is **150 / 70 / 9 / 61**, totaling 64,769 tokens. Every document appears exactly once, with zero overlap and zero orphan documents.
