# CHAKRVIEW — Batches 1981–2020
## Corpus Quality Audit & Split Concentration Analysis

### Executive Summary

A comprehensive document-by-document audit was conducted on the candidate natural text corpus staged in Batches 1941–1980 ([CORPUS_MANIFEST.json](file:///d:/Project/ChakrView/experiments/batch1941_1980/data/CORPUS_MANIFEST.json) and [SPLIT_MANIFEST.json](file:///d:/Project/ChakrView/experiments/batch1941_1980/data/SPLIT_MANIFEST.json)).

While the corpus successfully replaced synthetic template generators with genuine historical literature and specifications, the audit revealed **extreme source concentration in the test split**, along with linguistic and script labeling inaccuracies that must be addressed before this dataset can serve as an uncompromised evaluation standard.

---

### 1. Corpus Distribution Audit

The audited dataset totals:
- **Total Documents:** 2,619 (Unique content hashes: 2,619; Exact duplicates: 0)
- **Total Tokens:** 455,953
- **Tiny Documents (<100 characters):** 0
- **Empty / Malformed Documents:** 0

#### Distribution Across Splits:

| Split Name | Document Count | Token Count | Percentage of Tokens | Assigned Independent Sources |
| :--- | :--- | :--- | :--- | :--- |
| **Train** | 650 | 115,913 | 25.4% | `gutenberg_tagore_gitanjali`, `ietf_rfc793_tcp`, `cpython_algorithms_heapq`, `wikipedia_hi_premchand` (4 sources) |
| **Validation** | 26 | 5,948 | 1.3% | `cpython_algorithms_bisect`, `wikipedia_hi_ganit` (2 sources) |
| **Test** | 1,943 | 334,092 | 73.3% | `gutenberg_loney_coordinate_geometry`, `wikipedia_hi_sanskrit_sahitya` (2 sources) |

---

### 2. Critical Quality Findings & Concentration Risk

#### A. Extreme Test Set Source Monopolization
- The test split contains 1,943 documents and 334,092 tokens, but **99.5% of the test documents (1,934 / 1,943) and 98.6% of the test tokens (329,285 / 334,092) originate from a single book**: *The Elements of Coordinate Geometry* by S. L. Loney (`gutenberg_loney_coordinate_geometry`).
- The remaining source (`wikipedia_hi_sanskrit_sahitya`) contributes only 9 documents (1.4% of tokens).
- **Audit Verdict:** The current test split does not evaluate broad natural language capability; it evaluates next-token prediction on 19th-century coordinate geometry textbooks. A source-disjoint split is **not** automatically a representative or balanced evaluation split.

#### B. Validation Split Scarcity
- The validation split contains only **26 documents and 5,948 tokens** across two sources (`bisect.py` and `wikipedia_hi_ganit`).
- This small sample volume provides insufficient statistical power for reliable checkpoint selection or hyperparameter tuning.

#### C. Language and Script Mislabeling
- `wikipedia_hi_sanskrit_sahitya` is labeled under domain `sanskrit` and language `hi-sa`.
- Inspection of the actual text reveals that the article is written in **standard modern Hindi about Sanskrit literature**, not Classical Sanskrit text itself:
  - Text excerpt: `"ऋग्वेदकाल से लेकर आज तक संस्कृत भाषा के माध्यम से सभी प्रकार के वाङ्मय का निर्माण होता आ रहा है..."`
- Labeling Hindi prose as Sanskrit distorts language fertility benchmarks and falsely inflates Sanskrit coverage claims.

---

### 3. Recommendations for Corpus Refinement

1. **Rebalance Test Split:** Sub-sample Loney's geometry text to approximately 200 documents (~35,000 tokens) and incorporate genuine diverse literature across all domains into the test set.
2. **Correct Sanskrit Labeling:** Classify Hindi articles on Sanskrit literature under `hindi_humanities`, reserving the `sanskrit` label strictly for authentic Sanskrit passages (Devanagari / IAST verse and sutras).
3. **Expand Validation Set:** Increase validation sources to at least 5 independent source groups with >15,000 tokens to ensure reliable model checkpoint selection.
