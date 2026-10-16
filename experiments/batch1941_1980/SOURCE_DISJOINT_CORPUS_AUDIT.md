# CHAKRVIEW — Batches 1941–1980
## Audit of the Existing Source-Disjoint Corpus (`source_disjoint_corpus.json`)

### Executive Summary

In Phase 4 of Batches 1901–1940, a candidate benchmark of 88 documents ([source_disjoint_corpus.json](file:///d:/Project/ChakrView/experiments/batch1901_1940/data/source_disjoint_corpus.json)) was assembled to evaluate checkpoint generalization beyond the contaminated Batch 1861 training corpus. While the 6-word overlap filtering prevented exact verbatim train string leakage, a rigorous multi-tier audit reveals structural limitations that prevent this corpus from being considered a fully external, gold-standard natural benchmark.

---

### 1. Forensic Findings on the 88-Document Benchmark

#### A. Presence of Stage-B Derived Synthetic Files
Although the 88 documents were selected from files outside the Batch 1861 training manifest, several selected source files were themselves created during Stage B synthetic corpus processing:
- `data/processed/stage_b/validation/sanskrit.txt` (9 documents)
- `data/processed/stage_b/validation/numbers.txt` (6 documents)
- `data/validation/code.txt` (2 documents)
- `data/validation/hinglish.txt` (2 documents)

While these files did not overlap with Batch 1861 training chunks, they originated from the same synthetic generation pipelines as Stage B data.

#### B. Artificial and Synthetic Content Structures
1. **`numbers_dataset.txt` (`indep_mathematics_0044`, `0045`):**
   - Contains artificial lists of sequential integers:
     `"0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 ..."`
   - Followed by phone number strings:
     `"+91-9876543210 +1-800-555-0199 +44-20-7946-0958 ..."`
   - These are synthetic stress tests, not natural mathematical or numerical text.
2. **`code_configs.txt` (`indep_code_0058`, `0059`):**
   - Contains raw JSON configuration schemas rather than natural Python code or algorithm implementations.

#### C. Cross-Split Overlap with Prior Validation Sets
Cross-referencing the 88 documents against the pre-existing validation splits revealed residual overlap:
- `indep_prose_0065` (`mixed_foundation.txt`) shared 6-word substrings with earlier validation sets (`"Devanagari script ke words ko BPE"`).
- `indep_hinglish_0082` (`hinglish.txt`) shared substrings with validation dialogue files (`"karo aur saare tests verify karo."`).

#### D. Domain Imbalance
The 88 documents had significant domain skew:
- Hindi: 29 documents (33.0%)
- Mathematics / Numbers: 16 documents (18.2%)
- Sanskrit: 12 documents (13.6%)
- Prose: 9 documents (10.2%)
- Technical: 8 documents (9.1%)
- Hinglish: 7 documents (8.0%)
- Code: 7 documents (8.0%)

---

### 2. Audit Conclusion & Recommendation

1. **Validity of Prior Audit Findings:**
   The 88-document benchmark was highly successful in exposing the **8.7x inflation** of Batch 1861's claims and capturing the 300-step overfitting regression. Because it eliminated the repeated prose boilerplate and intra-file continuous paragraph slices, its directional findings remain robust.
2. **Requirement for Pure Natural Text:**
   For future training and rigorous capability benchmarking, ChakrView must transition to a genuinely natural text corpus drawn from documented public-domain literature, open-source algorithms, and standard specifications, with zero synthetic lists or procedurally generated templates.
