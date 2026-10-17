# ChakrView Data Leakage & Split Isolation Audit — Batches 2021–2060
**Milestone:** Batches 2021–2060  
**Branch:** `dev/v0.6-v0.7-validation`  
**Status:** Passed  
**Date:** October 2026  

---

## 1. Audit Scope & Motivation

Previous milestones (Batches 1861–1900 and 1901–1940) demonstrated that synthetic template-based data generators introduce massive n-gram leakage between splits, producing falsely inflated cross-entropy reductions. In Batches 1941–1980, natural text was introduced, but S. L. Loney's geometry text accounted for 98.6% of test tokens, masking whether general language capability was being evaluated.

This audit evaluates the integrity of **Corpus v2** and **Split v2** under three strict isolation dimensions:
1. **Source-Family Isolation:** Ensuring that no author, book, module, or repository is shared across Train, Validation, and Test splits.
2. **Exact Cryptographic Overlap:** Verifying that zero documents share SHA-256 hashes across splits.
3. **N-Gram Subsequence Leakage:** Measuring overlapping 8-grams across all splits to detect verbatim or near-verbatim passage leakage.

---

## 2. Source-Family Assignment Matrix

In Corpus v2, every source family is assigned strictly and deterministically to a single split partition.

| Partition | Source ID | Domain | Language | Justification |
|:---|:---|:---|:---|:---|
| **TRAIN** | `gutenberg_tagore_gitanjali` | Prose | `en` | Poetic and reflective English prose |
| **TRAIN** | `ietf_rfc793_tcp` | Technical | `en` | Formal procedural engineering specifications |
| **TRAIN** | `cpython_algorithms_heapq` | Code | `code` | Real executable Python data structures |
| **TRAIN** | `wikipedia_hi_premchand` | Prose | `hi` | High-quality literary Hindi narrative |
| **TRAIN** | `wikipedia_en_turing_machine` | Technical | `en` | Computer science expository theory |
| **TRAIN** | `wikipedia_hi_chemistry` | Technical | `hi` | Scientific Devanagari exposition |
| **VALIDATION** | `cpython_algorithms_bisect` | Code | `code` | Algorithm implementations completely disjoint from `heapq` |
| **VALIDATION** | `wikipedia_hi_ganit` | Technical | `hi` | Hindi mathematical explanations disjoint from Premchand/Chemistry |
| **VALIDATION** | `google_hinglish_top_dialogues` | Hinglish | `hi-en` | Authentic code-switched conversational validation |
| **TEST** | `gutenberg_loney_coordinate_geometry` | Math | `en` | Analytical geometry derivations (capped at 150 docs) |
| **TEST** | `wikipedia_hi_sanskrit_sahitya` | Hindi | `hi` | Hindi literary exposition disjoint from training authors |
| **TEST** | `wikipedia_sa_bhagavad_gita` | Sanskrit | `sa` | Authentic classical Sanskrit verses |
| **TEST** | `wikipedia_en_information_theory` | Math | `en` | Mathematical entropy theory disjoint from Loney geometry |

### Set Disjointness Verification:
- $\text{Train} \cap \text{Validation} = \emptyset$
- $\text{Train} \cap \text{Test} = \emptyset$
- $\text{Validation} \cap \text{Test} = \emptyset$

---

## 3. Cryptographic & Exact Match Audit

Every chunked document in `CORPUS_V2_MANIFEST.json` was hashed using SHA-256 over its raw UTF-8 bytes.

```python
train_shas = {d['sha256'] for d in docs if d['split'] == 'train'}
val_shas = {d['sha256'] for d in docs if d['split'] == 'validation'}
test_shas = {d['sha256'] for d in docs if d['split'] == 'test'}
```

### Results:
- **Train vs Validation SHA-256 Overlap:** `0` documents.
- **Train vs Test SHA-256 Overlap:** `0` documents.
- **Validation vs Test SHA-256 Overlap:** `0` documents.

---

## 4. N-Gram Subsequence Leakage Audit

To ensure that common extraction artifacts, licensing headers, or verbatim phrases did not leak across partitions, we extracted all sliding 8-word sequences (8-grams) across all normalized document strings ($N=8$).

### Quantitative N-Gram Metrics:

| Split Partition | Unique 8-Gram Count | Overlap with Train | Overlap with Validation | Overlap with Test | Leakage Rate |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Train** | 47,826 | - | 0 | 0 | **0.000%** |
| **Validation** | 12,860 | 0 | - | 0 | **0.000%** |
| **Test** | 18,532 | 0 | 0 | - | **0.000%** |

### Findings:
- Total unique 8-grams evaluated: **79,218**.
- Total overlapping 8-grams across any partition boundary: **0**.
- **Conclusion:** There is zero n-gram leakage between the training corpus and the evaluation benchmarks.

---

## 5. Test Split Concentration Audit

In Batch 1941, the test split suffered from an extreme concentration failure where S. L. Loney's *Coordinate Geometry* represented 98.6% of all test tokens.

### Before vs After Concentration Comparison:

```
Batch 1941 Test Split (Corpus v1):
  [========================================================================================] 98.6% Loney Geometry
  [|] 1.4% Sanskrit Sahitya (Hi)

Batch 2021 Test Split (Corpus v2):
  [====================================] 37.5% Loney Geometry
  [==============================] 30.8% Bhagavad Gita (Sanskrit)
  [=======================] 24.3% Information Theory (En Math)
  [=======] 7.3% Sanskrit Sahitya (Hindi Lit)
```

By capping Loney to 150 documents and introducing Information Theory (`en`), Sanskrit Sahitya (`hi`), and Classical Sanskrit (`sa`), no single source family exceeds 37.5% of test tokens, establishing a resilient multi-source, multi-lingual evaluation benchmark.

---

## 6. Audit Verdict

**PASSED.**  
Corpus v2 satisfies all criteria for strict source-family isolation, zero cross-split leakage, and balanced test representation.
