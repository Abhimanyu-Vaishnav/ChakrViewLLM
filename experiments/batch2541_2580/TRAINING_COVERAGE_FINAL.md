# FINAL TRAINING COVERAGE DERIVATION REPORT

**Batch:** 2541–2580  
**Audit Phase:** Phase 1 — Fully Computed Training Coverage & Zero-Assertion Verification  
**Primary Split Manifest:** [`experiments/batch2021_2060/SPLIT_V2_MANIFEST.json`](file:///d:/Project/ChakrView/experiments/batch2021_2060/SPLIT_V2_MANIFEST.json)  
**Cleaned Corpus Manifest:** [`experiments/batch2101_2140/CORPUS_V2_CLEANED_MANIFEST.json`](file:///d:/Project/ChakrView/experiments/batch2101_2140/CORPUS_V2_CLEANED_MANIFEST.json)  
**Canonical Tokenizer:** [`release/chakrview-v0.5.0/data/experiments/vocab_4096`](file:///d:/Project/ChakrView/release/chakrview-v0.5.0/data/experiments/vocab_4096)  
**Per-Document Audit Records:** [`experiments/batch2541_2580/TRAINING_COVERAGE_PER_DOCUMENT.jsonl`](file:///d:/Project/ChakrView/experiments/batch2541_2580/TRAINING_COVERAGE_PER_DOCUMENT.jsonl)  
**Machine-Readable Summary:** [`experiments/batch2541_2580/TRAINING_COVERAGE_FINAL.json`](file:///d:/Project/ChakrView/experiments/batch2541_2580/TRAINING_COVERAGE_FINAL.json)  
**Reconciliation Record:** [`experiments/batch2541_2580/TRAINING_COVERAGE_RECONCILIATION.json`](file:///d:/Project/ChakrView/experiments/batch2541_2580/TRAINING_COVERAGE_RECONCILIATION.json)  
**Status:** **100% COMPUTED FROM RAW EVIDENCE — ZERO HARDCODED CLAIMS**  

---

## 1. Executive Summary & Audit Correction

In earlier batches, several summary fields in training coverage JSON artifacts were populated via static dictionary construction or assumed properties. In Phase 1 of Batch 2541–2580, we replaced every static field with explicit runtime computations:
1. `resolved_in_cleaned_corpus`: Computed dynamically as `len(train_ids) - len(missing_train_ids) = 782`.
2. `missing_document_ids`: Populated by checking each training ID against `CORPUS_V2_CLEANED_MANIFEST.json` (`len == 0`).
3. `duplicate_document_ids`: Calculated explicitly using `Counter(train_ids)` (`len == 0`).
4. `corpus_wide_sanskrit_docs_split_distribution`: Dynamically calculated across all splits in `SPLIT_V2_MANIFEST.json` for all 70 corpus Sanskrit documents (`{'test': 70}`).
5. `reconciliation_status`: Derived directly from boolean equality expressions (`is_doc_count_match and is_raw_token_count_match and is_zero_sanskrit`).
6. Partition exhaustiveness verified: All 1,258 documents in the cleaned corpus belong to exactly one split (782 train, 186 validation, 290 test; zero overlaps, zero unassigned).

---

## 2. Training Sequence Convention vs Evaluation Target Convention

Inspection of the actual CPT training runner ([`experiments/batch2141_2180/cpt_runner.py`](file:///d:/Project/ChakrView/experiments/batch2141_2180/cpt_runner.py#L180-L205)) reveals:
- **Training Batching:** Sequences longer than $context\_len + 1 = 257$ have windows sampled randomly during training steps (`rng.randint(0, max_start)`), predicting `tgt = chunk[1:]` from `inp = chunk[:-1]`.
- **Corpus Evaluation Chunking:** In deterministic corpus auditing, documents are non-overlappingly tiled with window stride $max\_seq - 1 = 255$, ensuring every token transition is evaluated exactly once without double-counting.
- **Token Separation:**
  - **Raw Encoded Tokens:** $147,476$ tokens across 782 training documents.
  - **Tiled Target Tokens ($max\_seq=256$):** $146,694$ target tokens across 782 training documents.
  - Both figures are separated and independently reconciled.

---

## 3. Derived Breakdown

| Source ID | Language | Documents | Raw Encoded Tokens | Tiled Target Tokens ($max\_seq=256$) |
| :--- | :---: | :---: | :---: | :---: |
| `ietf_rfc793_tcp` | `en` | 367 | 61,905 | 61,538 |
| `gutenberg_tagore_gitanjali` | `en` | 195 | 34,007 | 33,812 |
| `wikipedia_en_turing_machine` | `en` | 74 | 16,678 | 16,604 |
| `wikipedia_hi_chemistry` | `hi` | 58 | 14,885 | 14,827 |
| `cpython_algorithms_heapq` | `code` | 56 | 9,814 | 9,758 |
| `wikipedia_hi_premchand` | `hi` | 32 | 10,187 | 10,155 |
| **Total** | **3 distinct** | **782** | **147,476** | **146,694** |

- **Classical Sanskrit (`sa`) in Train:** **0 documents, 0 raw tokens, 0 target tokens**.
