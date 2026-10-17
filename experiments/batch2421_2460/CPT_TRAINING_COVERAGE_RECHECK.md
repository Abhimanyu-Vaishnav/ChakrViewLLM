# CPT TRAINING COVERAGE INDEPENDENT RECHECK REPORT

**Batch:** 2421–2460  
**Audit Phase:** Phase 3 — Training Coverage & Sanskrit Representation Audit  
**Training Split Manifest:** [`experiments/batch2021_2060/SPLIT_V2_MANIFEST.json`](file:///d:/Project/ChakrView/experiments/batch2021_2060/SPLIT_V2_MANIFEST.json)  
**Cleaned Corpus Manifest:** [`experiments/batch2101_2140/CORPUS_V2_CLEANED_MANIFEST.json`](file:///d:/Project/ChakrView/experiments/batch2101_2140/CORPUS_V2_CLEANED_MANIFEST.json)  
**Canonical Tokenizer:** [`release/chakrview-v0.5.0/data/experiments/vocab_4096`](file:///d:/Project/ChakrView/release/chakrview-v0.5.0/data/experiments/vocab_4096)  
**Machine-Readable Summary:** [`experiments/batch2421_2460/CPT_TRAINING_COVERAGE_RECHECK.json`](file:///d:/Project/ChakrView/experiments/batch2421_2460/CPT_TRAINING_COVERAGE_RECHECK.json)  
**Status:** **INDEPENDENTLY VERIFIED — ZERO CLASSICAL SANSKRIT TRAINING COVERAGE**  

---

## 1. Executive Summary & Verification Protocol

Rather than determining training coverage by superficial string searches (e.g., checking if source IDs contain `sa` or `sanskrit`), this audit followed a rigorous bottom-up procedure:
1. Resolved all 782 training document IDs directly from `SPLIT_V2_MANIFEST.json`.
2. Verified that each document ID exists uniquely without aliases or duplication and mapped to the primary corpus records in `CORPUS_V2_CLEANED_MANIFEST.json`.
3. Extracted actual source IDs and language metadata attributes stored in the verified records.
4. Tokenized every training document using the canonical BPE tokenizer (`vocab.json` + `merges.json`).
5. Computed target tokens using the documented sequence chunking convention ($max\_seq = 256$, prediction target length $\sum (\text{len}(chunk) - 1)$).
6. Separated source-family coverage, language metadata coverage, and tokenizer target-token coverage.

---

## 2. Training Input Resolution & Mapping Integrity

- **Train Document IDs in Split Manifest:** 782
- **Unique Document IDs:** 782 (0 duplicates)
- **Resolved Against Cleaned Corpus:** 782 / 782 (100% resolved, 0 missing IDs)
- **Total Training Documents Analyzed:** 782

---

## 3. Source-Family, Language Metadata & Token Coverage Breakdown

The 782 training documents encompass exactly 6 sources across 3 language categories:

| Source ID | Domain / Description | Language Metadata | Doc Count | Raw Encoded Tokens | Chunked Target Tokens (256-eval conv) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `ietf_rfc793_tcp` | Technical Standards | `en` | 367 | 61,905 | 61,538 |
| `gutenberg_tagore_gitanjali` | Literature / Poetry | `en` | 195 | 34,007 | 33,812 |
| `wikipedia_en_turing_machine` | Computer Science | `en` | 74 | 16,678 | 16,604 |
| `wikipedia_hi_chemistry` | Natural Science | `hi` | 58 | 14,885 | 14,827 |
| `cpython_algorithms_heapq` | Source Code | `code` | 56 | 9,814 | 9,758 |
| `wikipedia_hi_premchand` | Literature / Fiction | `hi` | 32 | 10,187 | 10,155 |
| **Total** | | **3 languages** | **782** | **147,476** | **146,694** |

---

## 4. Sanskrit Training Representation Verification

1. **Document-Level Sanskrit Presence:**
   - Classical Sanskrit documents in training split: **0**
   - Documents with language code `sa` in training split: **0**
2. **Target Token Sanskrit Presence:**
   - Evaluated target tokens for Classical Sanskrit in training split: **0**
3. **Corpus Split Allocation of Sanskrit Sources:**
   - In `CORPUS_V2_CLEANED_MANIFEST.json`, there are two sources associated with Sanskrit:
     - `wikipedia_sa_bhagavad_gita` (61 documents, Classical Sanskrit in Devanagari script, language: `sa`): **100% assigned to `test` split**.
     - `wikipedia_hi_sanskrit_sahitya` (9 documents, Hindi literary commentary about Sanskrit literature, language: `hi`): **100% assigned to `test` split**.
   - Neither of these sources has any document in the `train` or `validation` split.

---

## 5. Scientific Separation of Findings

- **Source-Family Coverage:** The training corpus contains 6 specific source families (technical networking, poetry, automata theory, Hindi chemistry, Python heapq, and Hindi prose). Classical Sanskrit philosophical/epic literature (`wikipedia_sa_bhagavad_gita`) is entirely absent from the training split.
- **Language Metadata Coverage:** The training corpus explicitly contains `en` (636 docs), `hi` (90 docs), and `code` (56 docs). Language code `sa` is completely absent.
- **Target-Token Coverage:** Total target tokens trained were 146,694. Exactly 0 tokens of Classical Sanskrit text were seen during continued pre-training.
- **Causality Boundary:** While the zero-training-token status is an independently established ground truth, we emphasize that observing a held-out regression on an unreinforced language source is consistent with representational drift, but causal attribution requires a controlled intervention experiment.
