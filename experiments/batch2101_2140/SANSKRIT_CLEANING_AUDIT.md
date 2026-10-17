# ChakrView Sanskrit Cleaning Audit — Batches 2101–2140
**Milestone:** Batches 2101–2140  
**Phase:** Phase C — Versioned Sanskrit Cleaning  
**Date:** October 2026  
**Auditor:** Independent Preflight Release Agent  

---

## 1. Executive Summary & Objective

In Batches 2061–2100, the data integrity audit detected residual MediaWiki formatting in raw Sanskrit extraction files, impacting 49.2% of Sanskrit documents in Corpus v2.

In this mission, we implemented a deterministic, non-destructive, versioned cleaning transformation that strips confirmed MediaWiki markup artifacts without rewriting legitimate Sanskrit text or modifying raw files or the committed `CORPUS_V2_MANIFEST.json`. The cleaned corpus is published as `experiments/batch2101_2140/CORPUS_V2_CLEANED_MANIFEST.json`.

---

## 2. Before vs After Artifact Inventory

| Markup / Formatting Dimension | Corpus v2 Raw Sanskrit | Corpus v2 Cleaned Manifest | Net Reduction |
|:---|:---:|:---:|:---:|
| **MediaWiki Bold/Italic Apostrophes (`'''` or `''`)** | 184 occurrences | **0 occurrences** | -184 (-100.0%) |
| **Line-Initial Indentation Colons (`:`)** | 45 occurrences | **0 occurrences** | -45 (-100.0%) |
| **Wiki Links (`[[...]]`)** | 0 occurrences | 0 occurrences | 0 |
| **Wiki Templates (`{{...}}`)** | 0 occurrences | 0 occurrences | 0 |
| **HTML Tags (`<...>`)** | 0 occurrences | 0 occurrences | 0 |
| **Documents Containing Markup** | 30 / 61 (49.2%) | **0 / 61 (0.0%)** | -30 (-100.0%) |
| **Total Sanskrit Characters** | 33,151 | 32,527 | -624 (-1.88%) |
| **Total Sanskrit Tokens (`vocab_4096`)** | 20,244 | 19,638 | -606 (-2.99%) |

---

## 3. Representative Before & After Text Examples

### Example 1: Title & Formatting Markup (Document `NATURAL_V2_TEST_01198`)
- **Before Cleaning:**
  ```text
  गीतोपदेशः भगवता गीता '''श्रीमद्भगवद्गीता'''। एतस्य गीतोपदेशः इत्यपि नामान्तरं वर्तते। श्रीकृष्णः अत्र उपदेशकः श्रोता अर्जुनः।
  ```
- **After Cleaning:**
  ```text
  गीतोपदेशः भगवता गीता श्रीमद्भगवद्गीता। एतस्य गीतोपदेशः इत्यपि नामान्तरं वर्तते। श्रीकृष्णः अत्र उपदेशकः श्रोता अर्जुनः।
  ```
- **Verification:** Apostrophe markers around `'''श्रीमद्भगवद्गीता'''` are cleanly removed, preserving the Sanskrit noun intact.

### Example 2: Shloka Indentation & Italic Quotes (Document `NATURAL_V2_TEST_01198`)
- **Before Cleaning:**
  ```text
  :'''''गीता सुगीता कर्तव्या किमन्यैः शास्त्रविस्तरैः।'''''
  :'''''या स्वयं पद्मनाभस्य मुखपद्माद्विनिःसृता॥'''''
  ```
- **After Cleaning:**
  ```text
  गीता सुगीता कर्तव्या किमन्यैः शास्त्रविस्तरैः।
  या स्वयं पद्मनाभस्य मुखपद्माद्विनिःसृता॥
  ```
- **Verification:** The line-initial colon (`:`) and fivefold apostrophes (`'''''`) are removed; the authentic shloka meter and danda punctuation (`।`, `॥`) are fully preserved.

---

## 4. Linguistic Authenticity & Inspection

Manual inspection of cleaned documents confirms:
1. **Linguistic Purity:** Shlokas from the Gita, introductory verses, and philosophical commentary retain authentic classical Sanskrit vocabulary and sandhi rules.
2. **Punctuation Preservation:** Single dandas (`।`), double dandas (`॥`), and legitimate hyphens are untouched.
3. **No Synthesized Content:** Zero characters or words were added or generated.
4. **Document Integrity:** All 61 Sanskrit documents retain their unique IDs (`NATURAL_V2_TEST_01198` to `NATURAL_V2_TEST_01258`), source family `wikipedia_sa_bhagavad_gita`, and test split allocation.

---

## 5. Provenance & License Record

- **Source Family:** `wikipedia_sa_bhagavad_gita`
- **Source Origin:** Sanskrit Wikipedia article on the Bhagavad Gita and classical Sanskrit public domain verses.
- **License:** CC BY-SA 4.0 / Public Domain classical text.
- **URL:** Recorded in `SOURCE_PROVENANCE_MANIFEST.json`.
- **Integrity Compliance:** Raw file `experiments/batch2021_2060/data/raw_sources/authentic_sanskrit_gita.txt` and `experiments/batch2021_2060/CORPUS_V2_MANIFEST.json` remain completely unmodified.

**Verdict:** PASSED. The cleaned manifest `CORPUS_V2_CLEANED_MANIFEST.json` is verified and ready for downstream preflight.
