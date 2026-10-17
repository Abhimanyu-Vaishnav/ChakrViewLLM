# ChakrView Corpus v2 Integrity & Data Provenance Audit — Batches 2061–2100
**Milestone:** Batches 2061–2100  
**Audit Scope:** Independent Reconciliation, Provenance Verification, Extraction Integrity & Leakage Analysis  
**Date:** October 2026  
**Status:** Audit Complete  

---

## 1. Executive Summary & Verification Totals

We independently audited `experiments/batch2021_2060/CORPUS_V2_MANIFEST.json` and `SPLIT_V2_MANIFEST.json` using fresh re-encoding, full string hashing, and independent n-gram / shingle analysis.

### Primary Reconciled Metrics:
- **Total Declared Documents:** `1,258`
- **Total Actual Documents:** `1,258` (**100.0% Match**)
- **Total Declared Tokens:** `245,758`
- **Total Re-Encoded Tokens:** `245,758` (**100.0% Match** using canonical `vocab_4096`)
- **Total Characters:** `533,433` (Sum of document strings: `533,433`, **100.0% Match**)
- **Document SHA-256 Mismatches:** `0` (All 1,258 documents match `hashlib.sha256(doc['text'].encode('utf-8')).hexdigest()`)
- **UTF-8 Decoding Errors:** `0`
- **Source-Level Cross-Split Contamination:** `0` (Source families are strictly disjoint across Train, Validation, and Test)

---

## 2. Partition & Source Distribution Reconciliations

### 2.1 Split Summary
| Partition | Documents | Tokens | Token Share (%) | Source Families Assigned |
|:---|:---:|:---:|:---:|:---|
| **Train** | 782 | 147,476 | 60.01% | 6 families (`rfc793`, `gitanjali`, `turing_machine`, `chemistry`, `premchand`, `heapq`) |
| **Validation** | 186 | 32,617 | 13.27% | 3 families (`google_hinglish`, `hi_ganit`, `bisect`) |
| **Test** | 290 | 65,665 | 26.72% | 4 families (`loney_geometry`, `bhagavad_gita`, `information_theory`, `sanskrit_sahitya`) |
| **Total** | **1,258** | **245,758** | **100.00%** | **13 families** |

### 2.2 Detailed Source Family Breakdown
| Source ID | Split | Domain | Lang | Docs | Tokens | Tokens / Split (%) | License |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| `ietf_rfc793_tcp` | Train | Technical | `en` | 367 | 61,905 | 41.98% | Public Domain / IETF RFC |
| `gutenberg_tagore_gitanjali` | Train | Prose | `en` | 195 | 34,007 | 23.06% | Public Domain (Project Gutenberg) |
| `wikipedia_en_turing_machine` | Train | Technical | `en` | 74 | 16,678 | 11.31% | CC BY-SA 4.0 (Wikipedia) |
| `wikipedia_hi_chemistry` | Train | Technical | `hi` | 58 | 14,885 | 10.09% | CC BY-SA 4.0 (Wikipedia) |
| `wikipedia_hi_premchand` | Train | Prose | `hi` | 32 | 10,187 | 6.91% | CC BY-SA 4.0 (Wikipedia) |
| `cpython_algorithms_heapq` | Train | Code | `code` | 56 | 9,814 | 6.65% | Python Software Foundation (PSF-2.0) |
| **Train Subtotal** | **Train** | - | - | **782** | **147,476** | **100.00%** | - |
| `google_hinglish_top_dialogues` | Val | Hinglish | `hi-en` | 159 | 26,607 | 81.57% | CC BY-SA 4.0 (Google Research) |
| `wikipedia_hi_ganit` | Val | Math | `hi` | 19 | 4,741 | 14.54% | CC BY-SA 4.0 (Wikipedia) |
| `cpython_algorithms_bisect` | Val | Code | `code` | 8 | 1,269 | 3.89% | Python Software Foundation (PSF-2.0) |
| **Val Subtotal** | **Val** | - | - | **186** | **32,617** | **100.00%** | - |
| `gutenberg_loney_coordinate_geometry` | Test | Math | `en` | 150 | 24,632 | 37.51% | Public Domain (Project Gutenberg) |
| `wikipedia_sa_bhagavad_gita` | Test | Sanskrit | `sa` | 61 | 20,244 | 30.83% | CC BY-SA 4.0 / Public Domain Text |
| `wikipedia_en_information_theory` | Test | Math | `en` | 70 | 15,982 | 24.34% | CC BY-SA 4.0 (Wikipedia) |
| `wikipedia_hi_sanskrit_sahitya` | Test | Hindi | `hi` | 9 | 4,807 | 7.32% | CC BY-SA 4.0 (Wikipedia) |
| **Test Subtotal** | **Test** | - | - | **290** | **65,665** | **100.00%** | - |

---

## 3. Investigation of Specific Risk Areas

### Risk A: Hinglish Data Construction & Conversational Coverage
- **Source Examined:** `google_hinglish_top_validation.txt` (derived from Google Hinglish-TOP human-annotated dataset, CC BY-SA 4.0).
- **Structure:** Short task-oriented conversational dialogue utterances and queries (e.g., *"San Francisco me traffic kaisi hai this evening"*, *"mere timer ko roko"*, *"kripya aaj evening ke liye Jogtime ka alarm set kariye"*).
- **Ingestion Artifact:** The ingestion script concatenates multiple query lines with period/space delimiters until a chunk length of $\ge 350$ characters is reached.
- **Finding:**
  1. The queries themselves are natural, human-annotated code-switched Hinglish utterances reflecting authentic bilingual Hindi-English syntax.
  2. However, the concatenation across unrelated domain queries (e.g., setting a timer immediately followed by asking about New York concerts, then traffic on highway 494) introduces **artificial narrative transitions**.
  3. **Operational Implication:** The dataset serves well as a tokenization and vocabulary exposure corpus for bilingual phrasing, but it should **not** be claimed as long-form multi-turn conversational discourse.

### Risk B: Sanskrit Extraction Quality & Markup Residue
- **Source Examined:** `authentic_sanskrit_gita.txt` (90,662 raw bytes, 207 text lines).
- **Findings:**
  1. **Devanagari Purity:** 85.7% of characters are pure Devanagari script (`\u0900`–`\u097f`), representing genuine verses and commentary.
  2. **Residual Wiki Markup:** Contains **184 occurrences** of raw MediaWiki bold/italic apostrophe markers (`'''` or `''`) and **45 lines** starting with indentation colons (`:`).
  3. **Contaminated Documents in Corpus v2:** In `CORPUS_V2_MANIFEST.json`, out of 61 documents for `wikipedia_sa_bhagavad_gita`, **30 documents (49.2%)** contain `'''` or `''` apostrophe sequences, and **17 documents (27.9%)** contain indentation colons.
  4. **Proposed Correction:** Apply a regex cleanup pass (`re.sub(r"'{2,5}", "", text)` and stripping line-initial colons `re.sub(r"^:+", "", line)`) prior to chunking. Existing raw artifacts must not be modified in-place; the clean ingestion pass should be registered for Phase 5 continued pretraining.

### Risk C: Historical Pretraining Contamination Audit
- **Canonical Training History:** Trained up to step 10,000 on synthetic/algorithmic shards in `data/tokenized/stage_c/`.
- **Manifests Inspected:** `stage_b_manifest.json`, `stage_c_manifest.json`, `experiments/batch16_21_combined/history_10000.json`.
- **Classification:**
  - **Confirmed Overlap:** None found. Corpus v2 utilizes natural texts (Project Gutenberg, Wikipedia, RFC 793, CPython stdlib), whereas canonical Stage C pretraining utilized indigenous algorithmic synthetic generation.
  - **Suspected Overlap:** CPython standard library functions (`heapq` in Train, `bisect` in Validation) share general Python standard library syntactical idioms with earlier coding benchmarks.
  - **Historical Training Raw Corpora Availability:** Tokenized binary shards (`shard_00000.bin`–`shard_00026.bin`) exist, but original un-tokenized raw training text generators are only partially documented in early batch logs.
  - **Strict Verdict:** No confirmed data leakage or document hash match exists between canonical pretraining records and Corpus v2.

### Risk D: Leakage Analysis (Exact Hashes, 8-Grams & Near-Duplicates)
We executed an independent, three-tier leakage evaluation:
1. **Exact Document SHA-256 Overlap:**
   - Train vs Val: `0`
   - Train vs Test: `0`
   - Val vs Test: `0`
2. **Whitespace-Delimited 8-Word Sequences (8-grams):**
   - Evaluated 47,833 Train 8-grams, 12,861 Val 8-grams, and 18,543 Test 8-grams.
   - **Cross-split overlap: 0 shared 8-grams** across all partition pairs.
3. **Regex-Normalized Word-Token 8-Grams:**
   - Under case-lowered, punctuation-stripped normalization, 7 shared 8-grams were detected between Train and Val, and 5 between Train and Test.
   - **Inspection of Shared Sequences:** All shared occurrences are standard Hindi function phrase sequences or multi-word dictionary compounds (e.g., *"के रूप में परिभाषित किया"* [defined in the form of], *"प्राकृतिक विज्ञान"* [natural science], *"गूगल पुस्तक लेखक"* [Google book author citation]). Zero verbatim narrative or passage leakage exists.
4. **Near-Duplicate 5-Shingle Jaccard Similarity:**
   - Sampled across partition boundaries.
   - Max Jaccard Val-to-Train: `0.0282` (CPython `bisect` vs `heapq` standard library docstring headers).
   - Max Jaccard Test-to-Train: `0.0000`.
   - **Conclusion:** Near-duplicate contamination is negligible and non-systemic.

---

## 4. Phase B Verdict

**PASSED WITH DOCUMENTED CONDITIONS.**  
Corpus v2 counts reconcile with 100% precision, source provenance is complete, and split isolation is verified. The residual MediaWiki markup in Sanskrit and the artificial concatenation in Hinglish are cataloged as remediable data-cleaning improvements for Phase 5 pre-flight.
