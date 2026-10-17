# ChakrView Natural Corpus Expansion Report — Batches 2021–2060
**Milestone:** Batches 2021–2060  
**Branch:** `dev/v0.6-v0.7-validation`  
**Status:** Complete  
**Date:** October 2026  

---

## 1. Executive Summary

In Batches 1941–1980 and 1981–2020, natural corpus construction succeeded in removing synthetic template bias, but an independent evaluation audit exposed critical structural bottlenecks:
1. **Severe Test Concentration:** A single source family (*The Elements of Coordinate Geometry* by S. L. Loney) accounted for **98.6% of test split tokens**, reducing the entire test benchmark effectively to an evaluation of 19th-century coordinate geometry prose.
2. **Missing Linguistic & Domain Diversity:** Crucial domains—notably conversational Hinglish, authentic Sanskrit text (as opposed to Hindi articles describing Sanskrit), and formal computer science documentation—were unrepresented or unseparated.
3. **Imbalanced Token Counts:** Uncapped source extraction skewed dataset composition.

In Batches 2021–2060, we resolved these bottlenecks by building **Corpus v2** and **Split v2**:
- Ingested **5 new independent, verifiably reusable source families** across Computer Science, Information Theory, Chemistry (Hindi), Conversational Hinglish, and Classical Sanskrit.
- Expanded the corpus to **13 verified source families** across 7 domains and 5 language representations (`en`, `hi`, `hi-en`, `sa`, `code`).
- Capped S. L. Loney's geometry text to 150 documents, reducing its test token share from **98.6% down to 37.5%**, and diversified the test benchmark across 4 independent sources (`en`, `hi`, `sa`, and `mathematics`).
- Achieved **0% document and 0% 8-gram cross-split contamination** under strict source-family isolation.
- Total Corpus v2 size: **1,258 deduplicated documents**, **245,758 tokens**, and **533,433 UTF-8 characters**.

---

## 2. Ingested Sources Catalog & Verification

All 13 sources in Corpus v2 possess cryptographic content verification, verified provenance URLs, clear legal licensing (Public Domain, CC BY-SA 4.0, Apache-2.0, or Python Software Foundation License), and deduplicated content.

### Summary Table of All 13 Source Families

| Source ID | Title / Work | Author / Provider | Domain | Language | License | Total Docs | Total Tokens | Split Assignment |
|:---|:---|:---|:---|:---|:---|:---:|:---:|:---:|
| `gutenberg_tagore_gitanjali` | *Gitanjali* | Rabindranath Tagore / Project Gutenberg | Prose | `en` | Public Domain | 195 | 34,007 | Train |
| `ietf_rfc793_tcp` | *RFC 793 Transmission Control Protocol* | J. Postel / IETF | Technical | `en` | IETF Trust / Public Domain | 367 | 61,905 | Train |
| `cpython_algorithms_heapq` | *Python `heapq` Module & Docs* | Python Software Foundation | Code | `code` | PSF License 2.0 | 56 | 9,814 | Train |
| `wikipedia_hi_premchand` | *Munshi Premchand Biography & Works* | Hindi Wikipedia Contributors | Hindi | `hi` | CC BY-SA 4.0 | 32 | 10,187 | Train |
| `wikipedia_en_turing_machine` | *Turing Machine Expository Theory* | English Wikipedia Contributors | Technical / CS | `en` | CC BY-SA 4.0 | 74 | 16,678 | Train |
| `wikipedia_hi_chemistry` | *Rasayan Vigyan (Chemistry Overview)* | Hindi Wikipedia Contributors | Technical / Hindi | `hi` | CC BY-SA 4.0 | 58 | 14,885 | Train |
| `cpython_algorithms_bisect` | *Python `bisect` Module & Docs* | Python Software Foundation | Code | `code` | PSF License 2.0 | 8 | 1,269 | Validation |
| `wikipedia_hi_ganit` | *Ganit (Mathematics in Hindi)* | Hindi Wikipedia Contributors | Hindi / Math | `hi` | CC BY-SA 4.0 | 19 | 4,741 | Validation |
| `google_hinglish_top_dialogues` | *Google Research Hinglish Top Dialogues* | Google Research (Gupta et al.) | Hinglish | `hi-en` | Apache-2.0 | 159 | 26,607 | Validation |
| `gutenberg_loney_coordinate_geometry` | *The Elements of Coordinate Geometry* | S. L. Loney / Project Gutenberg | Mathematics | `en` | Public Domain | 150 | 24,632 | Test |
| `wikipedia_hi_sanskrit_sahitya` | *Sanskrit Sahitya (Literature Overview)* | Hindi Wikipedia Contributors | Hindi | `hi` | CC BY-SA 4.0 | 9 | 4,807 | Test |
| `wikipedia_sa_bhagavad_gita` | *Bhagavad Gita Classical Slokas* | Sanskrit Wikipedia / Wikisource | Sanskrit | `sa` | CC BY-SA 4.0 / PD | 61 | 20,244 | Test |
| `wikipedia_en_information_theory` | *Information Theory & Entropy* | English Wikipedia Contributors | Mathematics | `en` | CC BY-SA 4.0 | 70 | 15,982 | Test |

---

## 3. Detailed Examination of New Expansion Sources

### 3.1 Google Research Hinglish Top Dialogues (`google_hinglish_top_dialogues`)
- **Origin & Provenance:** Sourced from the official Google Research repository (`google-research/language/hinglish_top`).
- **Nature of Material:** 159 human-annotated conversational Hinglish queries exhibiting natural code-switching between Hindi (written in Latin script) and English (e.g., *"mujhe reminder set karna hai for tomorrow 9 am call to mom"*, *"aaj weather kaisa rahega in Mumbai"*).
- **Licensing:** Apache-2.0.
- **Role in Corpus:** Placed exclusively in the **Validation Split** to measure model behavior on authentic Indian conversational inputs without contaminating training or test splits.

### 3.2 Authentic Classical Sanskrit (`wikipedia_sa_bhagavad_gita`)
- **Origin & Provenance:** Sourced from Sanskrit Wikisource/Wikipedia classical verse records.
- **Distinction from Hindi Articles:** Previous corpora possessed Hindi articles *about* Sanskrit literature (`wikipedia_hi_sanskrit_sahitya`). This new source family consists purely of classical Devanagari Sanskrit verses and authentic prose commentary, enabling true linguistic differentiation between Hindi and Sanskrit.
- **Licensing:** CC BY-SA 4.0 and Public Domain.
- **Role in Corpus:** Placed in the **Test Split** (61 documents, 20,244 tokens) to evaluate zero-shot cross-lingual transfer to ancient inflected classical languages.

### 3.3 Computer Science & Theoretical Mechanics (`wikipedia_en_turing_machine`)
- **Origin & Provenance:** Formal expository technical writing on Alan Turing's 1936 universal computing machine, state transitions, halting problem, and formal grammar.
- **Licensing:** CC BY-SA 4.0.
- **Role in Corpus:** Placed in the **Train Split** (74 documents, 16,678 tokens) to provide rigorous procedural and theoretical exposition.

### 3.4 Mathematics & Shannon Entropy (`wikipedia_en_information_theory`)
- **Origin & Provenance:** Detailed mathematical derivations of Shannon entropy, mutual information, channel capacity, and log-loss.
- **Licensing:** CC BY-SA 4.0.
- **Role in Corpus:** Placed in the **Test Split** (70 documents, 15,982 tokens) as an independent English mathematical benchmark, counterbalancing Loney's geometry prose.

### 3.5 Hindi Educational Chemistry (`wikipedia_hi_chemistry`)
- **Origin & Provenance:** Detailed scientific Devanagari text on chemical elements, periodic table, molecular bonding, and atomic theory (Rasayan Vigyan).
- **Licensing:** CC BY-SA 4.0.
- **Role in Corpus:** Placed in the **Train Split** (58 documents, 14,885 tokens) to provide non-literary, technical Hindi prose.

---

## 4. Split Distribution and Balance Analysis

Corpus v2 achieves a deterministic source-family disjoint split structure.

```
+-----------------------------------------------------------------------------------------+
|                                    CORPUS V2 (1,258 Docs)                               |
+------------------------------+------------------------------+---------------------------+
|          TRAIN SPLIT         |       VALIDATION SPLIT       |         TEST SPLIT        |
|      (782 Docs, 147k Toks)   |      (186 Docs, 32.6k Toks)  |    (290 Docs, 65.7k Toks) |
+------------------------------+------------------------------+---------------------------+
| • Gitanjali (en prose)       | • Bisect (code)              | • Loney Geometry (math)   |
| • RFC 793 (en technical)     | • Ganit (hi math)            | • Info Theory (en math)   |
| • Heapq (code)               | • Hinglish Top (hi-en conv)  | • Sanskrit Sahitya (hi)   |
| • Premchand (hi prose)       |                              | • Bhagavad Gita (sa)      |
| • Turing Machine (en cs)     |                              |                           |
| • Chemistry (hi technical)   |                              |                           |
+------------------------------+------------------------------+---------------------------+
```

### Test Split Balance Comparison (Corpus v1 vs Corpus v2)

| Metric | Corpus v1 (Batch 1941) | Corpus v2 (Batch 2021) | Resolution Impact |
|:---|:---:|:---:|:---|
| **Total Test Documents** | 1,943 | 290 | Capped redundant Loney coordinate exercises |
| **Total Test Tokens** | 338,983 | 65,665 | Clean, manageable, high-density evaluation set |
| **Dominant Source Share** | **98.6%** (Loney) | **37.5%** (Loney) | **Eliminated test monopoly** |
| **Unique Source Families** | 2 | **4** | Diversified across English, Hindi, and Sanskrit |
| **Languages Represented** | `en`, `hi` | `en`, `hi`, `sa` | Authentic classical Sanskrit added |
| **Domains Represented** | Math, Hindi Lit | Math, Info Theory, Hindi Lit, Sanskrit Classical | Multi-domain evaluation restored |

---

## 5. Ingestion and Cleaning Invariants

1. **Boilerplate Stripping:** Gutenberg legal disclaimers, license footers, Wikipedia navigation menus, edit notices, and category tags were deterministically stripped using delimiter parsing before tokenization.
2. **Deterministic Chunking:** Document chunking was standardized to 350+ character paragraph-preserving segments with a trailing threshold of 120 characters.
3. **No Synthetic Backfilling:** Zero template-generated sentences, repeated fillers, or fake citations were allowed. Every token is verified natural text.
4. **Exact Content Hashes:** Every document in `CORPUS_V2_MANIFEST.json` contains an immutable SHA-256 digest of its exact UTF-8 byte payload.

---

## 6. Conclusion

Corpus v2 provides the foundational data integrity required for meaningful language model training and evaluation. By eliminating the test concentration bottleneck and introducing authentic Hinglish, Sanskrit, and CS exposition, the repository now possesses a balanced, leakage-safe natural corpus.
