# CHAKRVIEW — Batches 1941–1980
## Natural Corpus Provenance, Licensing & Acquisition Specification

### Executive Summary

To establish an uncompromised foundation for future language modeling and continued-pretraining experiments, Batches 1941–1980 assembled the **ChakrView Natural Text Corpus v1**.

Every source in this corpus was acquired from authoritative public repositories, with verifiable legal provenance, clear licensing, recorded source hashes, and strict source-level group isolation across splits.

---

### 1. Ingested Sources Catalog & Provenance Register

| Source Identifier | Title & Author | Domain | Language | License / Basis | Canonical URL / Origin | Raw SHA256 (First 16 chars) | Raw Characters |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`gutenberg_tagore_gitanjali`** | *Gitanjali (Song Offerings)*<br>Rabindranath Tagore | Prose / Literature | English (`en`) | **Public Domain**<br>(Gutenberg #7164, Author died 1941) | [Project Gutenberg](https://www.gutenberg.org/cache/epub/7164/pg7164.txt) | `665ea9b282c0a542` | 97,888 |
| **`gutenberg_loney_coordinate_geometry`** | *The Elements of Coordinate Geometry*<br>S. L. Loney | Mathematics / Geometry | English (`en`) | **Public Domain**<br>(Gutenberg #43379, Published 1895) | [Project Gutenberg](https://www.gutenberg.org/cache/epub/43379/pg43379.txt) | `0ad94ad6d8f38b23` | 762,190 |
| **`cpython_algorithms_bisect`** | *CPython Standard Library: `bisect.py`*<br>Python Software Foundation | Code / Algorithms | Python (`code`) | **PSF License 2.0**<br>(Permissive Open Source) | [GitHub CPython](https://raw.githubusercontent.com/python/cpython/main/Lib/bisect.py) | `f1cf7b85fc36b5da` | 3,423 |
| **`cpython_algorithms_heapq`** | *CPython Standard Library: `heapq.py`*<br>Python Software Foundation | Code / Algorithms | Python (`code`) | **PSF License 2.0**<br>(Permissive Open Source) | [GitHub CPython](https://raw.githubusercontent.com/python/cpython/main/Lib/heapq.py) | `70210fab9de1fef4` | 23,397 |
| **`ietf_rfc793_tcp`** | *RFC 793: Transmission Control Protocol*<br>DARPA / Jon Postel | Technical Systems | English (`en`) | **IETF Public Domain**<br>(Standard Specification, 1981) | [RFC Editor](https://www.rfc-editor.org/rfc/rfc793.txt) | `e55b1faa35edbeec` | 172,710 |
| **`wikipedia_hi_premchand`** | *Munshi Premchand Biography & Literary Overview*<br>Wikipedia Contributors | Hindi Literature | Hindi (`hi`) | **CC-BY-SA-4.0**<br>(Creative Commons Attribution) | [Hindi Wikipedia](https://hi.wikipedia.org/w/api.php?action=query&prop=extracts&explaintext=1&titles=%E0%A4%AA%E0%A5%8D%E0%A4%B0%E0%A5%87%E0%A4%AE%E0%A4%9A%E0%A4%82%E0%A4%A6&format=json) | `3ff5440ba385eff4` | 18,930 |
| **`wikipedia_hi_ganit`** | *Mathematics (Ganit) Historical Overview*<br>Wikipedia Contributors | Mathematics / History | Hindi (`hi`) | **CC-BY-SA-4.0**<br>(Creative Commons Attribution) | [Hindi Wikipedia](https://hi.wikipedia.org/w/api.php?action=query&prop=extracts&explaintext=1&titles=%E0%A4%97%E0%A4%A3%E0%A4%BF%E0%A4%A4&format=json) | `1e5ffff5916b1315` | 10,183 |
| **`wikipedia_hi_sanskrit_sahitya`** | *Sanskrit Literature (Sahitya) Overview*<br>Wikipedia Contributors | Sanskrit Studies | Hindi-Sanskrit (`hi-sa`) | **CC-BY-SA-4.0**<br>(Creative Commons Attribution) | [Hindi Wikipedia](https://hi.wikipedia.org/w/api.php?action=query&prop=extracts&explaintext=1&titles=%E0%A4%B8%E0%A4%82%E0%A4%B8%E0%A5%8D%E0%A4%95%E0%A5%83%E0%A4%A4_%E0%A4%B8%E0%A4%BE%E0%A4%B9%E0%A4%BF%E0%A4%A4%E0%A5%8D%E0%A4%AF&format=json) | `ae02ef489823f7ad` | 10,392 |

---

### 2. Preprocessing & Quality Controls

1. **Header and Footer Removal:**
   - Gutenberg legal notices and transcriber notes were programmatically sliced off using exact boundary markers (`*** START OF` and `*** END OF`).
2. **Deterministic Chunking & Deduplication:**
   - Text was wrapped and chunked into coherent passage-level documents (~350–500 characters).
   - Exact document-level SHA256 hashing was enforced; zero duplicates were permitted.
3. **Absence of Synthetic Fallbacks:**
   - The corpus builder contains **no fallback while-loops** or procedural string generators. If a domain or source reaches its natural end, ingestion stops.

---

### 3. Split Isolation Strategy

To completely eliminate intra-source proximity leakage (the primary flaw of Batch 1861), the split assignment enforces **strict source-level group isolation**:
- Every source belongs to **exactly one split**. No single source file is ever split across train, validation, or test.

| Split | Assigned Sources | Document Count | Token Count | Primary Roles |
| :--- | :--- | :--- | :--- | :--- |
| **Train** | `gutenberg_tagore_gitanjali`<br>`ietf_rfc793_tcp`<br>`cpython_algorithms_heapq`<br>`wikipedia_hi_premchand` | **650** | **115,913** | English literature, technical specifications, Python algorithms, Hindi prose |
| **Validation** | `cpython_algorithms_bisect`<br>`wikipedia_hi_ganit` | **26** | **5,948** | Clean checkpoint selection & hyperparameter tuning |
| **Test** | `gutenberg_loney_coordinate_geometry`<br>`wikipedia_hi_sanskrit_sahitya` | **1,943** | **334,092** | Completely unshared source-disjoint final capability evaluation |

*Corpus catalog and document manifests are serialized in [CORPUS_MANIFEST.json](file:///d:/Project/ChakrView/experiments/batch1941_1980/data/CORPUS_MANIFEST.json) and [SPLIT_MANIFEST.json](file:///d:/Project/ChakrView/experiments/batch1941_1980/data/SPLIT_MANIFEST.json).*
