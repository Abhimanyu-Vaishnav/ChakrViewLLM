# CHAKRVIEW — Batches 1981–2020
## Provenance & Licensing Verification Audit

### Executive Summary

To comply with ChakrView's strict data governance standards, every raw text source in the staged natural corpus was audited against its canonical repository, copyright duration, open-source license terms, and attribution requirements.

---

### 1. Detailed Provenance & License Verification Register

| Source Identifier | Canonical Source URL | Author / Responsible Organization | License Claim | License Evidence & Verification | Attribution Requirements | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`gutenberg_tagore_gitanjali`** | `https://www.gutenberg.org/ebooks/7164` | Rabindranath Tagore (1861–1941) | **Public Domain** | Published 1913; author died 1941 (>60 years post-mortem in India; >70 years in US/EU). In public domain worldwide. Gutenberg license applies. | Retain Project Gutenberg header/attribution where full texts are redistributed. | **VERIFIED (Approved)** |
| **`gutenberg_loney_coordinate_geometry`** | `https://www.gutenberg.org/ebooks/43379` | S. L. Loney (1860–1938) | **Public Domain** | First published 1895 by Macmillan and Co.; author died 1938 (>80 years post-mortem). Free public domain worldwide. | Standard Gutenberg terms. | **VERIFIED (Approved)** |
| **`cpython_algorithms_bisect`** | `https://github.com/python/cpython/blob/main/Lib/bisect.py` | Python Software Foundation | **PSF License 2.0** | Permissive open source license authorizing reproduction, distribution, and adaptation. | Retain PSF copyright notice in documentation. | **VERIFIED (Approved)** |
| **`cpython_algorithms_heapq`** | `https://github.com/python/cpython/blob/main/Lib/heapq.py` | Python Software Foundation | **PSF License 2.0** | Permissive open source license authorizing reproduction, distribution, and adaptation. | Retain PSF copyright notice in documentation. | **VERIFIED (Approved)** |
| **`ietf_rfc793_tcp`** | `https://www.rfc-editor.org/rfc/rfc793.txt` | Information Sciences Institute / Jon Postel | **IETF Public Domain** | Standard RFC published September 1981 by DARPA / IETF. Free public specification document. | Reference RFC 793 title and editor. | **VERIFIED (Approved)** |
| **`wikipedia_hi_premchand`** | `https://hi.wikipedia.org/wiki/%E0%A4%AA%E0%A5%8D%E0%A4%B0%E0%A5%87%E0%A4%AE%E0%A4%9A%E0%A4%82%E0%A4%A6` | Hindi Wikipedia Contributors | **CC BY-SA 4.0** | Creative Commons Attribution-ShareAlike 4.0 International. Permitted for AI training and text analysis. | Attribute Wikipedia contributors and provide CC BY-SA 4.0 license notice. | **VERIFIED (Approved)** |
| **`wikipedia_hi_ganit`** | `https://hi.wikipedia.org/wiki/%E0%A4%97%E0%A4%A3%E0%A4%BF%E0%A4%A4` | Hindi Wikipedia Contributors | **CC BY-SA 4.0** | Creative Commons Attribution-ShareAlike 4.0 International. | Attribute Wikipedia contributors. | **VERIFIED (Approved)** |
| **`wikipedia_hi_sanskrit_sahitya`** | `https://hi.wikipedia.org/wiki/%E0%A4%B8%E0%A4%82%E0%A4%B8%E0%A5%8D%E0%A4%95%E0%A5%83%E0%A4%A4_%E0%A4%B8%E0%A4%BE%E0%A4%B9%E0%A4%BF%E0%A4%A4%E0%A5%8D%E0%A4%AF` | Hindi Wikipedia Contributors | **CC BY-SA 4.0** | Creative Commons Attribution-ShareAlike 4.0 International. | Attribute Wikipedia contributors. *(Flagged for language mislabeling: text is Hindi, not Sanskrit).* | **VERIFIED (Approved with Label Correction)** |

---

### 2. Attribution & Data Governance Policy Compliance

1. **All 8 Raw Sources Legally Permitted:** Every source operates under verifiable public domain status, permissive open-source licensing (PSF-2.0), or open educational licensing (CC BY-SA 4.0).
2. **Absence of Web-Scraped Unlicensed Text:** No scraped copyrighted books (e.g., Books3, BookCorpus, unvetted crawl dumps) are included in this corpus.
3. **Traceability:** Every document record in `CORPUS_MANIFEST.json` contains its parent `source_id`, verifiable author/organization, and content SHA-256 hash.
