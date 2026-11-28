# NR-057-R1 LICENSING & SOURCE PROVENANCE AUDIT REPORT

**Date:** November 28, 2026  
**Mission:** NR-057-R1 Corpus Expansion Completion & Training Gate Revalidation  
**Artifact ID:** `NR057R1_LICENSE_AUDIT`  
**License Audit JSON:** [license_audit.json](file:///d:/Project/ChakrView/experiments/corpus_nr057_r1/license_audit.json)

---

## 1. Executive Summary

In adherence to non-negotiable mission constraints, all training tokens utilized in ChakrView must possess transparent, verifiable, and legally permissive licensing. No copyrighted books, unauthorized scrapings, or proprietary API outputs are tolerated.

Every source component in the current and expanding training pool has been subjected to a legal rights audit:
- **Commercial Use Rights:** Confirmed permitted across all sources.
- **Redistribution Rights:** Confirmed permitted with standard attribution.
- **Quarantined Sources:** **0 sources quarantined.** All 5 component pipelines are verified.

---

## 2. Per-Source Provenance & Legal Classification

| Source Identifier | Official Origin / Repository | Exact License | Commercial Use | Redistribution | Collection Methodology | Token Yield | Audit Verdict |
|---|---|---|---|---|---|---|---|
| **Python Standard Library** | [python/cpython](https://github.com/python/cpython) | **PSF License 2.0** (OSI approved) | **Permitted** | **Permitted** (with notice) | Structural AST segmentation of standard library modules | 1,510,114 | **APPROVED** |
| **English Wikipedia Expository** | [en.wikipedia.org](https://en.wikipedia.org) | **CC-BY-SA 4.0 & GFDL** | **Permitted** | **Permitted** (with attribution) | MediaWiki REST API text extracts (`prop=extracts`) | 836,047 | **APPROVED** |
| **Hindi Wikipedia Devanagari** | [hi.wikipedia.org](https://hi.wikipedia.org) | **CC-BY-SA 4.0 & GFDL** | **Permitted** | **Permitted** (with attribution) | MediaWiki REST API Hindi extracts | 171,548 | **APPROVED** |
| **Local Raw Foundations** | `data/raw/` | **Apache 2.0 & Public Domain** | **Permitted** | **Permitted** | Curated human engineering and public domain texts | 500,109 | **APPROVED** |
| **Corpus V4 Curated Clean** | `data/corpus_v4/` | **Apache 2.0 / Project Internal** | **Permitted** | **Permitted** | Deduplicated, contamination-filtered foundational pairs | 547,488 | **APPROVED** |

---

## 3. Strict Prohibitions Enforced

1. **No External LLM Weights:** No parameter weights from Llama, GPT, Mistral, Gemma, or BERT are present or utilized as teachers.
2. **No External LLM APIs:** Zero tokens originated from OpenAI, Anthropic, or Google Gemini APIs.
3. **No Unlicensed Web Scrapes:** Bulk Common Crawl data is excluded due to copyright ambiguity, spam, and PII risks.
