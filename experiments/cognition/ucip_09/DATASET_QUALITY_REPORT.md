# UCIP-09: Dataset Quality, Lineage & Separation Report

**Project:** ChakrView  
**Experiment Branch:** `exp/cbr-01-conversational-breakthrough`  
**Date:** November 17, 2026  

---

## 1. Dataset Generation & Cryptographic Manifest

To address the severe data volume deficit identified in prior audits, we built a newly structured 3,600-sample conversational and capability corpus with **0% overlap** against all frozen benchmarks:

| Dataset Partition | File Path | Record Count | SHA-256 Digest |
|---|---|---|---|
| **Training Partition** | `experiments/cognition/ucip_09/data/ucip09_train.jsonl` | 3,240 | `2b251762e8216cfd956acdbd343e757e64bf2c33309cc24fa0f653e4ef37108d` |
| **Validation Partition** | `experiments/cognition/ucip_09/data/ucip09_validation.jsonl` | 360 | `711ec499fef6073dfe1ad602adc7524596344334db835d3e7c477be77db48beb` |

---

## 2. Category Distribution & Semantic Coverage

```
Capability Category               Sample Count   Percentage
------------------------------------------------------------
English Conversation & Identity            500        13.9%
English Conceptual Explanations            500        13.9%
Devanagari Hindi Multilingual              600        16.7%
Roman Hindi & Hinglish Technical           700        19.4%
Elementary Word-Problem Arithmetic         600        16.7%
Exact Verbatim Copying                     400        11.1%
Instruction Following & Constraints        300         8.3%
------------------------------------------------------------
TOTAL DATASET SIZE                       3,600       100.0%
```

---

## 3. Benchmark Contamination & Overlap Verification

- **Contamination Shield:** During dataset construction, all 500 prompts from `sealed_language_intelligence_500.json` and all 100 prompts from `sealed_owner_reality_100.json` (599 unique frozen stems) were loaded as hard assertions.
- **Verification Guarantee:** 0 prompts or targets in `ucip09_train.jsonl` overlap with the frozen benchmark suites.
- **Entity Diversity:** Arithmetic examples vary operands across 10 distinct nouns (apples, books, pens, coins, laptops, cards, candies, bottles), preventing the entity-entrainment trap observed in UCIP-06.
