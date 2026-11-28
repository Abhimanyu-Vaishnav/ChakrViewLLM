# NR-056: BENCHMARK V2 CONTAMINATION & GENERALIZATION AUDIT

**Mission Identifier:** NR-056  
**Date:** November 28, 2026  
**Auditor:** ChakrView Independent Forensic Engineering Team  
**Scope:** Systematic search of all 95 Benchmark V2 prompts and target answers across all historical training corpora (Corpus V1, V2, V3, V4, and NR-051).

---

## 1. Executive Summary

A core question of NR-056 is whether Benchmark V2 tested genuine zero-shot generalization or whether the non-zero capability scores (e.g., 6.0% for NR-052 38M) were driven by memorization of training facts.

Every prompt and answer entity was audited against 29,268 historical training records using 4-gram matching and entity co-occurrence detection.

### Key Finding:
**All non-zero factual hits in Benchmark V2 were directly exposed in the training corpora:**
- `FQA-01` (Tokyo, capital of Japan): Confirmed near-duplicate in Corpus V1 & V3.
- `FQA-02` (Pacific Ocean): Confirmed near-duplicate in Corpus V1.
- `FQA-03` (Paris, capital of France): Factual co-occurrence in Corpus V3.
- `FQA-10` (Canberra, capital of Australia): Factual co-occurrence in Corpus V1.
- `FQA-14` (British Pound, currency of UK): Confirmed exact match in Corpus V3.

Across the entire 95-prompt suite, **zero questions outside the scope of training facts were answered correctly**. The models exhibited zero zero-shot generalization on genuinely novel entities or unexposed arithmetic problems.

---

## 2. Contamination Classification Breakdown

| Classification Category | Item Count | Percentage | Definition |
| :--- | :---: | :---: | :--- |
| **CONFIRMED_EXACT** | 6 | 6.3% | Exact prompt text appears verbatim in training data |
| **CONFIRMED_NEAR_DUPLICATE** | 4 | 4.2% | Prompt matches training sample with $\ge 75\%$ 4-gram overlap |
| **FACTUAL_EXPOSURE** | 38 | 40.0% | Prompt entity and answer entity co-occur in training text |
| **ANSWER_ONLY_OVERLAP** | 23 | 24.2% | Target answer appears in training text without prompt context |
| **NO_MATCH_FOUND** | 24 | 25.3% | Unseen prompt and unexposed question pattern |
| **Total** | **95** | **100.0%** | |

---

## 3. Scientific Impact on Capability Claims

1. **Benchmark V2 is NOT Uncontaminated:** Although authored independently without copying from other LLMs, the author chose common world knowledge and conversational queries that overlapped heavily with the factual QA datasets included in Corpus V1 and Corpus V3.
2. **Capability Modeled is Memorization / Retrieval, Not Generalization:** The fact that models only scored points on questions with confirmed training exposure proves that the models are operating as statistical retrieval systems for high-frequency training facts, rather than reasoning engines with general world models.
