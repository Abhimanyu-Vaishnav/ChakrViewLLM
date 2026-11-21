# UCIP-06 Dataset Audit Report

## 1. Overview
This audit inspects the curated foundational dataset created for UCIP-06 to ensure high semantic diversity, authentic multilinguality, balanced topic representation, and absolute zero prompt leakage against the sealed 500-probe benchmark.

- **Authoritative Train Path:** `experiments/cognition/ucip_06/data/ucip06_train.jsonl`
- **Train SHA-256:** `75d436ec77ee39c4a4f89d31d451fe836e5ea9070db79fbbce1434c3821033a3`
- **Authoritative Val Path:** `experiments/cognition/ucip_06/data/ucip06_validation.jsonl`
- **Validation SHA-256:** `6b77dcf0f0cbeb062c3f81e3a24fe721f58dfb4bcbfbebaaa465e929d29ebf77`
- **Total Samples:** 1,838 (Train: 1,562 [85%], Validation: 276 [15%])

---

## 2. Quantitative Distributions

### A. Category & Task Distribution
| Category | Train Count | Val Count | Total | Percentage |
|---|---|---|---|---|
| Elementary Arithmetic | 678 | 122 | 800 | 43.5% |
| Science & Concept Explanations | 215 | 45 | 260 | 14.1% |
| Exact Copying | 204 | 36 | 240 | 13.1% |
| Devanagari Hindi | 108 | 20 | 128 | 7.0% |
| Hinglish | 102 | 18 | 120 | 6.5% |
| English Conversation | 98 | 17 | 115 | 6.3% |
| Roman Hindi | 82 | 13 | 95 | 5.2% |
| English Sentence Generation | 45 | 5 | 50 | 2.7% |
| Basic Instructions | 30 | 0 | 30 | 1.6% |
| **Total** | **1,562** | **276** | **1,838** | **100.0%** |

### B. Target Uniqueness & Repetition Metrics
- **Unique Prompts:** 1,838 / 1,838 (100.0% prompt uniqueness)
- **Unique Targets:** 1,374 / 1,838 (74.8% target uniqueness)
  - Arithmetic produces structured target answers across combinations (e.g., `"The sum is 43."`, `"You now have 12 coins in total."`).
  - No single target comprises more than 0.8% of the total dataset.
  - In comparison, UCIP-05 had identical boilerplate targets covering over 15% of records.

### C. Length Distribution
- **Prompt Token Length:** Mean 14.8 tokens (Min: 6, Max: 38)
- **Target Token Length:** Mean 16.2 tokens (Min: 3, Max: 52)
- **Combined Context Length:** Well within the 256/512 ceiling of ChakrMicro and ChakrMedium.

---

## 3. Benchmark Contamination & Overlap Verification
- **Sealed Benchmark Size:** 500 probes (`sealed_500_benchmark.json`)
- **Sealed Benchmark SHA-256:** `554d45f9f7ad8e0a09710396604791c64bcfd1dc4f3d1019e268d719a83278cf`
- **Exact Prompt Overlap:** **0 / 500 (0.00% overlap)**
- **Target Leakage:** All benchmark targets were excluded from dataset generation scripts.

---

## 4. Qualitative Assessment
1. **Linguistic Balance:** Genuine Devanagari Hindi, Roman Hindi, and modern colloquial Hinglish are represented with natural phrasing rather than machine-translated artifacts.
2. **Concept Explanations:** Over 50 distinct science concepts across physics, chemistry, biology, and computing with clear educational definitions.
3. **Autoregressive Alignment:** Prompts are uniformly formatted as `User: <input>\nAssistant:` with single-space prepended targets (` <target>`) matching tokenizer BPE prefix rules.
