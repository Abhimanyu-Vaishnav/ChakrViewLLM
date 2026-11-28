# NR-057 BENCHMARK CONTAMINATION & PROTECTION AUDIT REPORT

**Date:** November 28, 2026  
**Mission:** NR-057 Clean Corpus Expansion & Training Readiness Engineering  
**Artifact ID:** `NR057_BENCHMARK_PROTECTION_AUDIT`  
**Exclusion Registry JSON:** [contamination_exclusion_registry.json](file:///d:/Project/ChakrView/experiments/corpus_nr057/contamination_exclusion_registry.json)

---

## 1. Executive Summary

Forensic audits in NR-053 and NR-056 revealed that historical models suffered from benchmark contamination:
- In NR-056, **40% of Benchmark V2 evaluation prompts** had exact or near-exact n-gram overlap with historical pretraining corpora, creating an illusion of conversational competence.
- When evaluated against the sealed, truly uncontaminated **Benchmark V3** (300 test items across 7 cognitive domains), all candidate models scored $\le 1.22\%$ (effectively 0.0% on reasoning and extraction).

To guarantee scientific validity for all subsequent training runs, NR-057 has instituted an immutable **Benchmark Contamination Firewall**. No document will enter `Corpus V5` if it exhibits 8-gram token overlap or substring leakage with any historical or sealed benchmark test cases.

---

## 2. Protected Benchmark Inventory

The Benchmark Contamination Firewall loaded, parsed, normalized, and extracted protected feature sets from all active benchmark evaluation assets:

| Benchmark Asset | Path | Protected Items | Extracted 8-Grams | Domain Coverage |
|---|---|---|---|---|
| **Benchmark V3 (Sealed)** | `data/evaluation/benchmark_v3/benchmark_v3_300.json` | 300 | 7,853 | Arithmetic, Syllogism, Negation, Reading Extraction, Python Code, Multilingual Hindi/Hinglish, Common Sense |
| **Benchmark V2 (Audited)** | `data/evaluation/benchmark_v2/benchmark_v2_100.json` | 95 | 1,503 | Multi-turn reasoning, factual recall, identity, entity binding |
| **NR-053 Fresh Heldout** | `experiments/forensics_nr053/fresh_heldout_benchmark_50.json` | 50 | 1,500 | Independent heldout generalization queries |
| **Total Protected Set** | **All Sealed Test Sets** | **445** | **6,116 unique** | **Complete repository test surface** |

---

## 3. Firewall Algorithm & Exclusion Mechanism

During Corpus V5 ingestion, every candidate document undergoes a dual-layer contamination audit:
1. **Token 8-Gram Intersection:** The candidate document is tokenized using the canonical BPE 4096 tokenizer. Every sliding window of 8 consecutive tokens ($[t_i, \dots, t_{i+7}]$) is matched against the set of 6,116 protected test n-grams. Any document exhibiting non-zero intersection with test-specific prompts or solutions is immediately discarded.
2. **Normalized String Containment:** Candidate documents are checked against normalized benchmark questions ($>25$ characters) to prevent paraphrased or token-split leakage.
3. **Exclusion Logging:** Any rejected document is recorded in the contamination audit registry with its file of origin, document index, and matching n-gram signature.
