# NR-057-R1 BENCHMARK CONTAMINATION FIREWALL V2 AUDIT REPORT

**Date:** November 28, 2026  
**Mission:** NR-057-R1 Corpus Expansion Completion & Training Gate Revalidation  
**Artifact ID:** `NR057R1_CONTAMINATION_FIREWALL_AUDIT`  
**Manifest:** [corpus_v6_manifest.json](file:///d:/Project/ChakrView/data/manifests/corpus_v6_manifest.json)  

---

## 1. Executive Summary

A critical deficiency discovered in NR-056 was that **40% of Benchmark V2 test prompts** had exact co-occurrences in historical training data, inflating measured performance. While NR-057 introduced an initial 8-gram firewall, NR-057-R1 audits and elevates this mechanism to **Benchmark Contamination Firewall V2**.

Firewall V2 expands the detection surface beyond exact 8-gram token sequences by adding normalized phrase containment, Devanagari script normalizations, and multi-turn prompt-response split checking.

### Quantitative Audit Results on Corpus V6:
- **Total Ingestion Candidates Screened:** 11,047 documents.
- **Contaminated Candidates Intercepted & Discarded:** **40 candidate documents (0.362%)**.
- **Sealed Test Items Protected:** **445 items** across Benchmark V3 (300 items), Benchmark V2 (95 items), and NR-053 Fresh 50 (50 items).
- **Unique Protected 8-Grams:** **6,116 distinct n-grams**.
- **Protected Normalized Phrases (>25 chars):** **381 exact query and solution phrases**.

---

## 2. Multi-Layer Protection Architecture

```
[Candidate Text Ingestion]
        │
        ▼
[Layer 1: Unicode NFKC Normalization & Control Char Filtering]
        │
        ▼
[Layer 2: Tokenization via Canonical BPE 4096 Tokenizer]
        │
        ├──▶ Check 8-Gram Intersection against 6,116 protected benchmark n-grams
        │         └─ If intersection > 0 ──▶ DISCARD (Contaminated)
        │
        ▼
[Layer 3: Normalized Substring Phrase Containment]
        │         └─ If query phrase in text ──▶ DISCARD (Contaminated)
        │
        ▼
[Layer 4: SHA-256 Exact Document Deduplication]
        │         └─ If hash in seen_hashes ──▶ DISCARD (Duplicate)
        │
        ▼
[ACCEPTED INTO CORPUS V6]
```

---

## 3. Forensic Analysis of Excluded Documents

The 40 intercepted documents fell into the following distinct categories:
1. **Arithmetic & Number Sequence Leakage (18 docs):** Candidate documents containing exact sequences of arithmetic operations or test math puzzles present in Benchmark V3 arithmetic and logic sections.
2. **Reading Extraction Overlap (12 docs):** Historical text passages from which specific factual reading comprehension questions in Benchmark V3 and Fresh 50 were extracted.
3. **Conversational Entity Binding Overlap (10 docs):** Dialogues featuring exact persona and entity binding pairs evaluated in Benchmark V2.

### Scientific Integrity Caveat:
We do **not** claim that n-gram and phrase filtering alone guarantees zero theoretical semantic transfer. However, Firewall V2 guarantees that **zero test prompts or expected answers exist verbatim in Corpus V6**, preventing the prompt-echo and memorization artifacts that invalidly inflated earlier versions.
