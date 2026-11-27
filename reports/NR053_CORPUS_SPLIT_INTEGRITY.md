# NR-053: Corpus Split Integrity & Leakage Forensic Report

**Mission:** CHAKRVIEW NR-053 — Generalization, Data Leakage & Overfitting Forensic Validation  
**Date:** November 27, 2026  
**Auditor:** ChakrGenesis Independent Forensic Review Team  
**Artifact Path:** [corpus_v4_split_integrity.json](file:///D:/Project/ChakrView/experiments/forensics_nr053/corpus_v4_split_integrity.json)

---

## 1. Executive Summary

This forensic investigation audited document-level and token-level boundaries across `data/corpus_v4/` (Train vs Validation splits), `data/corpus_v3/` (SFT dataset), and the sealed evaluation benchmarks.

### Critical Forensic Discoveries
1. **Corpus V4 Total Token Clarification:**
   The documented total of **624,068 tokens** represents the sum of the training split (592,824 tokens) and the validation split (31,244 tokens). The training split contains 5,828 documents, and the validation split contains 306 documents.
2. **Train vs Validation Split Cleanliness:**
   - Exact document hash overlap: **0 / 306 (0.0% overlap)**.
   - Normalized text hash overlap: **0 / 306 (0.0% overlap)**.
   - Near-duplicate overlap (Jaccard > 0.8): **0 occurrences**.
   - Train and validation splits originate from the same 10 underlying raw foundation files, but document chunks were strictly partitioned with zero document sharing.
3. **The Paraphrase Leakage Discovery:**
   - Previous missions claimed 0.0% contamination based solely on strict exact string matching against raw prompts.
   - Forensic analysis revealed that **Corpus V3 contains training items with `(Ref #...)` suffixes and minor rephrasings of benchmark questions**:
     - *Benchmark EN-03:* `"Name the capital city of Australia."` $\longleftrightarrow$ *Corpus V3 Train:* `"What is the capital city of Australia? (Ref #15)"` $\to$ Response: `"The capital city of Australia is Canberra."`
     - *Benchmark HI-09:* `"भारत की राजधानी का नाम क्या है?"` $\longleftrightarrow$ *Corpus V3 Train:* `"भारत की राजधानी क्या है और यह किस नदी के तट पर स्थित है?"` $\to$ Response: `"भारत की राजधानी नई दिल्ली..."`
     - *Benchmark QA-07:* `"Which planet is closest to the Sun?"` $\longleftrightarrow$ *Corpus V3 Train:* `"Which planet is closest to the Sun? (Ref #12)"` $\to$ Response: `"Mercury is the closest planet to the Sun."`
     - *Benchmark QA-10:* `"How many continents are there on Earth?"` $\longleftrightarrow$ *Corpus V3 Train:* `"How many continents are there on Earth? (Ref #26)"` $\to$ Response: `"There are seven recognized continents on Earth."`
   - **Forensic Impact:** The "exact factual recall" observed in NR-052 for Canberra, New Delhi, and Mercury was **not ungrounded zero-shot reasoning**, but rather **successful conversational memorization and generalization across minor syntactic paraphrases** from Corpus V3 training data.
