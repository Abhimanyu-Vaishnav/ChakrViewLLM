# SANSKRIT CURRICULUM PREREGISTRATION AUDIT REPORT

**Batch:** 2501–2540  
**Audit Phase:** Phase 5 — Deep Scientific & Methodological Audit of Preregistration Protocol  
**Audited Documents:**  
- [`experiments/batch2461_2500/SANSKRIT_CURRICULUM_PREREGISTRATION.md`](file:///d:/Project/ChakrView/experiments/batch2461_2500/SANSKRIT_CURRICULUM_PREREGISTRATION.md)  
- [`experiments/batch2461_2500/SANSKRIT_CURRICULUM_PREREGISTRATION.json`](file:///d:/Project/ChakrView/experiments/batch2461_2500/SANSKRIT_CURRICULUM_PREREGISTRATION.json)  
**Machine-Readable Audit Record:** [`experiments/batch2501_2540/PREREGISTRATION_AUDIT.json`](file:///d:/Project/ChakrView/experiments/batch2501_2540/PREREGISTRATION_AUDIT.json)  
**Required Action Item Specification:** [`experiments/batch2501_2540/PREREGISTRATION_REQUIRED_CHANGES.md`](file:///d:/Project/ChakrView/experiments/batch2501_2540/PREREGISTRATION_REQUIRED_CHANGES.md)  
**Protocol Classification:** **REQUIRES REVISION (CONDITIONAL EXPERIMENT READINESS)**  

---

## 1. Executive Summary & Audit Mandate

Phase 5 requires an exhaustive audit of the preregistration protocol formulated in Batch 2461–2500. Mere existence of a preregistration file is insufficient; the experimental design, leakage detection guarantees, starting weight lineages, and decision thresholds must satisfy rigorous scientific standards.

Our audit identified several high-quality design principles in the Batch 2461 protocol, but also uncovered **critical ambiguities and methodological vulnerabilities** that require formal revision before any training run can be authorized.

---

## 2. Strengths of the Current Protocol
1. **Four-Arm Factorial Isolation:** Explicitly defines Arm 0 (Baseline), Arm 1 (Seed 42 Reference), Arm 2 (Sanskrit-Augmented CPT), and Arm 3 (Volume-Matched Control), properly distinguishing Sanskrit-specific semantic transfer from generic compute/token scaling.
2. **Fixed Invariants:** Freezes architecture (ChakrMicro, 3.44M params), sequence window (256), optimizer (AdamW, cosine schedule, 50 warmup steps), and gradient accumulation settings.
3. **Multi-Seed Variance Policy:** Mandates a 3-seed evaluation (`seeds=[42, 43, 44]`) to account for initialization noise and empirical standard deviation.
4. **Mandatory Reporting:** Requires reporting all runs including failed or inconclusive runs without selective pruning.

---

## 3. Identified Deficiencies & Methodological Vulnerabilities

### A. Leakage Gate Limitation (8-Gram / MinHash Screening Heuristic)
- **Deficiency:** The protocol claims that zero exact 8-gram matches and MinHash Jaccard $< 0.15$ guarantees zero leakage.
- **Scientific Reality:** An 8-gram or MinHash check is an approximate *screening criterion*, not a mathematical proof of zero semantic leakage. In Classical Sanskrit, morphological sandhi (euphonic vowel/consonant fusion) and inflected compounds can express identical verses with variations in segmentation that evade superficial 8-gram exact matches.
- **Required Revision:** The leakage protocol must specify **Devanagari Unicode NFKC normalization**, removal of virama/accent markers, and word-level lemmatization prior to n-gram screening. Furthermore, any similarity trigger must require human linguistic review rather than automatic clearance.

### B. Ambiguity in Seed 42 Candidate Lineage
- **Deficiency:** The protocol describes Arm 1 as the "existing CPT Seed 42 candidate", but is ambiguous regarding whether Arm 2 trains *from* the Seed 42 checkpoint or branches *from* the canonical step 10,000 baseline.
- **Scientific Reality:** If Arm 2 continues from Seed 42, it receives double the total training steps (1000 steps vs 500 steps) compared to Arm 1. To maintain a valid controlled experiment, Arm 2 and Arm 3 must initialize from the canonical step 10,000 baseline checkpoint (`96357705a683...`) and train for identical steps (500 steps).
- **Required Revision:** Explicitly specify that Arm 2 and Arm 3 branch directly from the canonical step 10,000 checkpoint.

### C. Underspecified Provenance & Tokenization for New Sanskrit Data
- **Deficiency:** The protocol mentions "public domain Classical Sanskrit corpora completely disjoint from Wikipedia/Bhagavad Gita", but does not specify the exact text source, licensing, sentence boundary handling, or whether the existing BPE vocabulary contains sufficient subword coverage for the new vocabulary.
- **Required Revision:** Require a pre-training vocabulary coverage audit measuring unigram token fragmentation on the proposed Sanskrit corpus before freezing the training manifest.

### D. Threshold Arbitrariness in Decision Rules
- **Deficiency:** Thresholds such as $\text{TW-CE}_{\text{sa}} \le 4.3400$ were chosen based on point estimates rather than statistical confidence intervals derived from seed standard deviations.
- **Required Revision:** Base the acceptance threshold on achieving a statistically significant reduction ($\ge 2\sigma$) below Candidate Seed 42 without degrading control domains by more than $1\sigma$.

---

## 4. Preregistration Audit Classification

- **Status:** **`REQUIRES_REVISION`**
- **Readiness:** The experiment protocol cannot be approved as final in its current state. The revisions specified in [`PREREGISTRATION_REQUIRED_CHANGES.md`](file:///d:/Project/ChakrView/experiments/batch2501_2540/PREREGISTRATION_REQUIRED_CHANGES.md) must be incorporated before human experiment authorization.
