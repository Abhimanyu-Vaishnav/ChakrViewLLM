# PREREGISTRATION AMENDMENT DIFF REPORT

**Batch:** 2541–2580  
**Phase:** Phase 3 — Detailed Audit Diff of Protocol Amendments  
**Prior Specification:** [`experiments/batch2461_2500/SANSKRIT_CURRICULUM_PREREGISTRATION.md`](file:///d:/Project/ChakrView/experiments/batch2461_2500/SANSKRIT_CURRICULUM_PREREGISTRATION.md)  
**Amended Specification:** [`experiments/batch2541_2580/SANSKRIT_CURRICULUM_PREREGISTRATION_AMENDED.md`](file:///d:/Project/ChakrView/experiments/batch2541_2580/SANSKRIT_CURRICULUM_PREREGISTRATION_AMENDED.md)  

---

## Comparative Section-by-Section Diff

### 1. Checkpoint Lineage Invariants
```diff
- Arm 2: Proposed Sanskrit-augmented CPT. (Starting checkpoint implied or ambiguous relative to Seed 42).
+ Arm 2: Starts strictly from Step 10,000 baseline (experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt, SHA-256: 96357705a683...).
+ Guaranteed identical 500-step budget across Arm 1, Arm 2, and Arm 3 to prevent training duration confounding.
```

### 2. Leakage Screening Protocol
```diff
- Exact Match Check: 0 identical 8-gram matches. MinHash Jaccard < 0.15. Described as proof of zero leakage.
+ Three-Tier Screening Pipeline:
+ Tier 1: Exact 8-gram token sequence check (0 matches allowed).
+ Tier 2: Unicode NFKC normalization, punctuation/virama stripping, 8-gram Jaccard < 0.05.
+ Tier 3: Subword MinHash LSH 13-gram Jaccard < 0.10.
+ Explicitly classified as a screening heuristic, requiring mandatory linguistic review for any flagged passage.
```

### 3. Tokenizer Preflight & Subword Fragmentation
```diff
- (No tokenizer fragmentation audit defined).
+ Mandatory Preflight Specification:
+ Requires Characters-per-Token (CPT) compression ratio >= 1.80 chars/token on candidate Sanskrit text.
+ Rejects candidate corpora exhibiting high byte-fallback or excessive character fragmentation.
```

### 4. Decision Rules & Statistical Rigor
```diff
- Static Point Estimate: TW-CE_sa <= 4.3400.
+ Variance-Aware Statistical Hypothesis Testing:
+ Evaluates Mean(TW-CE_sa) across seeds [42, 43, 44].
+ Success requires improvement >= 1.96 * sigma_sa below Candidate Seed 42 without control domain degradation > 1.0 * sigma.
```

---

## Conclusion
The amended protocol resolves every defect identified in the Phase 1 and Batch 2501 audits, elevating the experimental design to full scientific readiness.
