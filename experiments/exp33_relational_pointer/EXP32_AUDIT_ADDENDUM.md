# AUDIT ADDENDUM: EXPERIMENT 32 PROTOCOL, METRICS & ARCHITECTURE

**Date:** 2026-10-27
**Auditor:** Principal ML Research Engineer & Experimental Auditor
**Audit Target:** `experiments/exp32_pointer_abstention/`
**Classification:** `VALID_FOR_STATED_SCOPE`

---

## 1. Executive Summary
Experiment 32 was independently audited prior to the design and implementation of Experiment 33.
The audit verified:
1. **Deduplication and Split Disjointness:** Programmatic verification of all 8 JSONL split files confirmed strict zero cross-split prompt leakage ($Split_i \cap Split_j = \emptyset, \forall i \neq j$) and 0 internal duplicates across all 350 test instances and 160 training instances.
2. **Abstention Metric Verification:** Recomputed confusion matrix values match `RESULTS.json` and `EXP32_REPORT.md` bit-for-bit:
   - True Positives (Group F): 13 / 50 ($26.00\%$)
   - False Negatives (Group F): 37 / 50 ($74.00\%$)
   - False Positives (Groups A–E): 11 / 300 ($3.67\%$)
   - Precision: $54.17\%$
   - Recall: $26.00\%$
   - F1 Score: $0.3514$
3. **Distractor Error Distribution Verification:**
   - In Arm 1 (Standard Pointer), of 204 retrieval errors, 202 ($99.02\%$) were distractor selections, and 2 were hallucinations.
   - In Arm 2 (Abstention-Aware Pointer), of 212 retrieval errors, 201 ($94.81\%$) were distractor selections, 11 were false abstentions, and **0 were hallucinations**.
   - This mathematically confirms the hypothesis: **Distractor slot confusion is the overwhelming dominant failure mode ($>99\%$ of non-abstention errors) of single-head pointer attention.**
4. **Preregistration Hash & Lineage:** Pre-registered hash `b47f2fd7b950513e6e1e7517e9c05243fed1fd34916bbaaf0a8eeb5b5b843ccf` matches the frozen preregistration file. Checkpoints are strictly isolated in `checkpoints/`.

---

## 2. Audit Conclusion
Experiment 32 is classified as **`VALID_FOR_STATED_SCOPE`**. Its empirical findings—specifically that distractor selection accounts for virtually all retrieval errors and that missing-key abstention can be induced via balanced training—form a solid foundation for Experiment 33.
